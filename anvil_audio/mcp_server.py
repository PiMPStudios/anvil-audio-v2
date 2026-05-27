"""
Anvil Audio MCP server.

Exposes Anvil's generation, editing, and registry capabilities to MCP
clients (Claude Desktop, Claude Code, custom agents) over stdio transport.

Usage — run directly::

    python -m anvil_audio.mcp_server

Or reference this file in an MCP config (see README for examples).

Models are loaded lazily on first use and cached between calls.
No Gradio or CLI argparser dependency — only anvil_audio core.
"""

from __future__ import annotations

import sys

# Suppress upstream deprecation noise before heavy imports.
try:
    from anvil_audio._warning_filters import apply_filters

    apply_filters()
except ImportError:
    pass

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from anvil_audio.core.registry import registry
from anvil_audio.core.output import GenerationMetadata, OutputManager
from anvil_audio.utils.memory import (
    cleanup_if_memory_pressure,
    estimate_values_size_mb,
    flush_memory_caches,
    memory_pressure_status,
    mlx_memory_status,
    process_rss_mb,
    system_memory_status,
    torch_memory_status,
)
from anvil_audio.utils.stdio_guard import stdout_to_stderr


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "anvil-audio",
    instructions=(
        "Anvil Audio: AI audio generation and post-processing. "
        "Use generate_audio for sound effects or music, edit_audio to post-process, "
        "and list_models to see what's available. "
        "For sound effects / ambient audio use a Stable Audio model. "
        "For full songs with lyrics or style tags use an ACE-Step model. "
        "If you're unsure which model to use, omit the model argument and let Anvil choose."
    ),
)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Loaded pipeline cache — avoids reloading the same model between calls.
# Key: (registry model name, ACE-Step MLX DiT override).  Value: BasePipeline
# instance.  The override matters because ACE-Step LoRA requires the PyTorch
# DiT path, while normal Apple Silicon generation may prefer native MLX DiT.
_pipeline_cache: dict[tuple[str, bool | None], Any] = {}
_pipeline_last_used: dict[tuple[str, bool | None], float] = {}

# Default project applied when project="" is passed to any tool.
_active_project: str = ""

# Keywords that suggest music-style prompts (prefer ACE-Step when available).
_MUSIC_KEYWORDS = frozenset(
    {
        "music",
        "song",
        "track",
        "melody",
        "beat",
        "bpm",
        "chord",
        "lyrics",
        "verse",
        "chorus",
        "bridge",
        "instrumental",
        "genre",
        "pop",
        "rock",
        "jazz",
        "electronic",
        "hip hop",
        "acoustic",
        "guitar",
        "piano",
        "drums",
        "synth",
        "bass line",
    }
)

_VALID_BACKEND_LABELS = {"default", "mlx_dit", "torch_dit"}
_DEFAULT_MAX_PIPELINES = 2
_MAX_PIPELINES_ENV = "ANVIL_MCP_MAX_PIPELINES"
_IDLE_TIMEOUT_ENV = "ANVIL_MCP_IDLE_TIMEOUT_SECONDS"
_MEMORY_ENV_PREFIX = "ANVIL_MCP_MEMORY"
_LARGE_OUTPUT_CLEANUP_MB_ENV = "ANVIL_MCP_LARGE_OUTPUT_CLEANUP_MB"
_DEFAULT_LARGE_OUTPUT_CLEANUP_MB = 128.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Write a status message to stderr (stdout is reserved for MCP protocol)."""
    print(f"[anvil-mcp] {msg}", file=sys.stderr, flush=True)


def _backend_label(value: bool | None) -> str:
    """Return the user-facing backend label for a pipeline cache key."""
    if value is None:
        return "default"
    return "mlx_dit" if value else "torch_dit"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _pipeline_cache_policy() -> dict[str, Any]:
    """Return MCP pipeline cache policy from env with safe defaults."""
    return {
        "max_pipelines": _env_int(_MAX_PIPELINES_ENV, _DEFAULT_MAX_PIPELINES),
        "idle_timeout_seconds": _env_float(_IDLE_TIMEOUT_ENV, 0.0),
    }


def _pipeline_cache_key_label(key: tuple[str, bool | None]) -> dict[str, str]:
    return {"model": key[0], "backend": _backend_label(key[1])}


def _evict_pipeline_key(
    key: tuple[str, bool | None],
    *,
    reason: str,
) -> dict[str, Any] | None:
    pipeline = _pipeline_cache.pop(key, None)
    _pipeline_last_used.pop(key, None)
    if pipeline is None:
        return None

    result: dict[str, Any] = _pipeline_cache_key_label(key)
    result["reason"] = reason
    result["release_actions"] = _release_pipeline(pipeline)
    del pipeline
    return result


def _enforce_pipeline_cache_policy(
    *,
    incoming_key: tuple[str, bool | None] | None = None,
) -> list[dict[str, Any]]:
    """Evict idle/LRU cached pipelines according to MCP cache policy."""
    evicted: list[dict[str, Any]] = []
    policy = _pipeline_cache_policy()
    now = time.monotonic()

    idle_timeout = float(policy["idle_timeout_seconds"])
    if idle_timeout > 0:
        for key, last_used in list(_pipeline_last_used.items()):
            if key == incoming_key:
                continue
            if key not in _pipeline_cache:
                _pipeline_last_used.pop(key, None)
                continue
            if now - last_used >= idle_timeout:
                item = _evict_pipeline_key(key, reason="idle_timeout")
                if item is not None:
                    evicted.append(item)

    max_pipelines = int(policy["max_pipelines"])
    if max_pipelines <= 0:
        return evicted

    target_len = max_pipelines
    if incoming_key is not None and incoming_key not in _pipeline_cache:
        target_len = max(0, max_pipelines - 1)

    while len(_pipeline_cache) > target_len:
        candidates = [key for key in _pipeline_cache if key != incoming_key]
        if not candidates:
            break
        lru_key = min(candidates, key=lambda key: _pipeline_last_used.get(key, 0.0))
        item = _evict_pipeline_key(lru_key, reason="lru_max_pipelines")
        if item is not None:
            evicted.append(item)

    return evicted


def _touch_pipeline_cache_key(key: tuple[str, bool | None]) -> None:
    _pipeline_last_used[key] = time.monotonic()


def _flush_memory_caches() -> dict[str, Any]:
    """Best-effort Python, torch, and MLX cache cleanup."""
    return flush_memory_caches()


def _process_rss_mb() -> float | None:
    """Return current process RSS in MB when it can be measured cheaply."""
    return process_rss_mb()


def _system_memory_status() -> dict[str, Any]:
    """Return system memory details when psutil is installed."""
    return system_memory_status()


def _torch_memory_status() -> dict[str, Any]:
    """Return accelerator memory stats exposed by torch."""
    return torch_memory_status()


def _mlx_memory_status() -> dict[str, Any]:
    """Return MLX memory stats when MLX is importable."""
    return mlx_memory_status()


def _large_output_cleanup_threshold_mb() -> float:
    return _env_float(
        _LARGE_OUTPUT_CLEANUP_MB_ENV,
        _DEFAULT_LARGE_OUTPUT_CLEANUP_MB,
    )


def _relieve_memory_pressure(
    *,
    protected_key: tuple[str, bool | None] | None = None,
    reason: str,
) -> dict[str, Any]:
    """Flush caches and evict one old pipeline when memory pressure is configured."""
    pressure_cleanup = cleanup_if_memory_pressure(
        reason=reason,
        env_prefix=_MEMORY_ENV_PREFIX,
    )
    evicted: list[dict[str, Any]] = []
    if not pressure_cleanup["triggered"]:
        return {"pressure_cleanup": pressure_cleanup, "evicted": evicted}

    candidates = [key for key in _pipeline_cache if key != protected_key]
    if candidates:
        lru_key = min(candidates, key=lambda key: _pipeline_last_used.get(key, 0.0))
        item = _evict_pipeline_key(lru_key, reason="memory_pressure")
        if item is not None:
            evicted.append(item)
            pressure_cleanup["post_evict_flush"] = _flush_memory_caches()

    return {"pressure_cleanup": pressure_cleanup, "evicted": evicted}


def _post_generation_cleanup(
    *,
    output_mb: float,
    protected_key: tuple[str, bool | None] | None,
) -> None:
    """Clean up after a generation call without unloading the current pipeline."""
    threshold_mb = _large_output_cleanup_threshold_mb()
    if threshold_mb > 0 and output_mb >= threshold_mb:
        _flush_memory_caches()

    pressure_result = _relieve_memory_pressure(
        protected_key=protected_key,
        reason="mcp.post_generation",
    )
    for item in pressure_result["evicted"]:
        _log(
            "Evicted cached model under memory pressure: "
            f"{item['model']} ({item['backend']})"
        )


def _pipeline_lora_status(pipeline: Any) -> dict[str, Any]:
    """Return LoRA status for a pipeline if it exposes one."""
    status_fn = getattr(pipeline, "lora_status", None)
    if not callable(status_fn):
        return {}
    try:
        status = status_fn()
    except Exception as exc:
        return {"error": str(exc)}
    return status if isinstance(status, dict) else {"status": status}


def _loaded_pipeline_summary() -> list[dict[str, Any]]:
    """Return a compact summary of loaded cached pipelines."""
    result = []
    for (model_name, backend), pipeline in _pipeline_cache.items():
        item = {
            "model": model_name,
            "backend": _backend_label(backend),
            "pipeline_class": pipeline.__class__.__name__,
        }
        lora_status = _pipeline_lora_status(pipeline)
        if lora_status:
            item["lora_status"] = lora_status
        result.append(item)
    return result


def _release_pipeline(pipeline: Any) -> list[str]:
    """Best-effort release hook before deleting a cached pipeline."""
    actions = []
    for method_name in ("unload", "close", "cleanup"):
        method = getattr(pipeline, method_name, None)
        if callable(method):
            try:
                method()
                actions.append(method_name)
            except Exception as exc:
                actions.append(f"{method_name} failed: {exc}")
            break
    return actions


def _disable_lora_for_plain_call(pipeline: Any) -> dict[str, Any] | None:
    """Disable an active cached LoRA when a call intentionally omits lora=."""
    status = _pipeline_lora_status(pipeline)
    if not status.get("loaded") or not status.get("active"):
        return None

    toggle = getattr(pipeline, "set_lora_enabled", None)
    if callable(toggle):
        result = toggle(False)
        return result if isinstance(result, dict) else {"message": str(result)}

    handler = getattr(pipeline, "_handler", None)
    set_use_lora = getattr(handler, "set_use_lora", None)
    if callable(set_use_lora):
        message = str(set_use_lora(False))
        return {"message": message, "status": _pipeline_lora_status(pipeline)}

    return {"warning": "Loaded LoRA is active but this pipeline cannot disable it."}


def _parse_server_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse MCP server flags and return any unconsumed args for FastMCP."""
    parser = argparse.ArgumentParser(
        prog="python -m anvil_audio.mcp_server",
        description="Run the Anvil Audio MCP server over stdio.",
    )
    mlx_group = parser.add_mutually_exclusive_group()
    mlx_group.add_argument(
        "--no-mlx-dit",
        action="store_true",
        help=(
            "Disable ACE-Step native MLX DiT/VAE for this MCP server process. "
            "Use this when generating with ACE-Step PEFT/LoKr LoRA adapters."
        ),
    )
    mlx_group.add_argument(
        "--use-mlx-dit",
        action="store_true",
        help=(
            "Force ACE-Step native MLX DiT/VAE for this MCP server process. "
            "Overrides ANVIL_ACESTEP_USE_MLX_DIT."
        ),
    )
    return parser.parse_known_args(argv)


def _apply_server_args(args: argparse.Namespace) -> None:
    """Apply process-level MCP server options before any model is loaded."""
    if args.no_mlx_dit:
        os.environ["ANVIL_ACESTEP_USE_MLX_DIT"] = "0"
        _log("ACE-Step native MLX DiT/VAE disabled for this MCP server")
    elif args.use_mlx_dit:
        os.environ["ANVIL_ACESTEP_USE_MLX_DIT"] = "1"
        _log("ACE-Step native MLX DiT/VAE forced on for this MCP server")


def _is_music_prompt(prompt: str) -> bool:
    words = set(prompt.lower().split())
    return bool(words & _MUSIC_KEYWORDS)


def _auto_select_model(prompt: str) -> str:
    """Pick the best available model for *prompt* based on simple heuristics."""
    models = registry.list_models()
    if not models:
        raise RuntimeError(
            "No models are registered. Add one to ~/.anvil-audio/registry.yaml."
        )

    if _is_music_prompt(prompt):
        for e in models:
            if e.pipeline_type == "acestep":
                return e.name

    for e in models:
        if e.pipeline_type == "diffusion":
            return e.name

    return models[0].name  # fallback: whatever is first


def _get_pipeline(model_name: str, *, use_mlx_dit: bool | None = None) -> Any:
    """Return a cached pipeline, loading it if necessary."""
    entry = registry.get_model(model_name)
    cache_key = (
        model_name,
        use_mlx_dit if entry is not None and entry.pipeline_type == "acestep" else None,
    )
    pressure_result = _relieve_memory_pressure(
        protected_key=cache_key,
        reason="mcp.before_pipeline_load",
    )
    for item in pressure_result["evicted"]:
        _log(
            "Evicted cached model under memory pressure: "
            f"{item['model']} ({item['backend']})"
        )

    evicted = _enforce_pipeline_cache_policy(incoming_key=cache_key)
    for item in evicted:
        _log(
            "Evicted cached model: "
            f"{item['model']} ({item['backend']}, {item['reason']})"
        )

    if cache_key not in _pipeline_cache:
        suffix = ""
        if entry is not None and entry.pipeline_type == "acestep" and use_mlx_dit is not None:
            suffix = f" (use_mlx_dit={use_mlx_dit})"
        _log(f"Loading model: {model_name}{suffix}")
        from anvil_audio.core.registry import load_pipeline

        # load_pipeline triggers model weight downloads and prints progress
        # messages (HF hub, tqdm, "Loading VAE...", etc.) to stdout.
        # Redirect to stderr — stdout is reserved for JSON-RPC.
        with stdout_to_stderr():
            _pipeline_cache[cache_key] = load_pipeline(
                model_name,
                use_mlx_dit=use_mlx_dit,
            )
        _log(f"Model ready: {model_name}{suffix}")
    _touch_pipeline_cache_key(cache_key)
    return _pipeline_cache[cache_key]


def _clamp_duration(model_name: str, duration: float) -> float:
    entry = registry.get_model(model_name)
    if entry and entry.max_duration and duration > entry.max_duration:
        _log(
            f"seconds_total={duration:.1f}s exceeds model max "
            f"{entry.max_duration:.1f}s — clamping"
        )
        return entry.max_duration
    return duration


def _resolve_project(project: str) -> str:
    return (project or _active_project).strip() or "default"


def _load_audio_file(file_path: str) -> tuple[int, np.ndarray]:
    """Load an audio file from disk and return (sample_rate, int16_ndarray)."""
    import soundfile as sf

    data, sr = sf.read(file_path, dtype="int16", always_2d=False)
    return sr, data


def _int16_to_float_tensor(arr: np.ndarray) -> Any:
    """Convert (N,)/(N,C) int16 ndarray → (C,N) float32 torch.Tensor in [-1, 1]."""
    import torch

    float_arr = arr.astype(np.float32) / 32768.0
    if float_arr.ndim == 1:
        float_arr = float_arr[np.newaxis, :]  # mono → (1, N)
    else:
        float_arr = float_arr.T  # (N, C) → (C, N)
    return torch.from_numpy(float_arr)


def _run_generate(
    prompt: str,
    model_name: str,
    duration: float,
    steps: int | None,
    cfg_scale: float | None,
    seed: int,
    fmt: str,
    project: str,
    lyrics: str,
    negative_prompt: str = "",
    lora: str = "",
    lora_scale: float = 1.0,
    lora_adapter_name: str = "",
    use_mlx_dit: bool | None = None,
) -> dict[str, Any]:
    """Core generation logic shared by generate_audio and batch_generate."""
    import numpy as np

    entry = registry.get_model(model_name)
    params = entry.resolved_params() if entry else {}
    is_acestep = entry is not None and entry.pipeline_type == "acestep"
    lora_ref = lora.strip()
    effective_use_mlx_dit = use_mlx_dit
    if is_acestep and lora_ref and effective_use_mlx_dit is None:
        effective_use_mlx_dit = False

    pipeline = _get_pipeline(model_name, use_mlx_dit=effective_use_mlx_dit)
    current_key = (model_name, effective_use_mlx_dit if is_acestep else None)

    effective_steps = steps if steps is not None else params.get("steps", 100)
    effective_cfg = cfg_scale if cfg_scale is not None else params.get("cfg_scale", 7.0)
    effective_seed = (
        int(seed)
        if int(seed) != -1
        else int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
    )

    if is_acestep:
        conditioning = [
            {
                "prompt": prompt,
                "lyrics": lyrics or "",
                "negative_prompt": negative_prompt or "",
                "seconds_total": duration,
            }
        ]
    else:
        conditioning = [
            {"prompt": prompt, "seconds_start": 0.0, "seconds_total": duration}
        ]

    extra: dict[str, Any] = {}
    if is_acestep and effective_use_mlx_dit is not None:
        extra["use_mlx_dit"] = effective_use_mlx_dit
    if lora_ref:
        if not is_acestep:
            raise ValueError("LoRA adapters are only supported with ACE-Step models.")
        from anvil_audio.lora import resolve_adapter_reference

        lora_path, lora_entry = resolve_adapter_reference(lora_ref)
        apply_lora = getattr(pipeline, "apply_lora_adapter", None)
        if not callable(apply_lora):
            raise RuntimeError("Selected ACE-Step pipeline does not support LoRA loading.")
        lora_status = apply_lora(
            str(lora_path),
            adapter_name=lora_adapter_name.strip() or None,
            scale=float(lora_scale),
        )
        extra["lora"] = {
            "reference": lora_ref,
            "path": str(lora_path),
            "scale": float(lora_scale),
            "adapter_name": lora_adapter_name.strip() or None,
            "registry_id": lora_entry.id if lora_entry else None,
            "status": lora_status,
        }
    elif is_acestep:
        disabled_lora = _disable_lora_for_plain_call(pipeline)
        if disabled_lora is not None:
            extra["lora_disabled_for_call"] = disabled_lora

    gen_kwargs: dict[str, Any] = {"cfg_scale": effective_cfg}
    if not is_acestep:
        sampler = params.get("sampler_type", "dpmpp-3m-sde")
        sigma_min = params.get("sigma_min", 0.03)
        sigma_max = params.get("sigma_max", 500.0)
        gen_kwargs.update(
            sampler_type=sampler, sigma_min=sigma_min, sigma_max=sigma_max
        )
        if negative_prompt:
            gen_kwargs["negative_conditioning"] = [
                {
                    "prompt": negative_prompt,
                    "seconds_start": 0.0,
                    "seconds_total": duration,
                }
            ]

    result: Any | None = None
    audio: Any | None = None
    output_mb = 0.0
    try:
        # Samplers may print progress to stdout; redirect so MCP stdio is clean.
        _gen_t0 = time.perf_counter()
        with stdout_to_stderr():
            result = pipeline.generate(
                conditioning,
                steps=effective_steps,
                seed=effective_seed,
                disable_tqdm=True,
                **gen_kwargs,
            )  # [B, C, T]
        gen_duration = round(time.perf_counter() - _gen_t0, 3)

        audio = result[0]  # [C, T]
        sr = pipeline.sample_rate
        length = int(sr * duration)
        audio = audio[:, :length]
        output_mb = estimate_values_size_mb(result, audio)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_acestep and lyrics:
            extra["lyrics"] = lyrics

        meta = GenerationMetadata(
            prompt=prompt,
            model_name=model_name,
            seed=effective_seed,
            steps=effective_steps,
            cfg_scale=float(effective_cfg),
            sampler_type=params.get(
                "sampler_type", "ode" if is_acestep else "dpmpp-3m-sde"
            ),
            sigma_min=float(params.get("sigma_min", 0.0)),
            sigma_max=float(params.get("sigma_max", 0.0)),
            duration_seconds=audio.shape[-1] / sr,
            timestamp=ts,
            negative_prompt=negative_prompt or "",
            generation_duration_seconds=gen_duration,
            extra=extra,
        )

        output_manager = OutputManager(project=_resolve_project(project))
        path, sidecar = output_manager.save_audio(audio.cpu(), meta, sr, ext=fmt)

        return {
            "path": str(path),
            "sidecar_path": str(sidecar),
            "generation_duration_seconds": gen_duration,
            "metadata": meta.to_dict(),
        }
    finally:
        output_mb = max(output_mb, estimate_values_size_mb(result, audio))
        del audio
        del result
        _post_generation_cleanup(output_mb=output_mb, protected_key=current_key)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def prepare_music_prompt(
    prompt: str,
    duration_seconds: float = 60.0,
    negative_prompt: str = "",
    write_lyrics: bool = True,
    enhance: bool = True,
) -> dict[str, Any]:
    """Enhance a music prompt, suggest negatives, and optionally write lyrics.

    Uses the local MLX Llama model when available. The first run may download
    the default local LLM if it is not already present.

    Args:
        prompt: Base music prompt.
        duration_seconds: Target track length used for lyric density.
        negative_prompt: Existing negative prompt to preserve or improve.
        write_lyrics: Whether to write duration-aware lyrics.
        enhance: Whether to rewrite the prompt before lyric writing.

    Returns:
        dict with prompt, negative_prompt, lyrics, and raw LLM output.
    """
    try:
        from anvil_audio.intelligence import prepare_song_prompt

        return prepare_song_prompt(
            prompt,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            write_vocals=write_lyrics,
            enhance=enhance,
        ).to_dict()
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def generate_audio(
    prompt: str,
    model: str = "",
    negative_prompt: str = "",
    duration_seconds: float = 30.0,
    steps: int = 0,
    cfg_scale: float = 0.0,
    seed: int = -1,
    fmt: str = "wav",
    project: str = "",
    lyrics: str = "",
    lora: str = "",
    lora_scale: float = 1.0,
    lora_adapter_name: str = "",
    use_mlx_dit: bool | None = None,
) -> dict[str, Any]:
    """Generate audio from a text prompt using a registered model.

    For sound effects and ambient audio, describe what you want directly
    (e.g. "wooden door creak", "rain on a tin roof", "footsteps on gravel").
    For music, use style tags and optional lyrics with an ACE-Step model
    (e.g. "indie pop, acoustic guitar, warm vocals, upbeat").

    If model is not specified, Anvil picks one automatically: ACE-Step for
    music-style prompts (if installed), Stable Audio for everything else.

    Args:
        prompt: Text description of the audio to generate.
        model: Registry model name (e.g. "stable-audio-open-1.0",
               "acestep-v1.5-turbo"). Leave empty to auto-select.
        negative_prompt: Text describing sounds or qualities to avoid.
                         For ACE-Step this controls the LM/thinking path when
                         enabled.
        duration_seconds: Target length in seconds. Clamped to the model's
                          maximum (47s for Stable Audio, 600s for ACE-Step).
        steps: Diffusion/inference steps. 0 = use the model's default.
        cfg_scale: Guidance scale. 0.0 = use the model's default.
        seed: RNG seed for reproducibility. -1 = random.
        fmt: Output format — "wav", "flac", "mp3", or "ogg".
        project: Output project folder name. Uses the active project if
                 omitted (set with set_active_project).
        lyrics: Lyric text for ACE-Step models. Use section markers like
                [verse], [chorus], [bridge]. Leave blank for instrumental.
        lora: ACE-Step adapter id/name from `list_lora_adapters`, or a direct
              PEFT/LoKr adapter path. Leave blank for no adapter.
        lora_scale: Adapter strength. 1.0 = full strength; lower values blend
                    with the base model.
        lora_adapter_name: Optional runtime adapter name for ACE-Step.
        use_mlx_dit: ACE-Step backend override. False disables native MLX
                     DiT/VAE for this call, which is required for current
                     PEFT/LoKr LoRA adapters. True forces MLX DiT. Null/omitted
                     uses registry/env defaults, except LoRA calls default to
                     False automatically.

    Returns:
        dict with keys: path (str), sidecar_path (str), metadata (dict)
    """
    try:
        model_name = model.strip() or _auto_select_model(prompt)
        duration = _clamp_duration(model_name, duration_seconds)
        return _run_generate(
            prompt=prompt,
            model_name=model_name,
            duration=duration,
            steps=steps if steps > 0 else None,
            cfg_scale=cfg_scale if cfg_scale > 0.0 else None,
            seed=seed,
            fmt=fmt,
            project=project,
            lyrics=lyrics,
            negative_prompt=negative_prompt,
            lora=lora,
            lora_scale=lora_scale,
            lora_adapter_name=lora_adapter_name,
            use_mlx_dit=use_mlx_dit,
        )
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def batch_generate(
    items: list[dict[str, Any]],
    project: str = "",
) -> list[dict[str, Any]]:
    """Generate multiple audio clips in a single call.

    Each item in the list is a condition dict with the same keys as
    generate_audio: prompt (required), model, duration_seconds, steps,
    cfg_scale, seed, fmt, lyrics, negative_prompt, lora, lora_scale,
    lora_adapter_name, and use_mlx_dit. Items using the same model/backend
    share the cached pipeline — no reload penalty.

    Args:
        items: List of condition dicts. Each must have at least "prompt".
               Other keys are optional and default to the same values as
               generate_audio.
        project: Output project folder applied to all items in the batch.
                 Per-item "project" keys override this.

    Returns:
        List of result dicts (same structure as generate_audio), one per
        input item. Items that fail include an "error" key instead.
    """
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            prompt = item.get("prompt", "")
            if not prompt:
                results.append({"error": "item missing required 'prompt' key"})
                continue
            model_name = item.get("model", "").strip() or _auto_select_model(prompt)
            dur = _clamp_duration(model_name, float(item.get("duration_seconds", 30.0)))
            steps_val = int(item.get("steps", 0))
            cfg_val = float(item.get("cfg_scale", 0.0))
            result = _run_generate(
                prompt=prompt,
                model_name=model_name,
                duration=dur,
                steps=steps_val if steps_val > 0 else None,
                cfg_scale=cfg_val if cfg_val > 0.0 else None,
                seed=int(item.get("seed", -1)),
                fmt=str(item.get("fmt", "wav")),
                project=item.get("project", "") or project,
                lyrics=str(item.get("lyrics", "")),
                negative_prompt=str(item.get("negative_prompt", "")),
                lora=str(item.get("lora", "")),
                lora_scale=float(item.get("lora_scale", 1.0)),
                lora_adapter_name=str(item.get("lora_adapter_name", "")),
                use_mlx_dit=item.get("use_mlx_dit"),
            )
            results.append(result)
        except Exception as exc:
            results.append({"error": str(exc), "prompt": item.get("prompt", "")})
    return results


@mcp.tool()
def edit_audio(
    file_path: str,
    # Normalize
    normalize: bool = False,
    normalize_target_db: float = -1.0,
    normalize_lufs: bool = False,
    # Trim silence
    trim_silence: bool = False,
    trim_top_db: float = 40.0,
    # Loop / clip to range
    loop_start: float = 0.0,
    loop_end: float = 0.0,
    # Time stretch + pitch shift
    time_stretch: float = 1.0,
    pitch_shift: float = 0.0,
    # EQ (three bands)
    eq_low_db: float = 0.0,
    eq_low_freq: float = 200.0,
    eq_mid_db: float = 0.0,
    eq_mid_freq: float = 1000.0,
    eq_high_db: float = 0.0,
    eq_high_freq: float = 8000.0,
    # Reverb
    reverb_mix: float = 0.0,
    reverb_size: float = 0.25,
    reverb_damping: float = 0.5,
    # Fade
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    # Output
    fmt: str = "wav",
    project: str = "",
) -> dict[str, Any]:
    """Post-process an audio file with a non-destructive effects chain.

    The chain is always applied in this fixed order:
    trim silence → loop/clip → time stretch + pitch shift → EQ → reverb
    → fade in/out → normalize.

    The source file is never modified. Export creates a new file via the
    output manager; the sidecar records the source path and full effects
    config so every edit is traceable.

    Args:
        file_path: Absolute path to the audio file to process.
        normalize: Apply loudness normalization.
        normalize_target_db: Target level in dBFS (peak) or LUFS.
        normalize_lufs: True = LUFS mode (-14 is streaming standard),
                        False = peak mode.
        trim_silence: Strip silence from the edges.
        trim_top_db: Silence threshold in dB below peak. Higher = more
                     aggressive trimming.
        loop_start: Trim everything before this time (seconds). 0 = off.
        loop_end: Trim everything after this time (seconds). 0 = off.
        time_stretch: Speed factor. 1.0 = no change, 0.5 = half speed,
                      2.0 = double speed. Pitch is held constant unless
                      pitch_shift is also set.
        pitch_shift: Semitones to shift pitch. +12 = up one octave.
        eq_low_db: Low shelf gain in dB (positive = boost, negative = cut).
        eq_low_freq: Low shelf cutoff frequency in Hz.
        eq_mid_db: Mid peak gain in dB.
        eq_mid_freq: Mid peak center frequency in Hz.
        eq_high_db: High shelf gain in dB.
        eq_high_freq: High shelf cutoff frequency in Hz.
        reverb_mix: Wet/dry mix — 0.0 = dry, 1.0 = fully wet.
        reverb_size: Room size, 0.0–1.0.
        reverb_damping: High-frequency damping, 0.0–1.0.
        fade_in: Fade-in duration in seconds (linear ramp from silence).
        fade_out: Fade-out duration in seconds (linear ramp to silence).
        fmt: Output format — "wav", "flac", "mp3", or "ogg".
        project: Output project folder. Uses the active project if omitted.

    Returns:
        dict with keys: path (str), sidecar_path (str), effects (dict)
    """
    try:
        from anvil_audio.interface.edit_tab import process_audio_chain

        audio_input = _load_audio_file(file_path)
        loop_enabled = loop_end > 0
        eq_enabled = any(abs(g) > 0.05 for g in (eq_low_db, eq_mid_db, eq_high_db))
        reverb_enabled = reverb_mix > 1e-4

        result = process_audio_chain(
            audio_input,
            trim_silence,
            trim_top_db,
            loop_enabled,
            loop_start,
            loop_end if loop_enabled else 30.0,
            time_stretch,
            pitch_shift,
            eq_enabled,
            eq_low_db,
            eq_low_freq,
            eq_mid_db,
            eq_mid_freq,
            eq_high_db,
            eq_high_freq,
            reverb_enabled,
            reverb_size,
            reverb_damping,
            reverb_mix,
            fade_in,
            fade_out,
            normalize,
            normalize_target_db,
            normalize_lufs,
        )

        if result is None:
            return {
                "error": "Processing chain returned no audio — check that the file exists and is a valid audio file."
            }

        out_sr, out_arr = result
        tensor = _int16_to_float_tensor(out_arr)

        source_p = file_path.strip()
        source_sidecar = ""
        candidate = Path(source_p).with_suffix(".json")
        if candidate.exists():
            source_sidecar = str(candidate)

        effects_config = {
            "trim_silence": {"enabled": trim_silence, "top_db": trim_top_db},
            "loop": {"enabled": loop_enabled, "start_s": loop_start, "end_s": loop_end},
            "time_stretch": {"factor": time_stretch, "pitch_semitones": pitch_shift},
            "eq": {
                "enabled": eq_enabled,
                "low": {"gain_db": eq_low_db, "freq": eq_low_freq},
                "mid": {"gain_db": eq_mid_db, "freq": eq_mid_freq},
                "high": {"gain_db": eq_high_db, "freq": eq_high_freq},
            },
            "reverb": {
                "enabled": reverb_enabled,
                "room_size": reverb_size,
                "damping": reverb_damping,
                "wet_dry": reverb_mix,
            },
            "fade": {"in_s": fade_in, "out_s": fade_out},
            "normalize": {
                "enabled": normalize,
                "target_db": normalize_target_db,
                "mode": "lufs" if normalize_lufs else "peak",
            },
        }

        meta = GenerationMetadata(
            prompt="(edit)",
            model_name="edit",
            seed=-1,
            steps=0,
            cfg_scale=1.0,
            sampler_type="edit",
            sigma_min=0.0,
            sigma_max=0.0,
            duration_seconds=tensor.shape[-1] / out_sr,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            extra={
                "source_audio_path": source_p,
                "source_sidecar_path": source_sidecar,
                "effects": effects_config,
                "export_format": fmt,
            },
        )

        output_manager = OutputManager(project=_resolve_project(project))
        path, sidecar = output_manager.save_audio(tensor, meta, out_sr, ext=fmt)

        return {
            "path": str(path),
            "sidecar_path": str(sidecar),
            "effects": effects_config,
        }

    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_lora_adapters() -> list[dict[str, Any]]:
    """List locally registered LoRA adapters that MCP generation can reference.

    Returns:
        List of adapter metadata dicts. Use the `id` value as generate_audio.lora
        or batch_generate item `lora`. Adapters with `loadable=false` are tracked
        but cannot be applied by ACE-Step's Python runtime.
    """
    from anvil_audio.lora import list_adapters

    return [
        {
            "id": entry.id,
            "name": entry.name,
            "path": entry.path,
            "format": entry.format,
            "loadable": entry.loadable,
            "source": entry.source,
            "base_model": entry.base_model,
            "repo_id": entry.repo_id,
            "revision": entry.revision,
            "created_at": entry.created_at,
            "notes": entry.notes,
        }
        for entry in list_adapters()
    ]


@mcp.tool()
def list_models() -> list[dict[str, Any]]:
    """List all registered models with their capabilities and current status.

    Returns one entry per model with: name, pipeline_type, max_duration,
    default_params, is_loaded (whether it's currently in the model cache),
    and supported_features (e.g. "lyrics" for ACE-Step models).
    """
    loaded = {key[0] for key in _pipeline_cache}
    result = []
    for entry in registry.list_models():
        features: list[str] = []
        if entry.pipeline_type == "acestep":
            features.extend(
                [
                    "lyrics",
                    "negative-prompt",
                    "music-generation",
                    "style-tags",
                    "lora",
                ]
            )
        else:
            features.extend(
                ["negative-prompt", "sound-effects", "ambient-audio", "init-audio"]
            )

        result.append(
            {
                "name": entry.name,
                "pipeline_type": entry.pipeline_type,
                "max_duration": entry.max_duration,
                "default_params": entry.resolved_params(),
                "is_loaded": entry.name in loaded,
                "loaded_backends": [
                    _backend_label(key[1])
                    for key in _pipeline_cache
                    if key[0] == entry.name
                ],
                "supported_features": features,
                "pretrained_name": entry.pretrained_name,
            }
        )
    return result


@mcp.tool()
def get_memory_status(flush: bool = False) -> dict[str, Any]:
    """Report MCP process memory, accelerator cache stats, and loaded models.

    Args:
        flush: If true, run Python GC plus torch/MLX cache cleanup before
               reporting memory. This does not unload model weights.

    Returns:
        dict with process RSS, system memory details when available, torch/MLX
        accelerator stats, and cached pipeline/Lora state.
    """
    flush_result = _flush_memory_caches() if flush else None
    return {
        "pid": os.getpid(),
        "process_rss_mb": _process_rss_mb(),
        "system_memory": _system_memory_status(),
        "torch_memory": _torch_memory_status(),
        "mlx_memory": _mlx_memory_status(),
        "memory_pressure": memory_pressure_status(env_prefix=_MEMORY_ENV_PREFIX),
        "cache_policy": _pipeline_cache_policy(),
        "loaded_pipelines": _loaded_pipeline_summary(),
        "flush": flush_result,
    }


@mcp.tool()
def unload_models(
    model: str = "",
    backend: str = "",
    flush: bool = True,
) -> dict[str, Any]:
    """Unload cached model pipelines from this MCP server process.

    Args:
        model: Optional registry model name to unload. Empty unloads all models.
        backend: Optional backend filter: default, mlx_dit, or torch_dit.
        flush: Run Python GC plus torch/MLX cache cleanup after unloading.

    Returns:
        dict describing unloaded cache entries and remaining loaded pipelines.
    """
    model_filter = model.strip()
    backend_filter = backend.strip().lower().replace("-", "_")
    if backend_filter and backend_filter not in _VALID_BACKEND_LABELS:
        return {
            "error": (
                "backend must be one of: "
                + ", ".join(sorted(_VALID_BACKEND_LABELS))
            )
        }

    unloaded: list[dict[str, Any]] = []
    for key in list(_pipeline_cache):
        model_name, backend_value = key
        backend_label = _backend_label(backend_value)
        if model_filter and model_name != model_filter:
            continue
        if backend_filter and backend_label != backend_filter:
            continue

        pipeline = _pipeline_cache.pop(key)
        _pipeline_last_used.pop(key, None)
        unloaded.append(
            {
                "model": model_name,
                "backend": backend_label,
                "release_actions": _release_pipeline(pipeline),
            }
        )
        del pipeline

    flush_result = _flush_memory_caches() if flush else None
    return {
        "unloaded": unloaded,
        "remaining_loaded_pipelines": _loaded_pipeline_summary(),
        "memory": {
            "process_rss_mb": _process_rss_mb(),
            "system_memory": _system_memory_status(),
            "torch_memory": _torch_memory_status(),
            "mlx_memory": _mlx_memory_status(),
            "memory_pressure": memory_pressure_status(env_prefix=_MEMORY_ENV_PREFIX),
        },
        "flush": flush_result,
    }


@mcp.tool()
def get_model_info(model: str) -> dict[str, Any]:
    """Get full details for a single registered model.

    Args:
        model: Registry model name (e.g. "stable-audio-open-1.0").

    Returns:
        dict with name, pipeline_type, max_duration, default_params,
        supported_features, is_loaded, and source (pretrained_name or
        local paths). Returns an error dict if the model is not found.
    """
    entry = registry.get_model(model)
    if entry is None:
        available = [e.name for e in registry.list_models()]
        return {
            "error": f"Model '{model}' not found in registry.",
            "available": available,
        }

    features: list[str] = []
    if entry.pipeline_type == "acestep":
        features.extend(
            ["lyrics", "negative-prompt", "music-generation", "style-tags", "lora"]
        )
    else:
        features.extend(
            [
                "negative-prompt",
                "sound-effects",
                "ambient-audio",
                "inpainting",
                "init-audio",
            ]
        )

    source: dict[str, Any] = {}
    if entry.pretrained_name:
        source["pretrained_name"] = entry.pretrained_name
    if entry.model_config_path:
        source["model_config_path"] = entry.model_config_path
    if entry.ckpt_path:
        source["ckpt_path"] = entry.ckpt_path
    if entry.acestep_project_root:
        source["acestep_project_root"] = entry.acestep_project_root

    return {
        "name": entry.name,
        "pipeline_type": entry.pipeline_type,
        "max_duration": entry.max_duration,
        "default_params": entry.resolved_params(),
        "supported_features": features,
        "is_loaded": any(key[0] == entry.name for key in _pipeline_cache),
        "loaded_backends": [
            _backend_label(key[1])
            for key in _pipeline_cache
            if key[0] == entry.name
        ],
        "source": source,
    }


@mcp.tool()
def list_recent_outputs(
    project: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent output files with their generation metadata.

    Scans the output directory for audio files, sorted newest-first.
    Includes sidecar metadata (prompt, model, seed, effects, etc.) when
    available.

    Args:
        project: Restrict to a specific project folder. Empty = all projects.
        limit: Maximum number of files to return (default 20).

    Returns:
        List of dicts with keys: path, project, modified_at, metadata (the
        parsed sidecar dict, or {} if no sidecar exists).
    """
    base = OutputManager.DEFAULT_BASE

    if project:
        roots = [base / project.strip()]
    else:
        roots = (
            sorted(base.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
            if base.exists()
            else []
        )

    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for ext in ("wav", "flac", "mp3", "ogg"):
            files.extend(root.rglob(f"*.{ext}"))

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    files = files[:limit]

    results: list[dict[str, Any]] = []
    for f in files:
        sidecar = f.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except Exception:
                pass
        results.append(
            {
                "path": str(f),
                "project": f.parent.name if f.parent != base else "default",
                "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "metadata": meta,
            }
        )
    return results


@mcp.tool()
def get_generation_metadata(file_path: str) -> dict[str, Any]:
    """Read the JSON sidecar for a specific audio file.

    Returns all recorded generation parameters: prompt, model, seed, steps,
    cfg_scale, duration, timestamp, and any extra pipeline-specific fields
    (lyrics for ACE-Step, effects chain for edited files).

    Args:
        file_path: Absolute path to an audio file (wav/flac/mp3). The
                   sidecar is expected at the same path with a .json suffix.

    Returns:
        Parsed sidecar dict, or an error dict if the sidecar doesn't exist.
    """
    sidecar = Path(file_path).with_suffix(".json")
    if not sidecar.exists():
        return {"error": f"No sidecar found at {sidecar}"}
    try:
        return json.loads(sidecar.read_text())
    except Exception as exc:
        return {"error": f"Failed to parse sidecar: {exc}"}


@mcp.tool()
def list_projects() -> list[dict[str, Any]]:
    """List all project folders under ~/anvil-audio-outputs/.

    Returns:
        List of dicts with: name, path, audio_file_count, last_modified.
        Sorted newest-first by last-modified time.
    """
    base = OutputManager.DEFAULT_BASE
    if not base.exists():
        return []

    projects: list[dict[str, Any]] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        audio_files = (
            list(d.rglob("*.wav"))
            + list(d.rglob("*.flac"))
            + list(d.rglob("*.mp3"))
            + list(d.rglob("*.ogg"))
        )
        mtime = d.stat().st_mtime
        projects.append(
            {
                "name": d.name,
                "path": str(d),
                "audio_file_count": len(audio_files),
                "last_modified": datetime.fromtimestamp(mtime).isoformat(),
            }
        )

    projects.sort(key=lambda p: p["last_modified"], reverse=True)
    return projects


@mcp.tool()
def set_active_project(project: str) -> dict[str, str]:
    """Set the default project for all subsequent generate and edit calls.

    Once set, you don't need to include project= in every call — all
    outputs will land in ~/anvil-audio-outputs/{project}/ automatically.
    Pass an empty string to reset to the "default" project.

    Args:
        project: Project folder name (e.g. "sfx-pack-v1", "session-2025").
                 Empty string resets to the "default" project.

    Returns:
        dict with key "active_project" confirming the new value.
    """
    global _active_project
    _active_project = project.strip()
    resolved = _active_project or "default"
    _log(f"Active project set to: {resolved}")
    return {"active_project": resolved}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server_args, remaining_argv = _parse_server_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining_argv]
    _apply_server_args(server_args)
    _log("Anvil Audio MCP server starting (stdio transport)")
    _log(f"Registry: {len(registry.list_models())} model(s) available")
    for entry in registry.list_models():
        _log(f"  [{entry.pipeline_type}] {entry.name}")
    mcp.run()
