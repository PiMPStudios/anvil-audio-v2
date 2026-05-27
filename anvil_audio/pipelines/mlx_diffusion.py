"""
MLXDiffusionPipeline — BasePipeline adapter for mlx-audiogen's Stable Audio Open.

mlx-audiogen (``pip install mlx-audiogen``) ports Stable Audio Open's DiT,
VAE, and T5 conditioner to Apple MLX, running entirely on the Metal GPU with
no PyTorch involvement during inference.

This module wraps ``StableAudioPipeline`` from mlx-audiogen behind Anvil's
``BasePipeline`` interface so that MLX-accelerated Stable Audio models
integrate with the registry, CLI batch generation, ``OutputManager``,
Gradio UI, and MCP server without any special-casing in those layers.

Weight conversion
-----------------
mlx-audiogen requires PyTorch weights to be converted to MLX safetensors
format before inference.  This adapter handles that automatically:

1. On first use the original HuggingFace weights are downloaded and
   converted via ``mlx_audiogen.models.stable_audio.convert_stable_audio()``.
2. Converted weights are cached at::

       ~/.cache/anvil-audio/mlx-weights/<model-slug>/

3. Subsequent loads skip conversion and load directly from the cache.

If ``weights_dir`` is given and already contains all required files
(``config.json``, ``vae.safetensors``, ``dit.safetensors``, etc.) it is
used as-is — no download or conversion occurs.

Audio I/O
---------
mlx-audiogen returns an ``mx.array`` of shape ``(1, channels, samples)`` in
``[-1, 1]`` float32 at 44 100 Hz.  This adapter materialises the MLX lazy
graph, copies the array to a NumPy buffer, and wraps it as a PyTorch tensor
``[B, channels, samples]`` so the rest of Anvil never sees an MLX type.

Vocabulary mapping
------------------
Anvil conditioning key  →  mlx-audiogen ``generate()`` parameter
``prompt``              →  ``prompt``
``negative_prompt``     →  ``negative_prompt``
``seconds_total``       →  ``seconds_total``
``steps``               →  ``steps``
``cfg_scale``           →  ``cfg_scale``
``seed``                →  ``seed`` (``None`` for random)
``sampler_type``        →  ``sampler`` (mapped: any non-euler/rk4 → ``"euler"``)
``sigma_max``           →  ``sigma_max`` (RF schedule max; default 1.0)

Usage
-----
::

    from anvil_audio.pipelines.mlx_diffusion import MLXDiffusionPipeline

    # weights are auto-downloaded and converted on first use
    pipe = MLXDiffusionPipeline(
        repo_id="stabilityai/stable-audio-open-small",
    )
    audio = pipe.generate(
        [{"prompt": "soft rain on leaves", "seconds_total": 10}]
    )
    # audio: Tensor [1, 2, T] at 44 100 Hz
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from anvil_audio.core.interfaces import BasePipeline
from anvil_audio.utils.memory import flush_memory_caches
from anvil_audio.utils.stdio_guard import stdout_to_stderr

# Files that must exist in a converted weights directory.
_REQUIRED_FILES = (
    "config.json",
    "vae.safetensors",
    "dit.safetensors",
    "t5.safetensors",
    "conditioners.safetensors",
)

# Anvil's local cache for auto-converted MLX weights.
_MLX_CACHE_ROOT = Path.home() / ".cache" / "anvil-audio" / "mlx-weights"

# Correct DiT configs for known Stability AI models.
# mlx-audiogen's convert_stable_audio() hardcodes small-model dims in the
# output config.json regardless of which model is converted.  We patch it
# afterwards for models whose architecture differs from the small model.
_KNOWN_DIT_CONFIGS: dict[str, dict] = {
    "stabilityai/stable-audio-open-1.0": {
        "io_channels": 64,
        "embed_dim": 1536,
        "depth": 24,
        "num_heads": 24,
        "cond_token_dim": 768,
        "global_cond_dim": 1536,
        "project_cond_tokens": False,
        "timestep_features_dim": 256,
    },
    # small model matches what the conversion writes; no patch needed.
}


def _patch_config_if_needed(config_path: Path, repo_id: str) -> None:
    """Overwrite the DiT section of *config_path* when the conversion wrote
    the wrong (hardcoded small-model) architecture.

    mlx-audiogen's ``convert_stable_audio`` always writes the small model's
    DiT config (embed_dim=1024, num_heads=8, depth=16) regardless of the
    source model.  For larger models like ``stable-audio-open-1.0`` this
    causes a shape mismatch when loading: the saved weights have
    ``transformer.rotary_pos_emb.inv_freq`` sized for the real architecture,
    but the model is initialised with the wrong dims and allocates a
    differently sized tensor.

    This function patches the config.json with the correct values from
    ``_KNOWN_DIT_CONFIGS`` so the model is built with the right architecture
    before weights are loaded.
    """
    if repo_id not in _KNOWN_DIT_CONFIGS:
        return
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    dit_override = _KNOWN_DIT_CONFIGS[repo_id]
    cfg.setdefault("dit", {}).update(dit_override)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(
        f"[mlx] Patched config.json for {repo_id} "
        f"(embed_dim={dit_override['embed_dim']}, "
        f"num_heads={dit_override['num_heads']})",
        file=sys.stderr,
    )


def _resolve_or_convert(repo_id: str, weights_dir: str | None) -> Path:
    """Return a path to a complete converted-weights directory.

    Resolution order
    ----------------
    1. If *weights_dir* is given and contains all required files → use it.
    2. Check the Anvil MLX cache (``~/.cache/anvil-audio/mlx-weights/<slug>``).
       If all files are present → use the cache.
    3. Download the original HuggingFace weights and convert them into the
       cache directory via ``convert_stable_audio()``.

    Args:
        repo_id:     HuggingFace repo ID (e.g. ``"stabilityai/stable-audio-open-small"``).
        weights_dir: Explicit path to pre-converted weights, or ``None`` to
                     use the auto-convert cache.

    Returns:
        ``Path`` to a directory that contains all required safetensors files.

    Raises:
        ImportError:  If ``mlx_audiogen`` is not installed.
        RuntimeError: If conversion fails.
    """
    # 1. Explicit pre-converted path.
    if weights_dir is not None:
        p = Path(weights_dir).expanduser().resolve()
        if p.is_dir() and all((p / f).exists() for f in _REQUIRED_FILES):
            return p
        # Path given but incomplete — warn and fall through to auto-convert.
        print(
            f"[mlx] weights_dir={weights_dir!r} is missing required files; "
            "falling back to auto-convert cache.",
            file=sys.stderr,
        )

    # 2. Check Anvil's local cache.
    slug = repo_id.split("/")[-1]   # "stable-audio-open-small"
    cache_path = _MLX_CACHE_ROOT / slug
    if cache_path.is_dir() and all((cache_path / f).exists() for f in _REQUIRED_FILES):
        print(f"[mlx] Using cached weights at {cache_path}", file=sys.stderr)
        # Patch config.json in case this cache was created before we added the
        # architecture correction (e.g. an existing stable-audio-open-1.0 cache
        # with the wrong hardcoded dims).
        _patch_config_if_needed(cache_path / "config.json", repo_id)
        return cache_path

    # 3. Auto-convert: download from HF and convert to MLX safetensors.
    try:
        from mlx_audiogen.models.stable_audio.convert import convert_stable_audio
    except ImportError as exc:
        raise ImportError(
            "mlx-audiogen is required for weight conversion.  Install it with:\n\n"
            "    pip install mlx-audiogen\n\n"
            f"Underlying error: {exc}"
        ) from exc

    print(
        f"[mlx] First-run weight conversion for {repo_id}\n"
        f"[mlx] Downloading from HuggingFace and converting to MLX format...\n"
        f"[mlx] This takes a few minutes and ~2 GB of disk space.\n"
        f"[mlx] Output: {cache_path}",
        file=sys.stderr,
    )
    try:
        # convert_stable_audio prints progress to stdout — redirect to stderr
        # so MCP stdio is not corrupted.
        with stdout_to_stderr():
            convert_stable_audio(repo_id=repo_id, output_dir=str(cache_path))
    except Exception as exc:
        raise RuntimeError(
            f"MLX weight conversion failed for {repo_id!r}.\n"
            f"  Cache path: {cache_path}\n"
            f"  Error: {exc}\n\n"
            "You can also convert manually with:\n"
            f"    mlx-audiogen-convert --model {repo_id} --output {cache_path}"
        ) from exc

    # Patch the config.json written by conversion with the correct architecture
    # for models that differ from the small model defaults.
    _patch_config_if_needed(cache_path / "config.json", repo_id)

    return cache_path


def is_mlx_available() -> bool:
    """Return True if running on Apple Silicon with mlx-audiogen installed.

    Checks both the platform (darwin + arm64) and whether the ``mlx_audiogen``
    package is importable.  Does **not** import mlx itself, so this is safe to
    call at module load time.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    import importlib.util
    return importlib.util.find_spec("mlx_audiogen") is not None


# Samplers understood by mlx-audiogen's StableAudioPipeline.generate()
_MLX_SAMPLERS = {"euler", "rk4"}

# Default sigma_max for the rectified-flow schedule used by mlx-audiogen
# (distinct from the DPM++ range of 0.3–500 used by the PyTorch diffusion path)
_RF_SIGMA_MAX = 1.0


class MLXDiffusionPipeline(BasePipeline):
    """``BasePipeline`` adapter for mlx-audiogen's Stable Audio Open pipeline.

    Wraps ``StableAudioPipeline`` from ``mlx-audiogen`` so MLX-accelerated
    Stable Audio models integrate with Anvil's registry, CLI, Gradio UI, and
    MCP server.

    Args:
        repo_id:       HuggingFace repo ID for the Stability AI model
                       (e.g. ``"stabilityai/stable-audio-open-small"``).
                       Used both as the conversion source and for tokenizer
                       download.
        weights_dir:   Path to a directory that already contains converted
                       MLX ``.safetensors`` files.  ``None`` (the default)
                       triggers auto-convert: weights are downloaded from
                       *repo_id* and cached at
                       ``~/.cache/anvil-audio/mlx-weights/<model-slug>/``.
        default_params: Generation parameter overrides applied when callers
                        omit individual kwargs.  Recognised keys:
                        ``steps``, ``cfg_scale``, ``sampler_type``,
                        ``sigma_max``.  ``sigma_min`` is accepted for API
                        compatibility but is not used by the RF sampler.
    """

    #: Stable Audio Open outputs 44.1 kHz stereo audio.
    _SAMPLE_RATE: int = 44100

    def __init__(
        self,
        repo_id: str = "stabilityai/stable-audio-open-small",
        weights_dir: str | None = None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        try:
            from mlx_audiogen.models.stable_audio import StableAudioPipeline
        except ImportError as exc:
            raise ImportError(
                "mlx-audiogen is required for the MLX backend.  Install it with:\n\n"
                "    pip install mlx-audiogen\n\n"
                "mlx-audiogen requires macOS on Apple Silicon (M1 or later).\n\n"
                f"Underlying error: {exc}"
            ) from exc

        self._repo_id: str = repo_id
        self._weights_dir: str | None = weights_dir
        self.default_params: dict[str, Any] = default_params or {
            "steps": 100,
            "cfg_scale": 7.0,
            "sampler_type": "euler",
            "sigma_min": 0.0,   # sentinel — RF sampler doesn't use sigma_min
            "sigma_max": _RF_SIGMA_MAX,
        }

        # Resolve (or trigger first-run auto-conversion of) the weights dir.
        resolved_weights = _resolve_or_convert(repo_id, weights_dir)

        print(
            f"->->-> Loading MLX Stable Audio  "
            f"repo={repo_id!r}  weights={resolved_weights}",
            file=sys.stderr,
        )
        # from_pretrained prints "Loading VAE/DiT/T5/conditioners..." to stdout;
        # redirect so MCP stdio is not corrupted.
        try:
            with stdout_to_stderr():
                self._pipe: Any = StableAudioPipeline.from_pretrained(
                    weights_dir=str(resolved_weights),
                    repo_id=repo_id,
                )
        except Exception as exc:
            exc_str = str(exc)
            # Shape mismatches during weight loading indicate a config/weight
            # mismatch — usually the conversion wrote the wrong architecture
            # dims.  Give a clear diagnostic rather than a cryptic traceback.
            if "shape" in exc_str.lower() or "expected" in exc_str.lower():
                pt_name = repo_id.split("/")[-1]
                raise RuntimeError(
                    f"MLX model loading failed for {repo_id!r} — the converted "
                    f"weights don't match the model architecture.\n\n"
                    f"  Error: {exc}\n\n"
                    f"Try deleting the cached weights and re-converting:\n"
                    f"    rm -rf {resolved_weights}\n\n"
                    f"Or use the PyTorch version instead:\n"
                    f"    anvil generate --model {pt_name}"
                ) from exc
            raise
        print("->->-> MLX Stable Audio ready", file=sys.stderr)

    # ------------------------------------------------------------------
    # BasePipeline abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """44 100 Hz — native output sample rate of Stable Audio Open."""
        return self._SAMPLE_RATE

    @property
    def sample_size(self) -> int:
        """Total samples in a generation, read from the loaded model config."""
        try:
            return int(self._pipe.config.sample_size)
        except AttributeError:
            # Fallback: 47 s at 44.1 kHz (Stable Audio Open 1.0 default)
            return self._SAMPLE_RATE * 47

    # ------------------------------------------------------------------
    # BasePipeline abstract method implementations
    # ------------------------------------------------------------------

    def generate(
        self,
        conditioning: list[dict[str, Any]],
        steps: int | None = None,
        seed: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Generate a batch of stereo audio waveforms via MLX Stable Audio.

        Each condition dict may contain:

        ===================  ==================================================
        Key                  Description
        ===================  ==================================================
        ``prompt``           Text description of desired audio.
        ``negative_prompt``  Negative guidance text (optional).
        ``seconds_total``    Target duration in seconds.
        ===================  ==================================================

        Args:
            conditioning: List of B condition dicts.
            steps:        Diffusion/flow steps.  Falls back to
                          ``default_params["steps"]`` (100).
            seed:         RNG seed; -1 draws a random seed each call.
            **kwargs:     Per-call overrides:
                          - ``cfg_scale`` (float)
                          - ``sampler_type`` (str): ``"euler"`` or ``"rk4"``
                            (any other value is mapped to ``"euler"``)
                          - ``sigma_max`` (float): RF noise schedule upper bound

        Returns:
            Float32 tensor ``[B, 2, T]`` in ``[-1, 1]`` at 44 100 Hz.
        """
        import mlx.core as mx

        effective_steps = (
            steps if steps is not None else self.default_params.get("steps", 100)
        )
        effective_cfg = float(
            kwargs.get("cfg_scale", self.default_params.get("cfg_scale", 7.0))
        )
        raw_sampler = str(
            kwargs.get("sampler_type", self.default_params.get("sampler_type", "euler"))
        )
        effective_sampler = raw_sampler if raw_sampler in _MLX_SAMPLERS else "euler"
        effective_sigma_max = float(
            kwargs.get("sigma_max", self.default_params.get("sigma_max", _RF_SIGMA_MAX))
        )
        # Clamp sigma_max to the RF range [0.01, 2.0] in case a caller passes
        # a DPM++ value like 500.0.
        if effective_sigma_max > 2.0 or effective_sigma_max <= 0.0:
            effective_sigma_max = _RF_SIGMA_MAX

        audio_tensors: list[Tensor] = []

        for i, cond in enumerate(conditioning):
            prompt: str = cond.get("prompt", "")
            negative_prompt: str = cond.get("negative_prompt", "")
            seconds_total = float(cond.get("seconds_total") or 30.0)

            # Offset seed per batch item so multi-item batches aren't identical.
            item_seed: int | None = None if seed == -1 else int(seed) + i

            # mx.array (1, C, T) in [-1, 1]
            # generate() prints "Encoding conditioning...", "Sampling...",
            # "Decoding latents..." — redirect to stderr for MCP safety.
            with stdout_to_stderr():
                audio_mx: Any = self._pipe.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seconds_total=seconds_total,
                    steps=int(effective_steps),
                    cfg_scale=effective_cfg,
                    sigma_max=effective_sigma_max,
                    seed=item_seed,
                    sampler=effective_sampler,
                )

            # Materialise the MLX lazy graph before converting to NumPy.
            mx.eval(audio_mx)

            # mx.array → NumPy → PyTorch.  np.array() copies on conversion
            # from MLX; the explicit .copy() ensures the buffer is writable.
            audio_np: np.ndarray = np.array(audio_mx)           # (1, C, T)
            audio_t: Tensor = (
                torch.from_numpy(audio_np.copy())
                .squeeze(0)   # (C, T)
                .float()
            )
            audio_tensors.append(audio_t)
            del audio_mx
            del audio_np
            del audio_t
            flush_memory_caches(
                include_gc=False,
                include_torch=False,
                include_mlx=True,
            )

        # Pad shorter clips to the batch maximum length, then stack → [B, C, T]
        max_len = max(t.shape[-1] for t in audio_tensors)
        padded = [
            F.pad(t, (0, max_len - t.shape[-1])) if t.shape[-1] < max_len else t
            for t in audio_tensors
        ]
        stacked = torch.stack(padded)
        del padded
        del audio_tensors
        flush_memory_caches(include_gc=False)
        return stacked

    def to(self, device: str | torch.device) -> "MLXDiffusionPipeline":
        """No-op: MLX always runs on the Metal GPU on Apple Silicon.

        Present for ``BasePipeline`` interface compatibility; returns ``self``
        without modification.
        """
        return self

    def unload(self) -> None:
        """Drop MLX model references before cache eviction."""
        self._pipe = None
        flush_memory_caches()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def eval(self) -> "MLXDiffusionPipeline":
        """No-op (MLX models are always in eval mode).  Returns ``self``."""
        return self

    def __repr__(self) -> str:
        return (
            f"MLXDiffusionPipeline("
            f"repo={self._repo_id!r}, "
            f"weights={self._weights_dir!r}, "
            f"sample_rate={self.sample_rate}"
            f")"
        )
