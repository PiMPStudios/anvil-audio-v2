"""
ACEStepPipeline — BasePipeline adapter for ACE-Step v1.5.

ACE-Step is kept as a separate package (pip-installable or local source tree).
This module imports from ``acestep`` at construction time only; a missing
installation raises a clear ``ImportError`` with remediation instructions
rather than a cryptic ``ModuleNotFoundError`` at import time.

Vocabulary mapping
------------------
Anvil conditioning key  →  ACE-Step parameter
``prompt``              →  ``captions`` (style / genre tags)
``lyrics``              →  ``lyrics`` (vocal content; use ``"[Instrumental]"``
                           or leave blank for instrumental output)
``negative_prompt``     →  ``lm_negative_prompt`` (ACE-Step LM/thinking control)
``seconds_total``       →  ``audio_duration`` (generation length in seconds)
``seed``                →  passed as the ``seed`` argument
``steps``               →  ``inference_steps``
``cfg_scale``           →  ``guidance_scale``
``scheduler_type``      →  ``infer_method`` (``"ode"`` or ``"sde"``)

LM thinking path
----------------
When ``lm_model_path`` is provided (and the ``LLMHandler`` import succeeds),
the adapter initialises ACE-Step's 5 Hz LM alongside the DiT.  Generation is
then delegated to ACE-Step's upstream ``generate_music`` orchestration, so
``thinking=True`` can generate semantic audio-code hints and metadata.  The
built-in SFT entry keeps thinking disabled by default to match AnvilApp's
known-good direct DiT conditioning path.

If the LM is unavailable (``lm_model_path=None``, import error, or
initialisation failure) the adapter falls back gracefully to DiT-only
generation and logs a warning.

Usage
-----
::

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    pipe = ACEStepPipeline(
        project_root="/path/to/ACE-Step",
        config_path="acestep-v15-turbo",
        device="auto",
        lm_model_path="acestep-5Hz-lm-1.7B",
    )
    audio = pipe.generate(
        [{"prompt": "upbeat indie pop", "lyrics": "[verse]\\nHello world", "seconds_total": 30}]
    )
    # audio: Tensor [1, 2, T] at 48 kHz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from anvil_audio.core.interfaces import BasePipeline
from anvil_audio.utils.stdio_guard import stdout_to_stderr

_XL_CHECKPOINT_PREFIX = "acestep-v15-xl-"
_XL_AUTO_DOWNLOAD_ENV = "ANVIL_ACESTEP_ALLOW_XL_AUTO_DOWNLOAD"
_CHECKPOINT_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.bin.index.json",
)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a bool-ish environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _acestep_checkpoints_dir(project_root: str) -> Path:
    """Return the checkpoint directory ACE-Step will use for this load."""
    env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(project_root) / "checkpoints"


def _is_xl_checkpoint(config_path: str) -> bool:
    """Return whether config_path names one of ACE-Step's XL DiT variants."""
    return Path(config_path).name.startswith(_XL_CHECKPOINT_PREFIX)


def _checkpoint_has_weights(checkpoint_dir: Path) -> bool:
    """Return whether a checkpoint directory appears to contain model weights."""
    if not checkpoint_dir.is_dir():
        return False
    return any(
        (checkpoint_dir / filename).exists()
        for filename in _CHECKPOINT_WEIGHT_FILENAMES
    )


def _raise_for_missing_xl_checkpoint(project_root: str, config_path: str) -> None:
    """Block surprise auto-downloads for large optional ACE-Step XL checkpoints."""
    if not _is_xl_checkpoint(config_path):
        return
    if _env_bool(_XL_AUTO_DOWNLOAD_ENV, False):
        return

    checkpoints_dir = _acestep_checkpoints_dir(project_root)
    checkpoint_dir = checkpoints_dir / config_path
    if _checkpoint_has_weights(checkpoint_dir):
        return

    raise RuntimeError(
        "ACE-Step XL checkpoint is not installed.\n"
        f"  model        : {config_path}\n"
        f"  expected at  : {checkpoint_dir}\n\n"
        "XL checkpoints are large optional downloads, so Anvil will not "
        "auto-download them during model load.\n\n"
        "Install this checkpoint explicitly, then retry:\n\n"
        f"    acestep-download --dir {checkpoints_dir} --model {config_path}\n\n"
        f"To intentionally restore upstream auto-download behavior for this "
        f"process, set {_XL_AUTO_DOWNLOAD_ENV}=1."
    )


class ACEStepPipeline(BasePipeline):
    """``BasePipeline`` adapter for ACE-Step v1.5.

    Wraps ``AceStepHandler`` (DiT/VAE) and optionally ``LLMHandler`` (5 Hz LM)
    so ACE-Step integrates with Anvil's registry, CLI batch
    generation, ``OutputManager``, and Gradio UI without copying any of
    ACE-Step's model weights or inference logic.

    Args:
        project_root:    Absolute path to the ACE-Step repository root (the
                         directory that contains ``checkpoints/``).  Can be
                         a cloned git repo or an installed package directory.
                         Pass ``None`` (default) when ACE-Step is pip-installed
                         and importable without path manipulation.
        config_path:     Model variant — ``"acestep-v15-turbo"`` (fast) or
                         ``"acestep-v15-sft"`` (full quality).  Defaults to
                         ``"acestep-v15-turbo"``.
        device:          Device hint passed to ``initialize_service`` — one
                         of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.
                         ``"auto"`` lets ACE-Step pick the best available.
        offload_to_cpu:  Enable sequential CPU offloading to lower peak VRAM
                         usage at the cost of slower generation.
        lm_model_path:   Path to the 5 Hz LM checkpoint, either
                         relative to ``<project_root>/checkpoints/`` (e.g.
                         ``"acestep-5Hz-lm-1.7B"``) or an absolute path.
                         ``None`` disables LM thinking/code hints.
        default_params:  Generation parameter overrides applied when callers
                         omit individual kwargs.  Recognised keys:
                         ``steps``, ``cfg_scale``, ``scheduler_type``,
                         ``shift``, ``thinking``, ``dcw_enabled``,
                         ``sigma_min``, ``sigma_max``, plus init-time ACE-Step
                         options such as ``offload_to_cpu``,
                         ``offload_dit_to_cpu``, ``quantization``,
                         ``prefer_source``, and ``vae_checkpoint``.
    """

    #: ACE-Step v1.5 VAE outputs 48 kHz stereo audio.
    _SAMPLE_RATE: int = 48000

    def __init__(
        self,
        project_root: str | None = None,
        config_path: str = "acestep-v15-turbo",
        device: str = "auto",
        offload_to_cpu: bool = False,
        lm_model_path: str | None = None,
        default_params: dict[str, Any] | None = None,
        use_mlx_dit: bool | None = None,
    ) -> None:
        # Inject project root into sys.path only when explicitly provided.
        # When None, use Anvil's cache-backed project root for checkpoints while
        # importing ACE-Step from the installed package.
        if project_root is not None:
            resolved_root = str(Path(project_root).resolve())
            is_source_checkout = (Path(resolved_root) / "acestep").is_dir()
            if is_source_checkout and resolved_root not in sys.path:
                sys.path.insert(0, resolved_root)
        else:
            resolved_root = str(Path.home() / ".cache" / "anvil-audio" / "acestep")

        try:
            from acestep.handler import AceStepHandler  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ACE-Step could not be imported. Install it with:\n\n"
                "    pip install anvil-audio[acestep]\n\n"
                "or run the platform install script:\n\n"
                "    bash install.sh\n\n"
                f"Underlying error: {exc}"
            ) from exc

        self._handler: Any = AceStepHandler()
        self._project_root: str | None = resolved_root
        self._config_path: str = config_path
        self._device_str: str = device

        self.default_params: dict[str, Any] = default_params or {
            "steps": 8,
            "cfg_scale": 7.0,
            "scheduler_type": "ode",
            # sigma_min / sigma_max are Stable-Audio concepts; store 0.0 so
            # GenerationMetadata can serialise them without KeyError.
            "sigma_min": 0.0,
            "sigma_max": 0.0,
        }
        if "offload_to_cpu" in self.default_params:
            offload_to_cpu = bool(self.default_params["offload_to_cpu"])
        offload_dit_to_cpu = bool(
            self.default_params.get("offload_dit_to_cpu", False)
        )
        quantization = self.default_params.get("quantization")
        prefer_source = self.default_params.get("prefer_source")
        vae_checkpoint = self.default_params.get("vae_checkpoint")

        if use_mlx_dit is None and "use_mlx_dit" in self.default_params:
            use_mlx_dit = bool(self.default_params["use_mlx_dit"])
        if use_mlx_dit is None:
            use_mlx_dit = _env_bool("ANVIL_ACESTEP_USE_MLX_DIT", sys.platform == "darwin")
        self._use_mlx_dit = bool(use_mlx_dit)

        # Keep ACE-Step internals pointed at the same checkpoint root that Anvil
        # is about to pass into initialize_service.  This avoids stale shell env
        # values sending downloads/logs to an old checkout.
        os.makedirs(resolved_root, exist_ok=True)
        os.environ["ACESTEP_PROJECT_ROOT"] = resolved_root
        _raise_for_missing_xl_checkpoint(resolved_root, config_path)

        # On macOS, set ACESTEP_LM_BACKEND=mlx unless the user has already
        # set it.
        if sys.platform == "darwin" and "ACESTEP_LM_BACKEND" not in os.environ:
            os.environ["ACESTEP_LM_BACKEND"] = "mlx"

        on_apple_silicon = sys.platform == "darwin"
        print(
            f"->->-> Initialising ACE-Step  "
            f"config={config_path!r}  device={device!r}  "
            f"offload={offload_to_cpu}"
            + (f"  dit_offload={offload_dit_to_cpu}" if offload_dit_to_cpu else "")
            + (f"  quantization={quantization}" if quantization else "")
            + ("  mlx=DiT+VAE+LM" if on_apple_silicon and self._use_mlx_dit else ""),
            file=sys.stderr,
        )
        # initialize_service loads model weights and may print to stdout;
        # redirect so MCP stdio is not corrupted.
        # use_mlx_dit activates native MLX acceleration for the DiT and VAE
        # when device resolves to "mps" or "cpu" on Apple Silicon.  It remains
        # overrideable for backend-parity debugging.
        with stdout_to_stderr():
            status, success = self._handler.initialize_service(
                project_root=self._project_root,
                config_path=config_path,
                device=device,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                quantization=quantization,
                prefer_source=prefer_source,
                use_mlx_dit=self._use_mlx_dit,
                vae_checkpoint=vae_checkpoint,
            )
        if not success:
            raise RuntimeError(
                f"ACE-Step model initialisation failed.\n"
                f"  project_root : {self._project_root}\n"
                f"  config_path  : {config_path!r}\n"
                f"  device       : {device!r}\n"
                f"  Status       : {status}"
            )
        print(f"->->-> ACE-Step ready  {status}", file=sys.stderr)

        # ------------------------------------------------------------------
        # 5 Hz LM thinking/code-hint path (optional)
        # ------------------------------------------------------------------
        self._lm_handler: Any = None
        self._lm_available: bool = False
        self._lm_model_name: str = ""

        if lm_model_path is not None:
            self._init_lm_planner(lm_model_path, device, offload_to_cpu)

        self._loaded_lora_adapters: dict[str, str] = {}

    def _init_lm_planner(
        self,
        lm_model_path: str,
        device: str,
        offload_to_cpu: bool,
    ) -> None:
        """Attempt to initialise the 5 Hz LM.

        Failures are non-fatal: a warning is printed and generation falls back
        to DiT-only mode.

        Args:
            lm_model_path:  Relative path (within ``<project_root>/checkpoints/``)
                            or absolute path to the LM checkpoint directory.
            device:         Device string forwarded to ``LLMHandler.initialize``.
            offload_to_cpu: CPU-offload flag forwarded to ``LLMHandler.initialize``.
        """
        try:
            from acestep.llm_inference import LLMHandler  # type: ignore[import]
        except ImportError as exc:
            print(
                f"[ACEStep] WARNING: LLMHandler import failed — LM thinking disabled.\n"
                f"  {exc}",
                file=sys.stderr,
            )
            return

        # Resolve lm_model_path: if not absolute, treat as relative to
        # <project_root>/checkpoints/.
        if os.path.isabs(lm_model_path):
            resolved_lm = lm_model_path
        else:
            resolved_lm = os.path.join(self._project_root, "checkpoints", lm_model_path)

        checkpoint_dir = os.path.join(self._project_root, "checkpoints")
        lm_backend = os.environ.get("ACESTEP_LM_BACKEND", "vllm")

        print(
            f"->->-> Initialising ACE-Step LM  "
            f"model={os.path.basename(resolved_lm)!r}  backend={lm_backend!r}",
            file=sys.stderr,
        )

        try:
            lm_handler: Any = LLMHandler()
            with stdout_to_stderr():
                lm_status, lm_success = lm_handler.initialize(
                    checkpoint_dir=checkpoint_dir,
                    lm_model_path=resolved_lm,
                    backend=lm_backend,
                    device=device,
                    offload_to_cpu=offload_to_cpu,
                )
        except Exception as exc:
            print(
                f"[ACEStep] WARNING: LM initialisation failed — "
                f"falling back to DiT-only generation.\n  {exc}",
                file=sys.stderr,
            )
            return

        if not lm_success:
            print(
                f"[ACEStep] WARNING: LM initialisation reported failure "
                f"({lm_status!r}) — falling back to DiT-only generation.",
                file=sys.stderr,
            )
            return

        self._lm_handler = lm_handler
        self._lm_available = True
        self._lm_model_name = os.path.basename(resolved_lm)
        print(f"->->-> ACE-Step LM ready  {lm_status}", file=sys.stderr)

    # ------------------------------------------------------------------
    # BasePipeline abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """48 kHz — native output sample rate of ACE-Step v1.5."""
        return self._SAMPLE_RATE

    @property
    def sample_size(self) -> int:
        """Default sample count for a 30-second clip (48 000 × 30)."""
        return self._SAMPLE_RATE * 30

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
        """Generate a batch of stereo music waveforms via ACE-Step.

        Each condition dict may contain:

        ============  =====================================================
        Key           Description
        ============  =====================================================
        ``prompt``    Tags / style caption (e.g. ``"upbeat indie pop"``).
        ``lyrics``    Lyric text.  Omit or pass ``"[Instrumental]"`` for
                      instrumental output.
        ``negative_prompt`` Negative text for ACE-Step's LM/thinking stage.
                      Direct DiT-only generation preserves this in metadata but
                      upstream ACE-Step does not expose a DiT negative prompt.
        ``seconds_total`` Target duration in seconds.  ``None`` or ``≤0``
                      lets ACE-Step choose automatically.
        ============  =====================================================

        When the LM is active, ACE-Step's upstream thinking path can generate
        semantic audio-code hints and metadata before the DiT pass.  Enable it
        per model or per call with ``thinking=True``.

        Args:
            conditioning: List of B condition dicts.
            steps:        Diffusion steps.  Falls back to
                          ``default_params["steps"]`` (8 for turbo).
            seed:         RNG seed; -1 draws a random seed each call.
            **kwargs:     Per-call overrides:
                          - ``cfg_scale`` (float): guidance strength
                          - ``scheduler_type`` (str): ``"ode"`` or ``"sde"``

        Returns:
            Float32 tensor ``[B, 2, T]`` in ``[-1, 1]`` at 48 kHz.

        Raises:
            RuntimeError: If ACE-Step returns a failure payload.
        """
        effective_steps = (
            steps if steps is not None else self.default_params.get("steps", 8)
        )
        effective_cfg = float(
            kwargs.get("cfg_scale", self.default_params.get("cfg_scale", 7.0))
        )
        effective_method = str(
            kwargs.get(
                "scheduler_type",
                kwargs.get("sampler_type", self.default_params.get("sampler_type", "ode")),
            )
        )
        effective_shift = float(kwargs.get("shift", self.default_params.get("shift", 3.0)))
        effective_use_adg = bool(kwargs.get("use_adg", self.default_params.get("use_adg", False)))
        effective_cfg_start = float(
            kwargs.get(
                "cfg_interval_start",
                self.default_params.get("cfg_interval_start", 0.0),
            )
        )
        effective_cfg_end = float(
            kwargs.get(
                "cfg_interval_end",
                self.default_params.get("cfg_interval_end", 1.0),
            )
        )
        effective_lm_cfg = float(
            kwargs.get("lm_cfg_scale", self.default_params.get("lm_cfg_scale", 2.0))
        )
        effective_thinking = bool(
            kwargs.get("thinking", self.default_params.get("thinking", True))
        )
        effective_lm_temperature = float(
            kwargs.get(
                "lm_temperature",
                self.default_params.get("lm_temperature", 0.85),
            )
        )
        effective_lm_top_k = int(
            kwargs.get("lm_top_k", self.default_params.get("lm_top_k", 0))
        )
        effective_lm_top_p = float(
            kwargs.get("lm_top_p", self.default_params.get("lm_top_p", 0.9))
        )
        effective_lm_negative_prompt = str(
            kwargs.get(
                "lm_negative_prompt",
                self.default_params.get("lm_negative_prompt", "NO USER INPUT"),
            )
        )
        effective_use_cot_metas = bool(
            kwargs.get("use_cot_metas", self.default_params.get("use_cot_metas", True))
        )
        effective_use_cot_caption = bool(
            kwargs.get(
                "use_cot_caption",
                self.default_params.get("use_cot_caption", False),
            )
        )
        effective_use_cot_language = bool(
            kwargs.get(
                "use_cot_language",
                self.default_params.get("use_cot_language", True),
            )
        )
        effective_use_constrained_decoding = bool(
            kwargs.get(
                "use_constrained_decoding",
                self.default_params.get("use_constrained_decoding", True),
            )
        )
        effective_dcw_enabled = bool(
            kwargs.get("dcw_enabled", self.default_params.get("dcw_enabled", False))
        )
        effective_dcw_mode = str(
            kwargs.get("dcw_mode", self.default_params.get("dcw_mode", "double"))
        )
        effective_dcw_scaler = float(
            kwargs.get("dcw_scaler", self.default_params.get("dcw_scaler", 0.05))
        )
        effective_dcw_high_scaler = float(
            kwargs.get(
                "dcw_high_scaler",
                self.default_params.get("dcw_high_scaler", 0.02),
            )
        )
        effective_dcw_wavelet = str(
            kwargs.get("dcw_wavelet", self.default_params.get("dcw_wavelet", "haar"))
        )
        effective_sampler_mode = str(
            kwargs.get("sampler_mode", self.default_params.get("sampler_mode", "euler"))
        )
        effective_velocity_norm_threshold = float(
            kwargs.get(
                "velocity_norm_threshold",
                self.default_params.get("velocity_norm_threshold", 0.0),
            )
        )
        effective_velocity_ema_factor = float(
            kwargs.get(
                "velocity_ema_factor",
                self.default_params.get("velocity_ema_factor", 0.0),
            )
        )
        use_random_seed = seed == -1

        audio_tensors: list[Tensor] = []

        for cond in conditioning:
            tags: str = cond.get("prompt", "")
            lyrics: str = cond.get("lyrics", "")
            negative_prompt = str(
                cond.get(
                    "lm_negative_prompt",
                    cond.get("negative_prompt", effective_lm_negative_prompt),
                )
            )

            raw_dur = cond.get("seconds_total")
            duration: float | None = (
                float(raw_dur) if isinstance(raw_dur, (int, float)) and float(raw_dur) > 0
                else None
            )

            try:
                from acestep.inference import (  # type: ignore[import]
                    GenerationConfig,
                    GenerationParams,
                    generate_music as upstream_generate_music,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "ACE-Step inference API could not be imported after model "
                    f"initialisation: {exc}"
                ) from exc

            gen_params = GenerationParams(
                task_type="text2music",
                caption=tags,
                lyrics=lyrics,
                vocal_language=str(cond.get("vocal_language", "en")),
                bpm=cond.get("bpm"),
                keyscale=str(cond.get("keyscale", cond.get("key_scale", ""))),
                timesignature=str(
                    cond.get("timesignature", cond.get("time_signature", ""))
                ),
                duration=duration if duration is not None else -1.0,
                inference_steps=int(effective_steps),
                guidance_scale=effective_cfg,
                infer_method=effective_method,
                shift=effective_shift,
                use_adg=effective_use_adg,
                cfg_interval_start=effective_cfg_start,
                cfg_interval_end=effective_cfg_end,
                thinking=effective_thinking,
                lm_temperature=effective_lm_temperature,
                lm_cfg_scale=effective_lm_cfg,
                lm_top_k=effective_lm_top_k,
                lm_top_p=effective_lm_top_p,
                lm_negative_prompt=negative_prompt,
                use_cot_metas=effective_use_cot_metas,
                use_cot_caption=effective_use_cot_caption,
                use_cot_language=effective_use_cot_language,
                use_constrained_decoding=effective_use_constrained_decoding,
                sampler_mode=effective_sampler_mode,
                velocity_norm_threshold=effective_velocity_norm_threshold,
                velocity_ema_factor=effective_velocity_ema_factor,
                dcw_enabled=effective_dcw_enabled,
                dcw_mode=effective_dcw_mode,
                dcw_scaler=effective_dcw_scaler,
                dcw_high_scaler=effective_dcw_high_scaler,
                dcw_wavelet=effective_dcw_wavelet,
            )
            gen_config = GenerationConfig(
                batch_size=1,
                use_random_seed=use_random_seed,
                seeds=None if use_random_seed else [int(seed)],
                audio_format="wav",
            )

            with stdout_to_stderr():
                result = upstream_generate_music(
                    self._handler,
                    self._lm_handler if self._lm_available else None,
                    params=gen_params,
                    config=gen_config,
                    save_dir=None,
                )

            if not result.success:
                raise RuntimeError(
                    f"ACE-Step generation failed: "
                    f"{result.error or result.status_message or 'unknown error'}"
                )

            audios: list[dict[str, Any]] = result.audios
            if not audios:
                raise RuntimeError(
                    "ACE-Step returned an empty audio list; "
                    "check the handler logs for details."
                )

            # audios[0] = {"tensor": Tensor[C, T], "sample_rate": int}
            audio_t: Tensor = audios[0]["tensor"].float()
            if audio_t.dim() == 1:
                # Mono: unsqueeze to [1, T]
                audio_t = audio_t.unsqueeze(0)

            audio_tensors.append(audio_t.cpu())

        # Pad shorter clips to the batch maximum length, then stack → [B, C, T]
        max_len = max(t.shape[-1] for t in audio_tensors)
        padded = [
            F.pad(t, (0, max_len - t.shape[-1])) if t.shape[-1] < max_len else t
            for t in audio_tensors
        ]
        return torch.stack(padded)

    def apply_lora_adapter(
        self,
        lora_path: str,
        *,
        adapter_name: str | None = None,
        scale: float = 1.0,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Load and activate an ACE-Step LoRA/LoKr adapter.

        The actual injection/scaling is delegated to ACE-Step's handler. This
        wrapper makes the operation idempotent for the common case where the UI
        or CLI sends the same adapter path across multiple generations.
        """
        path = str(Path(lora_path).expanduser().resolve())
        effective_name = adapter_name or Path(path).name

        loaded_path = self._loaded_lora_adapters.get(effective_name)
        if loaded_path != path:
            message = self._handler.add_lora(path, adapter_name=adapter_name)
            if not str(message).startswith("✅"):
                raise RuntimeError(str(message))
            self._loaded_lora_adapters[effective_name] = path
        else:
            message = f"Adapter already loaded: {effective_name}"

        active_message = ""
        if adapter_name and hasattr(self._handler, "set_active_lora_adapter"):
            active_message = str(self._handler.set_active_lora_adapter(adapter_name))

        scale_message = str(self._handler.set_lora_scale(effective_name, scale))
        enabled_message = str(self._handler.set_use_lora(bool(enabled)))
        status = (
            self._handler.get_lora_status()
            if hasattr(self._handler, "get_lora_status")
            else {}
        )
        return {
            "message": str(message),
            "active_message": active_message,
            "scale_message": scale_message,
            "enabled_message": enabled_message,
            "path": path,
            "adapter_name": effective_name,
            "scale": float(scale),
            "enabled": bool(enabled),
            "status": status,
        }

    def to(self, device: str | torch.device) -> "ACEStepPipeline":
        """No-op: ACE-Step initialises its device at construction time.

        ACE-Step does not support moving weights post-initialisation.  This
        method is present for ``BasePipeline`` interface compatibility and
        returns ``self`` without modification.
        """
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def eval(self) -> "ACEStepPipeline":
        """No-op (ACE-Step is always in eval mode).  Returns ``self``."""
        return self

    def __repr__(self) -> str:
        lm_info = (
            f", lm={self._lm_model_name!r}"
            if self._lm_available
            else ", lm=disabled"
        )
        return (
            f"ACEStepPipeline("
            f"config={self._config_path!r}, "
            f"device={self._device_str!r}, "
            f"sample_rate={self.sample_rate}"
            f"{lm_info}"
            f")"
        )
