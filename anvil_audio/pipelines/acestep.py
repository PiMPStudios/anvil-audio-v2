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
``seconds_total``       →  ``audio_duration`` (generation length in seconds)
``seed``                →  passed as the ``seed`` argument
``steps``               →  ``inference_steps``
``cfg_scale``           →  ``guidance_scale``
``scheduler_type``      →  ``infer_method`` (``"ode"`` or ``"sde"``)

LM lyric planner
----------------
When ``lm_model_path`` is provided (and the ``LLMHandler`` import succeeds),
the adapter initialises ACE-Step's 5 Hz LM lyric-planner alongside the DiT.
For any generation request that contains non-empty, non-instrumental lyrics the
planner runs first (``infer_type="llm_dit"``) and the resulting ``audio_codes``
are forwarded to ``generate_music`` as ``audio_code_string``.  This matches the
full quality path used by the standalone ACE-Step Gradio UI.

If the LM planner is unavailable (``lm_model_path=None``, import error, or
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

# Lyrics values that indicate instrumental generation — LM planning is skipped.
_INSTRUMENTAL_MARKERS = {"", "[instrumental]"}


class ACEStepPipeline(BasePipeline):
    """``BasePipeline`` adapter for ACE-Step v1.5.

    Wraps ``AceStepHandler`` (DiT/VAE) and optionally ``LLMHandler`` (5 Hz LM
    lyric planner) so ACE-Step integrates with Anvil's registry, CLI batch
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
        lm_model_path:   Path to the 5 Hz LM lyric-planner checkpoint, either
                         relative to ``<project_root>/checkpoints/`` (e.g.
                         ``"acestep-5Hz-lm-1.7B"``) or an absolute path.
                         ``None`` disables LM planning; generation falls back
                         to DiT-only with raw lyric text.
        default_params:  Generation parameter overrides applied when callers
                         omit individual kwargs.  Recognised keys:
                         ``steps``, ``cfg_scale``, ``scheduler_type``,
                         ``sigma_min``, ``sigma_max`` (last two stored as 0.0
                         so ``GenerationMetadata`` serialises cleanly).
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
    ) -> None:
        # Inject project root into sys.path only when explicitly provided.
        # When None, acestep must be importable already (pip-installed).
        if project_root is not None:
            resolved_root = str(Path(project_root).resolve())
            if resolved_root not in sys.path:
                sys.path.insert(0, resolved_root)
        else:
            resolved_root = None

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

        # On macOS, set ACESTEP_LM_BACKEND=mlx unless the user has already
        # set it.  The env var enables the MLX LM lyric-planner backend (2-3x
        # faster than PyTorch on Apple Silicon) for any ACE-Step code paths
        # that read it (API server, subprocesses, future handler versions).
        # Note: the DiT and VAE already use native MLX on Apple Silicon via
        # AceStepHandler's initialize_service(use_mlx_dit=True) default; this
        # env var covers the separate 5Hz LM planner component.
        if sys.platform == "darwin" and "ACESTEP_LM_BACKEND" not in os.environ:
            os.environ["ACESTEP_LM_BACKEND"] = "mlx"

        on_apple_silicon = sys.platform == "darwin"
        print(
            f"->->-> Initialising ACE-Step  "
            f"config={config_path!r}  device={device!r}  "
            f"offload={offload_to_cpu}"
            + ("  mlx=DiT+VAE+LM" if on_apple_silicon else ""),
            file=sys.stderr,
        )
        # initialize_service loads model weights and may print to stdout;
        # redirect so MCP stdio is not corrupted.
        # use_mlx_dit=True (default) activates native MLX acceleration for the
        # DiT and VAE when device resolves to "mps" or "cpu" on Apple Silicon.
        with stdout_to_stderr():
            status, success = self._handler.initialize_service(
                project_root=self._project_root,
                config_path=config_path,
                device=device,
                offload_to_cpu=offload_to_cpu,
                use_mlx_dit=True,
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
        # LM lyric planner (optional)
        # ------------------------------------------------------------------
        self._lm_handler: Any = None
        self._lm_available: bool = False
        self._lm_model_name: str = ""

        if lm_model_path is not None:
            self._init_lm_planner(lm_model_path, device, offload_to_cpu)

    def _init_lm_planner(
        self,
        lm_model_path: str,
        device: str,
        offload_to_cpu: bool,
    ) -> None:
        """Attempt to initialise the 5 Hz LM lyric planner.

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
                f"[ACEStep] WARNING: LLMHandler import failed — LM planning disabled.\n"
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
            f"->->-> Initialising ACE-Step LM planner  "
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
                f"[ACEStep] WARNING: LM planner initialisation failed — "
                f"falling back to DiT-only generation.\n  {exc}",
                file=sys.stderr,
            )
            return

        if not lm_success:
            print(
                f"[ACEStep] WARNING: LM planner initialisation reported failure "
                f"({lm_status!r}) — falling back to DiT-only generation.",
                file=sys.stderr,
            )
            return

        self._lm_handler = lm_handler
        self._lm_available = True
        self._lm_model_name = os.path.basename(resolved_lm)
        print(f"->->-> ACE-Step LM planner ready  {lm_status}", file=sys.stderr)

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
        ``seconds_total`` Target duration in seconds.  ``None`` or ``≤0``
                      lets ACE-Step choose automatically.
        ============  =====================================================

        When the LM lyric planner is active and the request contains
        non-empty, non-instrumental lyrics, the planner runs first and
        its ``audio_codes`` output is forwarded to the DiT for structured
        generation (matching the quality of standalone ACE-Step).  For
        instrumental requests the LM step is skipped automatically.

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
            kwargs.get("scheduler_type", self.default_params.get("scheduler_type", "ode"))
        )
        use_random_seed = seed == -1

        audio_tensors: list[Tensor] = []

        for cond in conditioning:
            tags: str = cond.get("prompt", "")
            lyrics: str = cond.get("lyrics", "")

            raw_dur = cond.get("seconds_total")
            duration: float | None = (
                float(raw_dur) if isinstance(raw_dur, (int, float)) and float(raw_dur) > 0
                else None
            )

            # ----------------------------------------------------------
            # LM lyric-planner pass (when available and lyrics present)
            # ----------------------------------------------------------
            audio_code_string: str = ""
            needs_lm = (
                self._lm_available
                and lyrics.strip().lower() not in _INSTRUMENTAL_MARKERS
            )
            if needs_lm:
                try:
                    with stdout_to_stderr():
                        lm_result = self._lm_handler.generate_with_stop_condition(
                            caption=tags,
                            lyrics=lyrics,
                            infer_type="llm_dit",
                            target_duration=duration,
                        )
                    if lm_result.get("success", False):
                        audio_code_string = lm_result.get("audio_codes", "")
                    else:
                        print(
                            f"[ACEStep] WARNING: LM planner returned failure "
                            f"({lm_result.get('error', 'unknown')}) — "
                            f"continuing without audio codes.",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(
                        f"[ACEStep] WARNING: LM planner call raised {exc!r} — "
                        f"continuing without audio codes.",
                        file=sys.stderr,
                    )

            # ----------------------------------------------------------
            # DiT generation pass
            # ----------------------------------------------------------
            # generate_music may print progress to stdout; redirect so MCP
            # stdio is not corrupted.
            generate_kwargs: dict[str, Any] = dict(
                captions=tags,
                lyrics=lyrics,
                inference_steps=int(effective_steps),
                guidance_scale=effective_cfg,
                seed=seed,
                use_random_seed=use_random_seed,
                audio_duration=duration,
                infer_method=effective_method,
                batch_size=1,
            )
            if audio_code_string:
                generate_kwargs["audio_code_string"] = audio_code_string

            with stdout_to_stderr():
                result = self._handler.generate_music(**generate_kwargs)

            if not result.get("success", False):
                raise RuntimeError(
                    f"ACE-Step generation failed: "
                    f"{result.get('error', 'unknown error')}"
                )

            audios: list[dict[str, Any]] = result.get("audios", [])
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
