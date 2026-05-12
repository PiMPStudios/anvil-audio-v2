"""
ACEStepPipeline — BasePipeline adapter for ACE-Step (pip-installable).

ACE-Step is installed via pip (``pip install 'anvil-audio[acestep]'`` or
``bash install.sh``).  The pip package exposes ``ACEStepPipeline`` in
``acestep.pipeline_ace_step`` and auto-downloads checkpoints to
``~/.cache/ace-step/`` on first run.

Vocabulary mapping
------------------
Anvil conditioning key  →  ACE-Step parameter
``prompt``              →  ``prompt`` (style / genre tags)
``lyrics``              →  ``lyrics`` (vocal content; ``""`` for instrumental)
``seconds_total``       →  ``audio_duration``
``seed``                →  ``manual_seeds``
``steps``               →  ``infer_step``
``cfg_scale``           →  ``guidance_scale``
``scheduler_type``      →  ``scheduler_type`` (``"euler"`` or ``"heun"``)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from anvil_audio.core.interfaces import BasePipeline


class ACEStepPipeline(BasePipeline):
    """``BasePipeline`` adapter for the pip-installed ACE-Step package.

    Wraps ``acestep.pipeline_ace_step.ACEStepPipeline``.  Checkpoints are
    downloaded automatically to ``~/.cache/ace-step/`` on first generation.

    Args:
        project_root:    Deprecated — no longer used.  ACE-Step is now
                         pip-installed and does not require a cloned repo.
                         Accepted for backwards compatibility; has no effect.
        config_path:     Ignored — ACE-Step v0.2+ selects its checkpoint
                         automatically.  Accepted for backwards compatibility.
        device:          Not used directly; ACE-Step v0.2+ auto-detects the
                         best available device (CUDA → MPS → CPU).
        offload_to_cpu:  Enable CPU offloading to reduce peak VRAM.
        lm_model_path:   Deprecated — ACE-Step v0.2+ integrates the lyric
                         planner internally.  Has no effect.
        default_params:  Generation parameter overrides.  Recognised keys:
                         ``steps`` (int), ``cfg_scale`` (float),
                         ``scheduler_type`` (str).
    """

    _SAMPLE_RATE: int = 48000

    def __init__(
        self,
        project_root: str | None = None,
        config_path: str = "acestep-v1.5-sft",
        device: str = "auto",
        offload_to_cpu: bool = False,
        lm_model_path: str | None = None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        # Legacy: inject project root for cloned-repo installs.
        if project_root is not None:
            resolved_root = str(Path(project_root).resolve())
            if resolved_root not in sys.path:
                sys.path.insert(0, resolved_root)

        try:
            from acestep.pipeline_ace_step import (  # type: ignore[import]
                ACEStepPipeline as _Upstream,
            )
        except ImportError as exc:
            raise ImportError(
                "ACE-Step could not be imported. Install it with:\n\n"
                "    pip install 'anvil-audio[acestep]'\n\n"
                "or run the platform install script:\n\n"
                "    bash install.sh\n\n"
                f"Underlying error: {exc}"
            ) from exc

        print(
            f"->->-> Initialising ACE-Step  device=auto  offload={offload_to_cpu}",
            file=sys.stderr,
        )
        self._pipe = _Upstream(cpu_offload=offload_to_cpu)

        self.default_params: dict[str, Any] = default_params or {
            "steps": 60,
            "cfg_scale": 15.0,
            "scheduler_type": "euler",
            "sigma_min": 0.0,
            "sigma_max": 0.0,
        }

    # ------------------------------------------------------------------
    # BasePipeline abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._SAMPLE_RATE

    @property
    def sample_size(self) -> int:
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

        Args:
            conditioning: List of condition dicts with keys ``prompt``,
                          ``lyrics``, and ``seconds_total``.
            steps:        Diffusion steps.  Falls back to
                          ``default_params["steps"]`` (60).
            seed:         RNG seed; -1 draws a random seed each call.
            **kwargs:     Per-call overrides: ``cfg_scale``, ``scheduler_type``.

        Returns:
            Float32 tensor ``[B, 2, T]`` in ``[-1, 1]`` at 48 kHz.
        """
        effective_steps = int(
            steps if steps is not None else self.default_params.get("steps", 60)
        )
        effective_cfg = float(
            kwargs.get("cfg_scale", self.default_params.get("cfg_scale", 15.0))
        )
        effective_method = str(
            kwargs.get("scheduler_type", self.default_params.get("scheduler_type", "euler"))
        )
        manual_seeds = [seed] if seed != -1 else None

        audio_tensors: list[Tensor] = []

        for cond in conditioning:
            raw_dur = cond.get("seconds_total")
            duration = float(raw_dur) if isinstance(raw_dur, (int, float)) and float(raw_dur) > 0 else 30.0

            with tempfile.TemporaryDirectory() as tmp_dir:
                result = self._pipe(
                    prompt=cond.get("prompt", ""),
                    lyrics=cond.get("lyrics", "") or "",
                    audio_duration=duration,
                    infer_step=effective_steps,
                    guidance_scale=effective_cfg,
                    scheduler_type=effective_method,
                    manual_seeds=manual_seeds,
                    save_path=tmp_dir,
                    batch_size=1,
                )
                # result = [wav_path_1, ..., params_dict] — filter to str paths only
                wav_paths = [p for p in result if isinstance(p, str)]
                if not wav_paths:
                    raise RuntimeError("ACE-Step returned no audio files")
                audio, _ = torchaudio.load(wav_paths[0])
                audio_tensors.append(audio.float().cpu())

        max_len = max(t.shape[-1] for t in audio_tensors)
        padded = [
            F.pad(t, (0, max_len - t.shape[-1])) if t.shape[-1] < max_len else t
            for t in audio_tensors
        ]
        return torch.stack(padded)

    def to(self, device: str | torch.device) -> "ACEStepPipeline":
        """No-op: ACE-Step selects its device at construction time."""
        return self

    def eval(self) -> "ACEStepPipeline":
        """No-op: ACE-Step is always in eval mode."""
        return self

    def __repr__(self) -> str:
        return (
            f"ACEStepPipeline("
            f"sample_rate={self.sample_rate}, "
            f"steps={self.default_params.get('steps')}, "
            f"cfg={self.default_params.get('cfg_scale')}"
            f")"
        )
