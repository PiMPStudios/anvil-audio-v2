"""
Concrete pipeline implementation and thin adapter classes.

DiffusionPipeline
-----------------
The primary ``BasePipeline`` implementation.  It wraps an existing
``ConditionedDiffusionModelWrapper`` (or any compatible model returned by
``create_model_from_config``) and delegates generation entirely to the
battle-tested ``generate_diffusion_cond`` function — no sampling logic is
duplicated here.

Component adapters
------------------
Thin ``nn.Module`` wrappers that make the existing model classes conform to
the ``BaseCompressor`` / ``BaseGenerator`` / ``BaseConditioner`` ABCs.  They
hold a reference to the underlying object and delegate every call; no model
weights are copied or modified.

    VAECompressorAdapter    wraps AutoencoderPretransform  → BaseCompressor
    DiTGeneratorAdapter     wraps ConditionedDiffusionModelWrapper → BaseGenerator
    T5ConditionerAdapter    wraps T5Conditioner             → BaseConditioner
    CLAPConditionerAdapter  wraps CLAPTextConditioner       → BaseConditioner
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .interfaces import BaseCompressor, BaseConditioner, BaseGenerator, BasePipeline
from anvil_audio.utils.torch_common import get_best_device

if TYPE_CHECKING:
    # Avoid heavyweight imports at module load; only needed for type checkers.
    from anvil_audio.models.pretransforms import AutoencoderPretransform
    from anvil_audio.models.conditioners import (
        T5Conditioner,
        CLAPTextConditioner,
    )
    from anvil_audio.models.diffusion import ConditionedDiffusionModelWrapper


# ---------------------------------------------------------------------------
# BaseCompressor adapter
# ---------------------------------------------------------------------------

class VAECompressorAdapter(BaseCompressor):
    """Adapts an ``AutoencoderPretransform`` to the ``BaseCompressor`` interface.

    Args:
        pretransform: A fully-initialised ``AutoencoderPretransform`` instance.
    """

    def __init__(self, pretransform: "AutoencoderPretransform") -> None:
        super().__init__()
        self._pretransform = pretransform

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def downsampling_ratio(self) -> int:
        return self._pretransform.downsampling_ratio  # type: ignore[return-value]

    @property
    def latent_dim(self) -> int:
        return self._pretransform.encoded_channels  # type: ignore[return-value]

    @property
    def io_channels(self) -> int:
        return self._pretransform.io_channels  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def encode(self, audio: Tensor) -> Tensor:
        """Encode *audio* to latents via the wrapped pretransform.

        Args:
            audio: ``[B, C, T]`` float waveform tensor.

        Returns:
            Latent tensor ``[B, latent_dim, T // downsampling_ratio]``.
        """
        return self._pretransform.encode(audio)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode *latents* to a waveform via the wrapped pretransform.

        Args:
            latents: ``[B, latent_dim, L]`` float tensor.

        Returns:
            Waveform tensor ``[B, io_channels, L * downsampling_ratio]``.
        """
        return self._pretransform.decode(latents)

    # ------------------------------------------------------------------
    # Delegate PyTorch module operations to the inner model
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward is encode; use ``decode`` for the inverse."""
        return self.encode(x)


# ---------------------------------------------------------------------------
# BaseGenerator adapter
# ---------------------------------------------------------------------------

class DiTGeneratorAdapter(BaseGenerator):
    """Adapts a ``ConditionedDiffusionModelWrapper`` to the ``BaseGenerator``
    interface.

    The adapter exposes only the raw denoising step: given noise and
    pre-encoded conditioning tensors it runs the sampler and returns the
    denoised latent.  Conditioning encoding and latent decoding remain the
    responsibility of the pipeline.

    Args:
        wrapper: A ``ConditionedDiffusionModelWrapper`` instance.
    """

    def __init__(self, wrapper: "ConditionedDiffusionModelWrapper") -> None:
        super().__init__()
        self._wrapper = wrapper

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def io_channels(self) -> int:
        return self._wrapper.io_channels  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def generate(
        self,
        noise: Tensor,
        conditioning: dict[str, Any],
        steps: int,
        **kwargs: Any,
    ) -> Tensor:
        """Run the diffusion denoising process.

        Args:
            noise:        Initial noise ``[B, io_channels, L]``.
            conditioning: Pre-encoded conditioning dict as returned by
                          ``ConditionedDiffusionModelWrapper.get_conditioning_inputs``.
            steps:        Number of denoising steps.
            **kwargs:     Forwarded to the sampler (cfg_scale, sigma_min, …).

        Returns:
            Denoised latent tensor of the same shape as *noise*.
        """
        from anvil_audio.inference.sampling import sample_k, sample_rf, sample_rf_denoiser

        diff_objective = self._wrapper.diffusion_objective
        cfg_scale = kwargs.pop("cfg_scale", 6.0)

        if diff_objective == "v":
            return sample_k(
                self._wrapper.model,
                noise,
                init_data=None,
                mask=None,
                steps=steps,
                cfg_scale=cfg_scale,
                batch_cfg=True,
                rescale_cfg=True,
                **conditioning,
                **kwargs,
            )
        elif diff_objective == "rectified_flow":
            kwargs.pop("sigma_min", None)
            kwargs.pop("sampler_type", None)
            return sample_rf(
                self._wrapper.model,
                noise,
                init_data=None,
                steps=steps,
                cfg_scale=cfg_scale,
                batch_cfg=True,
                rescale_cfg=True,
                **conditioning,
                **kwargs,
            )
        elif diff_objective == "rf_denoiser":
            kwargs.pop("sigma_min", None)
            kwargs.pop("sampler_type", None)
            return sample_rf_denoiser(
                self._wrapper.model,
                noise,
                init_data=None,
                steps=steps,
                **conditioning,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown diffusion objective: '{diff_objective}'")

    def forward(self, noise: Tensor, conditioning: dict[str, Any], steps: int, **kwargs: Any) -> Tensor:  # type: ignore[override]
        return self.generate(noise, conditioning, steps, **kwargs)


# ---------------------------------------------------------------------------
# BaseConditioner adapters
# ---------------------------------------------------------------------------

class T5ConditionerAdapter(BaseConditioner):
    """Adapts an existing ``T5Conditioner`` to the ``BaseConditioner`` interface.

    Args:
        conditioner: A fully-initialised ``T5Conditioner`` instance.
    """

    def __init__(self, conditioner: "T5Conditioner") -> None:
        super().__init__()
        self._conditioner = conditioner

    @property
    def output_dim(self) -> int:
        return self._conditioner.output_dim  # type: ignore[return-value]

    def encode(self, inputs: list[str]) -> tuple[Tensor, Tensor]:
        """Encode a batch of text strings.

        Args:
            inputs: List of B text strings.

        Returns:
            ``(embeddings [B, seq, output_dim], mask [B, seq])``.
        """
        return self._conditioner(inputs)  # type: ignore[return-value]

    def forward(self, inputs: list[str]) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.encode(inputs)


class CLAPConditionerAdapter(BaseConditioner):
    """Adapts an existing ``CLAPTextConditioner`` to the ``BaseConditioner``
    interface.

    Args:
        conditioner: A fully-initialised ``CLAPTextConditioner`` instance.
    """

    def __init__(self, conditioner: "CLAPTextConditioner") -> None:
        super().__init__()
        self._conditioner = conditioner

    @property
    def output_dim(self) -> int:
        return self._conditioner.output_dim  # type: ignore[return-value]

    def encode(self, inputs: list[str]) -> tuple[Tensor, Tensor]:
        """Encode a batch of text strings using CLAP.

        Args:
            inputs: List of B text strings.

        Returns:
            ``(embeddings [B, seq, output_dim], mask [B, seq])``.
        """
        return self._conditioner(inputs)  # type: ignore[return-value]

    def forward(self, inputs: list[str]) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.encode(inputs)


# ---------------------------------------------------------------------------
# DiffusionPipeline
# ---------------------------------------------------------------------------

class DiffusionPipeline(BasePipeline):
    """End-to-end audio generation pipeline backed by a diffusion model.

    This is the primary concrete ``BasePipeline`` implementation.  It wraps
    a ``ConditionedDiffusionModelWrapper`` (as returned by
    ``create_model_from_config`` for ``diffusion_cond`` model types) and
    delegates generation to ``generate_diffusion_cond``.  No sampling logic
    is duplicated.

    Args:
        model:          A model returned by ``create_model_from_config`` — must
                        be a ``ConditionedDiffusionModelWrapper``.
        model_config:   The raw JSON config dict that produced *model*.
        default_params: Generation parameter overrides applied when callers
                        omit individual kwargs.  Keys: steps, cfg_scale,
                        sampler_type, sigma_min, sigma_max.

    Example::

        from anvil_audio.core import load_pipeline
        pipe = load_pipeline("stable-audio-open-1.0")
        audio = pipe.generate(
            [{"prompt": "rain on a tin roof", "seconds_start": 0, "seconds_total": 10}]
        )
    """

    def __init__(
        self,
        model: "ConditionedDiffusionModelWrapper",
        model_config: dict[str, Any],
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._model_config = model_config
        self._default_params: dict[str, Any] = default_params or {
            "steps": 100,
            "cfg_scale": 7.0,
            "sampler_type": "dpmpp-3m-sde",
            "sigma_min": 0.3,
            "sigma_max": 500.0,
        }
        self._device: torch.device = get_best_device()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Audio sample rate of this pipeline in Hz."""
        return self._model_config["sample_rate"]

    @property
    def sample_size(self) -> int:
        """Default audio length in samples."""
        return self._model_config["sample_size"]

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def generate(
        self,
        conditioning: list[dict[str, Any]],
        steps: int | None = None,
        seed: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Generate audio from a list of condition dicts.

        Args:
            conditioning: List of B dicts, typically containing keys
                          ``"prompt"``, ``"seconds_start"``, ``"seconds_total"``.
            steps:        Diffusion steps.  Falls back to ``default_params``
                          if not provided.
            seed:         RNG seed; -1 uses a random seed.
            **kwargs:     Any kwarg overrides ``default_params`` for this call.
                          Common keys: cfg_scale, sampler_type, sigma_min,
                          sigma_max, cfg_rescale, init_audio, init_noise_level.

        Returns:
            Float audio tensor ``[B, channels, T]`` in ``[-1, 1]``.
        """
        from anvil_audio.inference.generation import generate_diffusion_cond

        # Build effective params: defaults → per-call overrides
        params = {**self._default_params, **kwargs}
        params_steps = params.pop("steps", None)
        effective_steps = steps if steps is not None else params_steps

        return generate_diffusion_cond(
            self._model,
            steps=effective_steps,
            cfg_scale=params.pop("cfg_scale", 7.0),
            conditioning=conditioning,
            sample_size=self.sample_size,
            seed=seed,
            device=self._device,
            sampler_type=params.pop("sampler_type", "dpmpp-3m-sde"),
            sigma_min=params.pop("sigma_min", 0.3),
            sigma_max=params.pop("sigma_max", 500.0),
            **params,
        )

    def to(self, device: str | torch.device) -> "DiffusionPipeline":
        """Move the underlying model to *device* and return ``self``.

        Args:
            device: Target device (``"cuda"``, ``"mps"``, ``"cpu"``, or a
                    ``torch.device``).

        Returns:
            ``self`` for chaining.
        """
        self._device = torch.device(device)
        self._model.to(self._device)
        return self

    # ------------------------------------------------------------------
    # Optional component accessors
    # ------------------------------------------------------------------

    def get_compressor(self) -> VAECompressorAdapter | None:
        """Return the pipeline's compressor wrapped as a ``BaseCompressor``.

        Returns ``None`` if the underlying model has no pretransform, or if
        the pretransform is not an ``AutoencoderPretransform``.
        """
        from anvil_audio.models.pretransforms import AutoencoderPretransform

        pt = getattr(self._model, "pretransform", None)
        if isinstance(pt, AutoencoderPretransform):
            return VAECompressorAdapter(pt)
        return None

    def get_generator(self) -> DiTGeneratorAdapter:
        """Return the pipeline's generator wrapped as a ``BaseGenerator``."""
        return DiTGeneratorAdapter(self._model)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def model(self) -> "ConditionedDiffusionModelWrapper":
        """The underlying ``ConditionedDiffusionModelWrapper``."""
        return self._model

    @property
    def model_config(self) -> dict[str, Any]:
        """The raw JSON model config dict."""
        return self._model_config

    @property
    def default_params(self) -> dict[str, Any]:
        """Generation defaults for this pipeline (copy)."""
        return dict(self._default_params)

    def eval(self) -> "DiffusionPipeline":
        """Set the underlying model to evaluation mode and return ``self``."""
        self._model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"DiffusionPipeline("
            f"model_type={self._model_config.get('model_type', '?')!r}, "
            f"sample_rate={self.sample_rate}, "
            f"sample_size={self.sample_size}, "
            f"device={self._device}"
            f")"
        )
