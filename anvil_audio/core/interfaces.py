"""
Abstract base classes that define the component contracts for swappable
audio generation components.

Every concrete model plugged into this framework must implement one of:
    BaseCompressor   — audio ↔ latent codec
    BaseGenerator    — latent-space denoising / generation backbone
    BaseConditioner  — raw input → embedding encoder
    BasePipeline     — end-to-end orchestrator

Design notes
------------
* BaseCompressor, BaseGenerator, and BaseConditioner extend nn.Module so
  they participate in PyTorch's device/dtype management normally.
* BasePipeline is a pure Python ABC (not an nn.Module).  It is an
  orchestrator that *holds* modules; callers use its .to() method to move
  everything at once.
* All abstract properties are declared with @property + @abstractmethod so
  subclasses must provide them as computed or plain attributes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# BaseCompressor
# ---------------------------------------------------------------------------

class BaseCompressor(nn.Module, ABC):
    """Interface for any audio autoencoder / codec.

    A compressor is responsible for two symmetric operations:
        encode : waveform  → latent representation
        decode : latents   → waveform

    Subclasses must expose the three shape properties so that callers can
    reason about tensor dimensions without instantiating the model.
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def downsampling_ratio(self) -> int:
        """Temporal compression factor.

        ``encode(audio).shape[-1] == audio.shape[-1] // downsampling_ratio``
        """

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Number of channels in the latent tensor."""

    @property
    @abstractmethod
    def io_channels(self) -> int:
        """Number of audio channels expected by ``encode`` (e.g. 1 or 2)."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, audio: Tensor) -> Tensor:
        """Encode a batch of waveforms to latents.

        Args:
            audio: Float tensor of shape ``[B, C, T]`` where ``C`` matches
                   ``self.io_channels``.

        Returns:
            Latent tensor of shape ``[B, latent_dim, T // downsampling_ratio]``.
        """

    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """Decode a batch of latents back to waveforms.

        Args:
            latents: Float tensor of shape ``[B, latent_dim, L]``.

        Returns:
            Waveform tensor of shape ``[B, io_channels, L * downsampling_ratio]``.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def latent_length(self, audio_length: int) -> int:
        """Return the latent time dimension for a given number of audio samples."""
        return audio_length // self.downsampling_ratio


# ---------------------------------------------------------------------------
# BaseGenerator
# ---------------------------------------------------------------------------

class BaseGenerator(nn.Module, ABC):
    """Interface for any latent-space generative model.

    A generator takes a noise tensor and a dict of pre-encoded conditioning
    tensors and iteratively denoises them into a clean latent.  The caller
    is responsible for sampling from the prior (noise generation) and for
    decoding the output latent back to audio via a ``BaseCompressor``.

    The ``conditioning`` dict is intentionally untyped — different generators
    accept different keys.  Concrete implementations should document the
    keys they consume.
    """

    @property
    @abstractmethod
    def io_channels(self) -> int:
        """Number of latent channels the generator operates on."""

    @abstractmethod
    def generate(
        self,
        noise: Tensor,
        conditioning: dict[str, Any],
        steps: int,
        **kwargs: Any,
    ) -> Tensor:
        """Run the generative process.

        Args:
            noise:        Initial noise tensor ``[B, io_channels, L]``.
            conditioning: Dict of pre-encoded conditioning tensors (e.g.
                          ``{"cross_attn_cond": ..., "global_cond": ...}``).
            steps:        Number of denoising steps.
            **kwargs:     Sampler-specific options (cfg_scale, sigma_min, …).

        Returns:
            Denoised latent tensor of the same shape as ``noise``.
        """


# ---------------------------------------------------------------------------
# BaseConditioner
# ---------------------------------------------------------------------------

class BaseConditioner(nn.Module, ABC):
    """Interface for any conditioning encoder.

    A conditioner maps raw inputs (strings, audio clips, numbers, …) to a
    fixed-width embedding tensor plus an attention mask.  The output contract
    matches the existing ``Conditioner.forward`` contract so adapters are
    one-liners.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by ``encode``."""

    @abstractmethod
    def encode(self, inputs: list[Any]) -> tuple[Tensor, Tensor]:
        """Encode a batch of raw inputs to embeddings.

        Args:
            inputs: List of B raw inputs (strings, tensors, ints, …).

        Returns:
            A 2-tuple ``(embeddings, mask)`` where:
                embeddings : ``[B, seq_len, output_dim]``
                mask       : ``[B, seq_len]`` boolean or float attention mask
        """


# ---------------------------------------------------------------------------
# BasePipeline
# ---------------------------------------------------------------------------

class BasePipeline(ABC):
    """Orchestrator that wires a compressor, generator, and conditioner(s)
    into a complete end-to-end generation flow.

    ``BasePipeline`` is *not* an ``nn.Module``; it is a façade that holds
    modules internally and exposes a single high-level ``generate`` method.
    Device management is handled explicitly via ``to()``.

    Subclasses own the details of:
        * how conditioning inputs are pre-processed
        * noise initialisation
        * sampler dispatch
        * latent → audio decoding
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate produced by this pipeline (Hz)."""

    @property
    @abstractmethod
    def sample_size(self) -> int:
        """Default number of audio samples generated per call."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        conditioning: list[dict[str, Any]],
        steps: int = 100,
        seed: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Generate a batch of audio waveforms.

        Args:
            conditioning: List of B condition dicts, each containing keys
                          appropriate for this pipeline (e.g. ``"prompt"``,
                          ``"seconds_start"``, ``"seconds_total"``).
            steps:        Number of diffusion/flow steps.
            seed:         RNG seed; -1 means random.
            **kwargs:     Pipeline-specific options forwarded to the sampler
                          (cfg_scale, sampler_type, sigma_min, sigma_max, …).

        Returns:
            Float audio tensor of shape ``[B, channels, T]`` in ``[-1, 1]``.
        """

    @abstractmethod
    def to(self, device: str | torch.device) -> "BasePipeline":
        """Move all internal modules to *device* and return ``self``."""

    def unload(self) -> None:
        """Release heavyweight resources before removing a pipeline from cache.

        Concrete pipelines can override this to drop model references, clear
        accelerator caches, or close external handles.  The default is a no-op
        so older/simple pipelines remain compatible with cache eviction.
        """
        return None

    # ------------------------------------------------------------------
    # Concrete output method — inherited by all subclasses automatically
    # ------------------------------------------------------------------

    def generate_and_save(
        self,
        conditioning: list[dict[str, Any]],
        output_manager: "Any",
        steps: int | None = None,
        seed: int = -1,
        model_name: str = "unknown",
        audio_format: str = "wav",
        **kwargs: Any,
    ) -> list["Any"]:
        """Generate audio and save every item to disk via *output_manager*.

        This is a concrete method; all ``BasePipeline`` subclasses inherit it
        for free.  It is the preferred entry point for programmatic use and
        is callable without Gradio or the CLI::

            from anvil_audio.core import load_pipeline
            from anvil_audio.core.output import OutputManager

            pipe = load_pipeline("stable-audio-open-1.0")
            manager = OutputManager(project="my-project")
            results = pipe.generate_and_save(
                [{"prompt": "rain", "seconds_start": 0, "seconds_total": 10}],
                output_manager=manager,
                seed=42,
            )
            print(results[0].path)       # Path to saved WAV
            print(results[0].metadata)   # Full GenerationMetadata

        Args:
            conditioning:  List of B condition dicts passed to ``generate()``.
            output_manager: An ``OutputManager`` instance that handles routing
                            and naming.  Typed as ``Any`` to avoid a hard
                            import at module level.
            steps:         Diffusion steps; falls back to pipeline default if
                           ``None``.
            seed:          RNG seed.  If ``-1``, a random seed is drawn *here*
                           so the exact value is captured in metadata.
            model_name:    Registry name stored in each ``GenerationMetadata``
                           for reproducibility.
            audio_format:  File extension (``"wav"``, ``"flac"``, ``"mp3"``).
            **kwargs:      Forwarded to ``generate()`` (cfg_scale, sampler_type,
                           sigma_min, sigma_max, …).

        Returns:
            List of ``GenerationResult`` objects, one per conditioning item.
            For batches > 1, a ``batch_manifest.json`` is also written.
        """
        import time
        import numpy as np
        from datetime import datetime
        from .output import GenerationMetadata, GenerationResult

        # Resolve seed before generation so metadata always has a real value.
        effective_seed = (
            seed if seed != -1
            else int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
        )

        # Determine effective steps for metadata (pipeline default if not given).
        default_params: dict[str, Any] = getattr(self, "default_params", {})
        effective_steps = steps if steps is not None else default_params.get("steps", 100)

        # Generate all items as a batch tensor: [B, C, T]
        _t0 = time.perf_counter()
        audio_batch = self.generate(
            conditioning,
            steps=steps,
            seed=effective_seed,
            **kwargs,
        )
        generation_duration = round(time.perf_counter() - _t0, 3)

        batch_size = audio_batch.shape[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        subfolder = (
            output_manager.make_batch_dir(batch_size) if batch_size > 1 else None
        )

        results: list[GenerationResult] = []
        for i, cond in enumerate(conditioning):
            audio_i = audio_batch[i]  # [C, T]

            meta = GenerationMetadata(
                prompt=cond.get("prompt", ""),
                model_name=model_name,
                seed=effective_seed,
                steps=effective_steps,
                cfg_scale=float(kwargs.get("cfg_scale", default_params.get("cfg_scale", 7.0))),
                sampler_type=str(kwargs.get("sampler_type", default_params.get("sampler_type", "dpmpp-3m-sde"))),
                sigma_min=float(kwargs.get("sigma_min", default_params.get("sigma_min", 0.3))),
                sigma_max=float(kwargs.get("sigma_max", default_params.get("sigma_max", 500.0))),
                duration_seconds=audio_i.shape[-1] / self.sample_rate,
                timestamp=ts,
                negative_prompt=cond.get("negative_prompt", ""),
                seconds_start=float(cond.get("seconds_start", 0.0)),
                seconds_total=float(cond.get("seconds_total", 0.0)),
                generation_duration_seconds=generation_duration,
            )

            path, sidecar = output_manager.save_audio(
                audio_i, meta, self.sample_rate, audio_format, subfolder
            )
            results.append(
                GenerationResult(audio=audio_i, path=path, sidecar_path=sidecar, metadata=meta)
            )

        if batch_size > 1 and subfolder is not None:
            output_manager.write_batch_manifest(subfolder, results)

        return results

    # ------------------------------------------------------------------
    # Optional component accessors
    # ------------------------------------------------------------------

    def get_compressor(self) -> "BaseCompressor | None":
        """Return the pipeline's compressor component, if any."""
        return None

    def get_generator(self) -> "BaseGenerator | None":
        """Return the pipeline's generator component, if any."""
        return None
