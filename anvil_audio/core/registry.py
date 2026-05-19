"""
Model registry for anvil_audio.

Provides a central catalogue of named pipeline configurations.  Each entry
maps a short name (e.g. ``"stable-audio-open-1.0"``) to everything needed to
instantiate a ready-to-run ``BasePipeline``.

Built-in entries cover known Stability AI HuggingFace models.  Users can
extend or override the registry via a YAML file at::

    ~/.anvil-audio/registry.yaml

YAML schema (all fields optional except ``name``)::

    - name: my-sfx-model
      pretrained_name: myorg/my-sfx-model      # HuggingFace Hub repo
      model_config_path: /path/to/config.json  # local config (alternative)
      ckpt_path: /path/to/model.ckpt           # local checkpoint
      default_params:
        steps: 100
        cfg_scale: 7.0
        sampler_type: dpmpp-3m-sde
        sigma_min: 0.3
        sigma_max: 500

Usage::

    from anvil_audio.core import registry, load_pipeline

    pipeline = load_pipeline("stable-audio-open-1.0")
    audio = pipeline.generate([{"prompt": "rain on a tin roof", "seconds_start": 0, "seconds_total": 10}])
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .interfaces import BasePipeline


# ---------------------------------------------------------------------------
# Default generation parameters
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[str, Any] = {
    "steps": 100,
    "cfg_scale": 7.0,
    "sampler_type": "dpmpp-3m-sde",
    "sigma_min": 0.3,
    "sigma_max": 500.0,
}


def _default_acestep_project_root() -> str:
    """Return Anvil's cache-backed ACE-Step project root."""
    return str(Path.home() / ".cache" / "anvil-audio" / "acestep")

# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegistryEntry:
    """A single named model configuration.

    For Stable Audio / diffusion models at least one of ``pretrained_name``
    or (``model_config_path`` + ``ckpt_path``) must be provided.

    For ACE-Step models set ``pipeline_type = "acestep"`` and supply
    ``acestep_project_root`` (the path to the ACE-Step repo / install
    directory that contains ``checkpoints/``).

    Attributes:
        name:               Short identifier used as the ``--model`` CLI flag.
        pretrained_name:    HuggingFace Hub repository ID (diffusion models).
        model_config_path:  Path to a local JSON model config file, or for
                            ACE-Step the model variant name
                            (e.g. ``"acestep-v15-turbo"``).
        ckpt_path:          Path to a local model checkpoint (.ckpt / .safetensors).
        pretransform_ckpt_path: Optional separate pretransform checkpoint path.
        default_params:     Generation parameters merged over global defaults.
                            Recognised keys: steps, cfg_scale, sampler_type,
                            sigma_min, sigma_max.
        pipeline_type:      ``"diffusion"`` (default), ``"acestep"``, or
                            ``"mlx_diffusion"`` (Apple Silicon only).
        acestep_project_root: Path to the ACE-Step repository root (required
                            when ``pipeline_type == "acestep"``).
        mlx_weights_dir:    Path to a local directory that already contains
                            converted MLX ``.safetensors`` files
                            (``config.json``, ``vae.safetensors``, etc.).
                            ``None`` (the default) triggers auto-convert on
                            first load: weights are downloaded from
                            ``pretrained_name`` and cached at
                            ``~/.cache/anvil-audio/mlx-weights/<model-slug>/``.
                            Only used when ``pipeline_type == "mlx_diffusion"``.
        lm_model_path:      Path (relative to ``checkpoints/``) or absolute
                            path to the ACE-Step 5 Hz LM
                            checkpoint directory.  Typical values:
                            ``"acestep-5Hz-lm-1.7B"`` (lighter/faster) or
                            ``"acestep-5Hz-lm-4B"`` (higher quality).  ``None``
                            disables LM thinking/code hints for this entry.
                            Only used when ``pipeline_type == "acestep"``.
                            Can also be set globally via the
                            ``ACESTEP_LM_MODEL_PATH`` environment variable.
        max_duration:       Maximum allowed generation duration in seconds.
                            For diffusion models this is ``sample_size /
                            sample_rate`` from the model config; set it
                            explicitly here to override or to declare a limit
                            for custom models that Anvil cannot infer
                            automatically.  ``None`` means no explicit limit
                            (the UI and CLI fall back to the model config or a
                            safe default).
    """

    name: str
    pretrained_name: str | None = None
    model_config_path: str | None = None
    ckpt_path: str | None = None
    pretransform_ckpt_path: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)
    pipeline_type: str = "diffusion"
    acestep_project_root: str | None = None
    mlx_weights_dir: str | None = None
    lm_model_path: str | None = None
    max_duration: float | None = None

    def __post_init__(self) -> None:
        if self.pipeline_type == "acestep":
            # Built-in ACE-Step models use Anvil's cache root.  Local checkouts
            # can still be selected explicitly with acestep_project_root.
            if self.acestep_project_root is None:
                object.__setattr__(
                    self,
                    "acestep_project_root",
                    _default_acestep_project_root(),
                )

    def resolved_params(self) -> dict[str, Any]:
        """Return generation params merged over the global defaults."""
        if self.pipeline_type == "acestep":
            _acestep_defaults: dict[str, Any] = {
                "steps": 8,
                "cfg_scale": 7.0,
                "sampler_type": "ode",
                "sigma_min": 0.0,
                "sigma_max": 0.0,
            }
            return {**_acestep_defaults, **self.default_params}
        if self.pipeline_type == "mlx_diffusion":
            _mlx_defaults: dict[str, Any] = {
                "steps": 100,
                "cfg_scale": 7.0,
                "sampler_type": "euler",
                "sigma_min": 0.0,
                "sigma_max": 1.0,
            }
            return {**_mlx_defaults, **self.default_params}
        return {**_DEFAULT_PARAMS, **self.default_params}

    def validate(self) -> None:
        """Raise ``ValueError`` if the entry is not usable."""
        if self.pipeline_type == "acestep":
            # project_root may be None here; ACEStepPipeline will raise a clear
            # error at load time if it is still unresolved then.
            return
        if self.pipeline_type == "mlx_diffusion":
            if self.pretrained_name is None:
                raise ValueError(
                    f"MLX registry entry '{self.name}' must have 'pretrained_name' "
                    "set to the HuggingFace repo ID (used for tokenizer download)."
                )
            return
        has_hf = self.pretrained_name is not None
        has_local = self.model_config_path is not None and self.ckpt_path is not None
        if not has_hf and not has_local:
            raise ValueError(
                f"Registry entry '{self.name}' must have either 'pretrained_name' "
                "or both 'model_config_path' and 'ckpt_path'."
            )


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central catalogue of named pipeline configurations.

    Thread-safety: registration is not thread-safe; load *before* spawning
    workers.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._load_builtin()
        self._load_user_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, entry: RegistryEntry) -> None:
        """Add or replace an entry in the registry.

        Args:
            entry: A ``RegistryEntry`` instance.  Existing entries with the
                   same name are silently replaced.
        """
        entry.validate()
        self._entries[entry.name] = entry

    def get(self, name: str) -> RegistryEntry:
        """Return the entry for *name*, raising ``KeyError`` if not found.

        Args:
            name: The model name as registered (case-sensitive).

        Raises:
            KeyError: If no entry with that name exists.
        """
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(
                f"Model '{name}' not found in registry.  "
                f"Available: {available}"
            )
        return self._entries[name]

    def get_model(self, name: str) -> "RegistryEntry | None":
        """Return the entry for *name*, or ``None`` if not found."""
        return self._entries.get(name)

    def list_models(self) -> list[RegistryEntry]:
        """Return all registered entries sorted by name."""
        return sorted(self._entries.values(), key=lambda e: e.name)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_builtin(self) -> None:
        """Register known Stability AI / HuggingFace models and ACE-Step."""
        builtin: list[RegistryEntry] = [
            RegistryEntry(
                name="stable-audio-open-1.0",
                pretrained_name="stabilityai/stable-audio-open-1.0",
                max_duration=47.0,
                default_params={
                    "steps": 100,
                    "cfg_scale": 7.0,
                    "sampler_type": "dpmpp-3m-sde",
                    "sigma_min": 0.3,
                    "sigma_max": 500.0,
                },
            ),
            RegistryEntry(
                name="stable-audio-open-small",
                pretrained_name="stabilityai/stable-audio-open-small",
                max_duration=11.0,
                default_params={
                    "steps": 25,
                    "cfg_scale": 1.0,
                    "sampler_type": "dpmpp-3m-sde",
                    "sigma_min": 0.3,
                    "sigma_max": 1.0,
                },
            ),
        ]

        # ACE-Step v1.5 built-in entries.
        # project_root is resolved in __post_init__ to Anvil's cache root when
        # acestep_project_root is None.  Entries are always registered; a clear
        # error is raised at pipeline load time if dependencies are missing.
        builtin += [
            RegistryEntry(
                name="acestep-v1.5-turbo",
                pipeline_type="acestep",
                acestep_project_root=None,
                model_config_path="acestep-v15-turbo",
                lm_model_path="acestep-5Hz-lm-1.7B",
                max_duration=600.0,
                default_params={
                    "steps": 8,
                    "cfg_scale": 1.0,
                    "audio_duration": 60,
                    "sampler_type": "ode",
                    "shift": 3.0,
                    "use_adg": False,
                    "cfg_interval_start": 0.0,
                    "cfg_interval_end": 1.0,
                    "lm_cfg_scale": 2.0,
                    "thinking": True,
                    "use_cot_metas": True,
                    "use_cot_caption": False,
                    "use_cot_language": True,
                    "dcw_enabled": False,
                    "velocity_norm_threshold": 0.0,
                    "velocity_ema_factor": 0.0,
                    "sigma_min": 0.0,
                    "sigma_max": 0.0,
                },
            ),
            RegistryEntry(
                name="acestep-v1.5-sft",
                pipeline_type="acestep",
                acestep_project_root=None,
                model_config_path="acestep-v15-sft",
                lm_model_path="acestep-5Hz-lm-4B",
                max_duration=600.0,
                default_params={
                    "steps": 50,
                    "cfg_scale": 7.5,
                    "audio_duration": 60,
                    "sampler_type": "ode",
                    "shift": 3.0,
                    "use_adg": False,
                    "cfg_interval_start": 0.0,
                    "cfg_interval_end": 1.0,
                    "lm_cfg_scale": 2.0,
                    "thinking": False,
                    "use_cot_metas": False,
                    "use_cot_caption": False,
                    "use_cot_language": False,
                    "dcw_enabled": False,
                    "velocity_norm_threshold": 0.0,
                    "velocity_ema_factor": 0.0,
                    "sigma_min": 0.0,
                    "sigma_max": 0.0,
                },
            ),
        ]

        # MLX-accelerated Stable Audio entries (Apple Silicon only).
        # Added when running on macOS arm64 with mlx-audiogen installed.
        # mlx-audiogen ships pre-converted weights on HuggingFace so no local
        # weight conversion is required; they are auto-downloaded on first use.
        _mlx_available = (
            sys.platform == "darwin"
            and importlib.util.find_spec("mlx_audiogen") is not None
        )
        if _mlx_available:
            builtin += [
                RegistryEntry(
                    name="stable-audio-open-small-mlx",
                    pipeline_type="mlx_diffusion",
                    pretrained_name="stabilityai/stable-audio-open-small",
                    mlx_weights_dir=None,   # auto-converts to ~/.cache/anvil-audio/mlx-weights/
                    max_duration=11.0,
                    default_params={
                        "steps": 25,
                        "cfg_scale": 1.0,
                        "sampler_type": "euler",
                        "sigma_min": 0.0,
                        "sigma_max": 1.0,
                    },
                ),
                RegistryEntry(
                    name="stable-audio-open-1.0-mlx",
                    pipeline_type="mlx_diffusion",
                    pretrained_name="stabilityai/stable-audio-open-1.0",
                    mlx_weights_dir=None,   # auto-converts to ~/.cache/anvil-audio/mlx-weights/
                    max_duration=47.0,
                    default_params={
                        "steps": 100,
                        "cfg_scale": 7.0,
                        "sampler_type": "euler",
                        "sigma_min": 0.0,
                        "sigma_max": 1.0,
                    },
                ),
            ]

        for entry in builtin:
            self._entries[entry.name] = entry

    def _load_user_registry(self) -> None:
        """Merge entries from ``~/.anvil-audio/registry.yaml`` if it exists.

        User entries override built-in entries with the same name.
        Invalid entries are skipped with a printed warning.
        """
        registry_path = Path.home() / ".anvil-audio" / "registry.yaml"
        if not registry_path.exists():
            return

        try:
            import yaml  # optional at import time; required only when file exists
        except ImportError:
            print(
                f"[registry] WARNING: {registry_path} found but PyYAML is not "
                "installed.  Run `pip install pyyaml` to enable user registry."
            )
            return

        try:
            with registry_path.open() as fh:
                raw = yaml.safe_load(fh)
        except Exception as exc:
            print(f"[registry] WARNING: Could not read {registry_path}: {exc}")
            return

        if not isinstance(raw, list):
            print(
                f"[registry] WARNING: {registry_path} must contain a YAML list "
                "of model entries.  Skipping."
            )
            return

        for item in raw:
            if not isinstance(item, dict) or "name" not in item:
                print(f"[registry] WARNING: Skipping invalid entry: {item!r}")
                continue
            try:
                raw_max = item.get("max_duration")
                entry = RegistryEntry(
                    name=item["name"],
                    pretrained_name=item.get("pretrained_name"),
                    model_config_path=item.get("model_config_path"),
                    ckpt_path=item.get("ckpt_path"),
                    pretransform_ckpt_path=item.get("pretransform_ckpt_path"),
                    default_params=item.get("default_params") or {},
                    pipeline_type=item.get("pipeline_type", "diffusion"),
                    acestep_project_root=item.get("acestep_project_root"),
                    mlx_weights_dir=item.get("mlx_weights_dir"),
                    lm_model_path=item.get("lm_model_path"),
                    max_duration=float(raw_max) if raw_max is not None else None,
                )
                entry.validate()
                self._entries[entry.name] = entry
            except (ValueError, TypeError) as exc:
                print(f"[registry] WARNING: Skipping entry '{item.get('name')}': {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton and convenience helpers
# ---------------------------------------------------------------------------

#: Global registry instance — import this directly::
#:
#:     from anvil_audio.core import registry
#:     registry.register(RegistryEntry(name="my-model", ...))
registry = ModelRegistry()


def load_pipeline(
    name: str,
    device: str | None = None,
    use_mlx_dit: bool | None = None,
) -> "BasePipeline":
    """Load a pipeline by registry name.

    Dispatches to the correct pipeline class based on the registry entry's
    ``pipeline_type`` field.  Diffusion models return a ``DiffusionPipeline``;
    ACE-Step models return an ``ACEStepPipeline``.

    Args:
        name:   Registry name (e.g. ``"stable-audio-open-1.0"`` or
                ``"acestep-v1.5-turbo"``).
        device: Target device string (``"cuda"``, ``"mps"``, ``"cpu"``).
                Auto-detects via ``get_best_device()`` if omitted.
        use_mlx_dit: ACE-Step-only override for native MLX DiT/VAE use.
                ``None`` keeps registry/env defaults.

    Returns:
        A ready-to-use ``BasePipeline`` instance.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    from anvil_audio.utils.torch_common import get_best_device

    entry = registry.get(name)
    _device = device or str(get_best_device())

    # ---------------------------------------------------------------
    # MLX diffusion pipeline (Apple Silicon + mlx-audiogen)
    # ---------------------------------------------------------------
    if entry.pipeline_type == "mlx_diffusion":
        from anvil_audio.pipelines.mlx_diffusion import MLXDiffusionPipeline

        return MLXDiffusionPipeline(
            repo_id=entry.pretrained_name,  # type: ignore[arg-type]
            weights_dir=entry.mlx_weights_dir,
            default_params=entry.resolved_params(),
        )

    # ---------------------------------------------------------------
    # ACE-Step pipeline
    # ---------------------------------------------------------------
    if entry.pipeline_type == "acestep":
        from anvil_audio.pipelines.acestep import ACEStepPipeline

        # For ACE-Step, map the generic device string to what AceStepHandler
        # understands ("auto", "cuda", "mps", "cpu").
        acestep_device = _device if _device in {"cuda", "mps", "cpu"} else "auto"
        # Resolve LM model path: registry entry > ACESTEP_LM_MODEL_PATH env var.
        lm_path = entry.lm_model_path or os.environ.get("ACESTEP_LM_MODEL_PATH")
        return ACEStepPipeline(
            project_root=entry.acestep_project_root,  # type: ignore[arg-type]
            config_path=entry.model_config_path or "acestep-v15-turbo",
            device=acestep_device,
            lm_model_path=lm_path,
            default_params=entry.resolved_params(),
            use_mlx_dit=use_mlx_dit,
        )

    # ---------------------------------------------------------------
    # Diffusion pipeline (default)
    # ---------------------------------------------------------------
    from .pipeline import DiffusionPipeline

    if entry.pretrained_name is not None:
        from anvil_audio.models.pretrained import get_pretrained_model
        model, model_config = get_pretrained_model(entry.pretrained_name)
    else:
        import json
        from anvil_audio.models.factory import create_model_from_config
        from anvil_audio.models.utils import load_ckpt_state_dict
        from anvil_audio.utils.torch_common import copy_state_dict
        with open(entry.model_config_path) as fh:  # type: ignore[arg-type]
            model_config = json.load(fh)
        model = create_model_from_config(model_config)
        copy_state_dict(model, load_ckpt_state_dict(entry.ckpt_path))

    pipeline = DiffusionPipeline(
        model=model,
        model_config=model_config,
        default_params=entry.resolved_params(),
    )
    pipeline.to(_device)
    return pipeline
