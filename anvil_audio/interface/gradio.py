"""
Gradio web interface for anvil_audio.

Generation is wired through ``DiffusionPipeline`` so that every call
participates in the output manager (filenames, JSON sidecars, batch
manifests).  The interface remains fully functional when launched from the
CLI via ``run_gradio.py``.

Module-level state
------------------
``_pipeline``            : the loaded ``DiffusionPipeline`` (set by ``load_model``).
``_model_name``          : registry name or ``"custom"`` (used in metadata).
``_default_project``     : project name set at launch via ``--project`` CLI arg.
``_pipeline_type``       : ``"diffusion"`` or ``"acestep"`` — updated on load.
                           MLX Stable Audio models set this to ``"diffusion"``
                           since they expose the same UI surface as PyTorch ones.
``_last_generated_path`` : absolute path of the most recently saved audio file;
                           used by the Edit tab's "Load Last Generation" button.
``sample_rate``          : updated whenever a new model is loaded.
``sample_size``          : updated whenever a new model is loaded.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torchaudio import transforms as T

from ..core.output import GenerationMetadata, OutputManager
from ..core.pipeline import DiffusionPipeline
from ..core.registry import registry
from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..training.viz import audio_spectrogram_image
from ..utils.audio_utils import float_to_int16_audio
from ..utils.memory import (
    cleanup_if_memory_pressure,
    estimate_values_size_mb,
    flush_memory_caches,
)
from ..utils.torch_common import copy_state_dict, exists, get_best_device

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_pipeline: DiffusionPipeline | None = None
_model_name: str = ""
_default_project: str = ""
_pipeline_type: str = (
    "diffusion"  # "diffusion" | "acestep"  (mlx_diffusion → "diffusion")
)
_last_generated_path: str = ""  # path of the most recently generated file
sample_rate: int = 32000
sample_size: int = 1920000

_GITHUB_REPO_URL = "https://github.com/PiMPStudios/anvil-audio-v2"
_THEME_DEFAULT_VALUE = "anvil-default"
_THEME_CSS_ATTR = "_anvil_custom_theme_css"
_THEME_BUILTIN_CLASSES = {
    "ocean": "Ocean",
    "citrus": "Citrus",
    "glass": "Glass",
    "monochrome": "Monochrome",
}
_THEME_HUB_REPOS = {
    "terminal": "hmb/terminal",
    "shiki": "Respair/Shiki",
    "minecraft": "YTheme/Minecraft",
    "sketch": "gstaff/sketch",
}
_MEMORY_ENV_PREFIX = "ANVIL_GRADIO_MEMORY"
_LARGE_OUTPUT_CLEANUP_MB_ENV = "ANVIL_GRADIO_LARGE_OUTPUT_CLEANUP_MB"
_DEFAULT_LARGE_OUTPUT_CLEANUP_MB = 128.0
_THEME_PRESETS: tuple[tuple[str, str, str], ...] = (
    (_THEME_DEFAULT_VALUE, "Anvil Default", "Use Gradio's standard theme controls."),
    ("ocean", "Ocean", "Emerald and blue, clean and spacious."),
    ("citrus", "Citrus", "Warm amber accents with high contrast."),
    ("glass", "Glass", "Cool translucent blue/gray surface treatment."),
    ("monochrome", "Monochrome", "Neutral, minimal, studio-console feel."),
    ("terminal", "Terminal", "CRT-inspired terminal styling from hmb/terminal."),
    ("shiki", "Shiki", "Warm Japanese paper-and-ink palette from Respair/Shiki."),
    ("minecraft", "Minecraft", "Blocky pixel-game styling from YTheme/Minecraft."),
    ("sketch", "Sketch", "Hand-drawn notebook styling from gstaff/sketch."),
)
_THEME_DESCRIPTION_JSON = json.dumps(
    {theme_id: description for theme_id, _label, description in _THEME_PRESETS},
    sort_keys=True,
)

_THEME_APPLY_JS = """
(theme) => {
  const descriptions = __THEME_DESCRIPTIONS__;
  const selected = theme || "anvil-default";
  if (selected === "anvil-default") {
    document.documentElement.removeAttribute("data-anvil-theme");
  } else {
    document.documentElement.setAttribute("data-anvil-theme", selected);
  }
  window.localStorage.setItem("anvil_audio_theme", selected);
  return [selected, descriptions[selected] || descriptions["anvil-default"]];
}
""".replace("__THEME_DESCRIPTIONS__", _THEME_DESCRIPTION_JSON)

_THEME_LOAD_JS = """
() => {
  const descriptions = __THEME_DESCRIPTIONS__;
  const selected = window.localStorage.getItem("anvil_audio_theme") || "anvil-default";
  if (selected === "anvil-default") {
    document.documentElement.removeAttribute("data-anvil-theme");
  } else {
    document.documentElement.setAttribute("data-anvil-theme", selected);
  }
  return [selected, descriptions[selected] || descriptions["anvil-default"]];
}
""".replace("__THEME_DESCRIPTIONS__", _THEME_DESCRIPTION_JSON)


def _get_pipeline() -> DiffusionPipeline:
    if _pipeline is None:
        raise RuntimeError("No model loaded.  Call load_model() first.")
    return _pipeline


def _large_output_cleanup_threshold_mb() -> float:
    raw = os.environ.get(_LARGE_OUTPUT_CLEANUP_MB_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_LARGE_OUTPUT_CLEANUP_MB
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_LARGE_OUTPUT_CLEANUP_MB


def _cleanup_before_generation() -> None:
    flush_memory_caches()


def _cleanup_after_generation(output_mb: float = 0.0) -> None:
    threshold_mb = _large_output_cleanup_threshold_mb()
    if threshold_mb > 0 and output_mb >= threshold_mb:
        flush_memory_caches()
        return
    cleanup_if_memory_pressure(
        reason="gradio.post_generation",
        env_prefix=_MEMORY_ENV_PREFIX,
    )


def _theme_cache_root() -> Path:
    return Path.home() / ".cache" / "anvil-audio" / "gradio-themes"


def _scope_theme_css(theme_id: str, css: str) -> str:
    css = css.replace(":root .dark", f'html[data-anvil-theme="{theme_id}"] .dark')
    return css.replace(":root", f'html[data-anvil-theme="{theme_id}"]')


def _load_hub_theme_css(repo_name: str) -> str:
    import gradio as gr

    return gr.themes.ThemeClass.from_hub(repo_name)._get_theme_css()


def _get_hub_theme_css(theme_id: str, repo_name: str) -> str:
    cache_path = _theme_cache_root() / f"{theme_id}.css"
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except OSError:
            pass

    css = _load_hub_theme_css(repo_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(css, encoding="utf-8")
    return css


def _build_custom_theme_css() -> str:
    """Return CSS for bundled runtime theme presets."""
    import gradio as gr

    chunks = [
        """
html .anvil-github-star {
  align-items: center;
  color: var(--body-text-color-subdued);
  display: flex;
  flex-wrap: wrap;
  font-size: var(--text-sm);
  gap: 8px;
  margin-top: 8px;
}

html .anvil-github-star a {
  align-items: center;
  background: var(--button-secondary-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 6px;
  color: var(--body-text-color);
  display: inline-flex;
  font-weight: 600;
  justify-content: center;
  line-height: 1;
  min-height: 32px;
  padding: 0 12px;
  text-decoration: none;
}

html .anvil-github-star a:hover {
  background: var(--button-secondary-background-fill-hover);
  border-color: var(--border-color-accent);
  color: var(--body-text-color);
}

html .anvil-github-star span {
  line-height: 1.35;
}

html[data-anvil-theme] .anvil-theme-picker {
  border-color: var(--border-color-accent);
}
"""
    ]
    for theme_id, theme_cls_name in _THEME_BUILTIN_CLASSES.items():
        theme_cls = getattr(gr.themes, theme_cls_name)
        css = theme_cls()._get_theme_css()
        chunks.append(_scope_theme_css(theme_id, css))
    for theme_id, repo_name in _THEME_HUB_REPOS.items():
        try:
            css = _get_hub_theme_css(theme_id, repo_name)
        except Exception as exc:
            warnings.warn(
                f"Could not load Gradio theme {repo_name}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        chunks.append(_scope_theme_css(theme_id, css))
    return "\n".join(chunks)


def _theme_dropdown_choices() -> list[tuple[str, str]]:
    return [(label, value) for value, label, _description in _THEME_PRESETS]


def _lora_dropdown_choices() -> list[tuple[str, str]]:
    """Return loadable LoRA adapters for the Gradio picker."""
    from anvil_audio.lora import list_adapters

    choices = [("No adapter", "")]
    for entry in list_adapters():
        if not entry.loadable:
            continue
        label = entry.name if entry.name == entry.id else f"{entry.name} ({entry.id})"
        choices.append((label, entry.id))
    return choices


def _refresh_lora_dropdown(current_value: str | None = None) -> Any:
    """Refresh the LoRA picker without dropping a typed custom path."""
    import gradio as gr

    choices = _lora_dropdown_choices()
    values = {value for _label, value in choices}
    value = (current_value or "").strip()
    if value and value not in values:
        return gr.update(choices=choices, value=value)
    return gr.update(choices=choices, value=value if value in values else "")


def _theme_markdown(value: str) -> str:
    descriptions = {
        theme_id: description for theme_id, _label, description in _THEME_PRESETS
    }
    return descriptions.get(value, descriptions[_THEME_DEFAULT_VALUE])


def _github_star_html() -> str:
    return f"""
<div class="anvil-github-star">
  <a href="{_GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer"
     aria-label="Open Anvil Audio v2 on GitHub to star the repository">
    Star on GitHub
  </a>
  <span>If Anvil is useful, a star helps other audio builders find it.</span>
</div>
"""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_config: dict[str, Any] | None = None,
    model_ckpt_path: str | None = None,
    pretrained_name: str | None = None,
    pretransform_ckpt_path: str | None = None,
    device: torch.device | str | None = None,
    model_half: bool = False,
    project: str = "",
) -> tuple[DiffusionPipeline, dict[str, Any]]:
    """Load a model and store it in module-level state.

    Accepts the same arguments as the original ``load_model`` so all existing
    callers (``create_ui``, tests) continue to work.

    Args:
        model_config:           Parsed JSON config dict.  Ignored if
                                *pretrained_name* is given.
        model_ckpt_path:        Local checkpoint path.  Used with
                                *model_config*.
        pretrained_name:        HuggingFace Hub repo ID.  Takes priority.
        pretransform_ckpt_path: Optional separate VAE checkpoint.
        device:                 Target device.  Auto-detected if ``None``.
        model_half:             Cast model weights to float16.
        project:                Project name for the output manager.

    Returns:
        ``(pipeline, model_config)`` — ``pipeline`` is also stored in the
        module-level ``_pipeline`` variable.
    """
    global _pipeline, _model_name, _default_project, sample_rate, sample_size

    resolved_device = torch.device(device) if device is not None else get_best_device()

    if pretrained_name is not None:
        print(f"->->-> Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
        _model_name = pretrained_name
    elif model_config is not None and model_ckpt_path is not None:
        print("->->-> Creating model from config")
        from ..models.factory import create_model_from_config

        model = create_model_from_config(model_config)
        print(f"->->-> Loading checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        _model_name = "custom"
    else:
        raise RuntimeError(
            "Provide either 'pretrained_name' or both 'model_config' and 'model_ckpt_path'."
        )

    if pretransform_ckpt_path is not None:
        print(f"->->-> Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(pretransform_ckpt_path), strict=False
        )

    model.to(resolved_device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    model_type = model_config.get("model_type", "diffusion_cond")
    if model_type in {"diffusion_cond", "diffusion_cond_inpaint", "diffusion_prior"}:
        _pipeline = DiffusionPipeline(
            model=model,
            model_config=model_config,
        )
        _pipeline._device = resolved_device
    else:
        # Non-pipeline model types (autoencoder, lm, diffusion_prior, etc.)
        # Store the raw model on a shim so the rest of the code can access it.
        _pipeline = _RawModelShim(model, model_config, resolved_device)  # type: ignore[assignment]

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    _default_project = project
    _pipeline_type = "diffusion"

    print(f"->->-> Model loaded on {resolved_device}")
    return _pipeline, model_config


def load_acestep_model(
    entry: Any,
    device: torch.device | str | None = None,
    project: str = "",
) -> None:
    """Load an ACE-Step pipeline from a registry entry into module-level state.

    Args:
        entry:   A ``RegistryEntry`` with ``pipeline_type == "acestep"``.
        device:  Target device.  ``None`` → auto-detect.
        project: Default project name for output routing.
    """
    global \
        _pipeline, \
        _model_name, \
        _default_project, \
        sample_rate, \
        sample_size, \
        _pipeline_type

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    resolved_device = torch.device(device) if device is not None else get_best_device()
    acestep_device = str(resolved_device.type)
    if acestep_device not in {"cuda", "mps", "cpu"}:
        acestep_device = "auto"

    _pipeline = ACEStepPipeline(  # type: ignore[assignment]
        project_root=entry.acestep_project_root,
        config_path=entry.model_config_path or "acestep-v15-turbo",
        device=acestep_device,
        lm_model_path=entry.lm_model_path,
        default_params=entry.resolved_params(),
    )
    _model_name = entry.name
    sample_rate = _pipeline.sample_rate
    sample_size = _pipeline.sample_size
    _default_project = project
    _pipeline_type = "acestep"


def load_mlx_model(
    entry: Any,
    project: str = "",
) -> None:
    """Load an MLX Stable Audio pipeline from a registry entry into module-level state.

    Args:
        entry:   A ``RegistryEntry`` with ``pipeline_type == "mlx_diffusion"``.
        project: Default project name for output routing.
    """
    global \
        _pipeline, \
        _model_name, \
        _default_project, \
        sample_rate, \
        sample_size, \
        _pipeline_type

    from anvil_audio.pipelines.mlx_diffusion import MLXDiffusionPipeline

    _pipeline = MLXDiffusionPipeline(  # type: ignore[assignment]
        repo_id=entry.pretrained_name,
        weights_dir=entry.mlx_weights_dir,
        default_params=entry.resolved_params(),
    )
    _model_name = entry.name
    sample_rate = _pipeline.sample_rate
    sample_size = _pipeline.sample_size
    _default_project = project
    _pipeline_type = "diffusion"  # MLX Stable Audio behaves like diffusion in the UI


class _RawModelShim:
    """Minimal shim so non-pipeline model types still work through the global."""

    def __init__(
        self, model: Any, config: dict[str, Any], device: torch.device
    ) -> None:
        self._model = model
        self._config = config
        self._device = device

    # Proxy attribute access to the underlying model so existing code that
    # does `model.pretransform` etc. still works.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    @property
    def sample_rate(self) -> int:
        return self._config["sample_rate"]

    @property
    def sample_size(self) -> int:
        return self._config["sample_size"]


def _prepare_init_audio(
    init_audio: tuple[int, Any] | None,
    use_init: bool,
) -> tuple[int, torch.Tensor] | None:
    """Convert Gradio audio input (sr, np.ndarray) to (sr, Tensor)."""
    if not use_init or init_audio is None:
        return None
    in_sr, audio_np = init_audio
    audio = torch.from_numpy(audio_np).float().div(32767)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1)
    if in_sr != sample_rate:
        audio = T.Resample(in_sr, sample_rate)(audio)
    return in_sr, audio


# ---------------------------------------------------------------------------
# Conditional generation
# ---------------------------------------------------------------------------


def generate_cond(
    prompt: str,
    negative_prompt: str | None = None,
    seconds_start: float = 0,
    seconds_total: float = 30,
    cfg_scale: float = 6.0,
    steps: int = 250,
    preview_every: int | None = None,
    seed: int = -1,
    sampler_type: str = "dpmpp-3m-sde",
    sigma_min: float = 0.03,
    sigma_max: float = 1000,
    cfg_rescale: float = 0.0,
    use_init: bool = False,
    init_audio: Any = None,
    init_noise_level: float = 1.0,
    mask_cropfrom: float | None = None,
    mask_pastefrom: float | None = None,
    mask_pasteto: float | None = None,
    mask_maskstart: float | None = None,
    mask_maskend: float | None = None,
    mask_softnessL: float | None = None,
    mask_softnessR: float | None = None,
    mask_marination: float | None = None,
    batch_size: int = 1,
    project: str = "",
    audio_format: str = "wav",
) -> tuple[str, list[Any], dict[str, Any]]:
    """Generate audio from a text prompt.

    Returns:
        ``(wav_path, spectrogram_images, metadata_dict)``
    """
    if not prompt or not prompt.strip():
        import gradio as gr

        gr.Warning("Please enter a prompt before generating.")
        return None, [], None  # type: ignore[return-value]

    pipeline = _get_pipeline()
    _cleanup_before_generation()

    print("=== Conditional generation ===")
    print(f"\tPrompt: {prompt}")
    print(f"\tStart (sec): {seconds_start}  |  Length (sec): {seconds_total}")
    print(f"\tCFG scale: {cfg_scale}  |  Steps: {steps}  |  Seed: {seed}")

    # Resolve seed before generation so it's captured in metadata
    effective_seed = (
        int(seed) if int(seed) != -1 else int(np.random.randint(0, 2**32 - 1))
    )

    conditioning = [
        {
            "prompt": prompt,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
        }
    ] * batch_size
    negative_conditioning = (
        [
            {
                "prompt": negative_prompt,
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
            }
        ]
        * batch_size
        if negative_prompt
        else None
    )

    init_audio_tensor = _prepare_init_audio(init_audio, use_init)

    # Extend sample_size if init_audio is longer than the model default
    input_sample_size = sample_size
    if init_audio_tensor is not None:
        _, audio_t = init_audio_tensor
        audio_length = audio_t.shape[-1]
        if audio_length > sample_size:
            min_len = getattr(pipeline, "min_input_length", 1)
            input_sample_size = (
                audio_length + (min_len - (audio_length % min_len)) % min_len
            )

    mask_args: dict[str, float] | None = None
    if mask_cropfrom is not None:
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }

    # Preview callback (optional step-by-step spectrograms)
    preview_images: list[Any] = []
    if preview_every == 0:
        preview_every = None

    def _preview_callback(callback_info: dict[str, Any]) -> None:
        from ..pipelines.mlx_diffusion import MLXDiffusionPipeline

        if isinstance(pipeline, MLXDiffusionPipeline):
            return
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        if (current_step - 1) % preview_every == 0:  # type: ignore[operator]
            if pipeline._model.pretransform is not None:
                denoised = pipeline._model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            preview_images.append(
                (
                    audio_spectrogram_image(denoised, sample_rate=sample_rate),
                    f"Step {current_step} sigma={callback_info['sigma']:.3f}",
                )
            )

    # Generate via pipeline
    import time as _time
    from ..pipelines.mlx_diffusion import MLXDiffusionPipeline

    _gen_t0 = _time.perf_counter()
    if isinstance(pipeline, MLXDiffusionPipeline):
        mlx_conditioning = []
        for i, cond in enumerate(conditioning):
            merged = dict(cond)
            if negative_conditioning is not None and i < len(negative_conditioning):
                merged["negative_prompt"] = negative_conditioning[i].get("prompt", "")
            mlx_conditioning.append(merged)
        audio = pipeline.generate(
            conditioning=mlx_conditioning,
            steps=steps,
            seed=effective_seed,
            cfg_scale=cfg_scale,
            sampler_type=sampler_type,
            sigma_max=sigma_max,
        )  # [B, C, T]
    else:
        audio = generate_diffusion_cond(
            pipeline._model,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            sample_size=input_sample_size,
            seed=effective_seed,
            device=pipeline._device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio_tensor,
            init_noise_level=init_noise_level,
            mask_args=mask_args,
            callback=_preview_callback if preview_every is not None else None,
            scale_phi=cfg_rescale,
        )  # [B, C, T]
    _gen_duration = round(_time.perf_counter() - _gen_t0, 3)

    # Take first item; clip to seconds_total
    audio_item = audio.squeeze(0)  # [C, T] or [B, C, T] → [C, T] for batch_size=1
    if audio.shape[0] > 1:
        audio_item = audio[0]
    audio_int16 = float_to_int16_audio(audio_item)
    length = int(sample_rate * seconds_total)
    audio_int16 = audio_int16[:, :length]

    # Save via output manager
    output_manager = OutputManager(project=project or _default_project or None)
    meta = GenerationMetadata(
        prompt=prompt,
        model_name=_model_name,
        seed=effective_seed,
        steps=steps,
        cfg_scale=float(cfg_scale),
        sampler_type=sampler_type,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        duration_seconds=audio_int16.shape[-1] / sample_rate,
        timestamp=__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
        negative_prompt=negative_prompt or "",
        seconds_start=float(seconds_start),
        seconds_total=float(seconds_total),
        generation_duration_seconds=_gen_duration,
    )
    path, _ = output_manager.save_audio(
        audio_int16.cpu(), meta, sample_rate, ext=audio_format
    )

    spectrogram = audio_spectrogram_image(audio_int16.cpu(), sample_rate=sample_rate)
    output_mb = estimate_values_size_mb(audio, audio_item, audio_int16, init_audio_tensor)
    del audio
    del audio_item
    del audio_int16
    del init_audio_tensor
    _cleanup_after_generation(output_mb)
    return str(path), [spectrogram, *preview_images], meta.to_dict()


# ---------------------------------------------------------------------------
# Unconditional generation
# ---------------------------------------------------------------------------


def generate_uncond(
    steps: int = 250,
    seed: int = -1,
    sampler_type: str = "dpmpp-3m-sde",
    sigma_min: float = 0.03,
    sigma_max: float = 1000,
    use_init: bool = False,
    init_audio: Any = None,
    init_noise_level: float = 1.0,
    batch_size: int = 1,
    preview_every: int | None = None,
    project: str = "",
) -> tuple[str, list[Any]]:
    pipeline = _get_pipeline()
    _cleanup_before_generation()

    init_audio_tensor = _prepare_init_audio(init_audio, use_init)
    input_sample_size = sample_size
    if init_audio_tensor is not None:
        _, audio_t = init_audio_tensor
        audio_length = audio_t.shape[-1]
        if audio_length > sample_size:
            min_len = getattr(pipeline, "min_input_length", 1)
            input_sample_size = (
                audio_length + (min_len - (audio_length % min_len)) % min_len
            )

    effective_seed = (
        int(seed) if int(seed) != -1 else int(np.random.randint(0, 2**32 - 1))
    )

    preview_images: list[Any] = []

    def _preview_callback(callback_info: dict[str, Any]) -> None:
        if preview_every is None or (callback_info["i"] - 1) % preview_every != 0:
            return
        denoised = callback_info["denoised"]
        if pipeline._model.pretransform is not None:
            denoised = pipeline._model.pretransform.decode(denoised)
        denoised = rearrange(denoised, "b d n -> d (b n)")
        denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        preview_images.append(
            (
                audio_spectrogram_image(denoised, sample_rate=sample_rate),
                f"Step {callback_info['i']} sigma={callback_info['sigma']:.3f}",
            )
        )

    from ..pipelines.mlx_diffusion import MLXDiffusionPipeline

    if isinstance(pipeline, MLXDiffusionPipeline):
        uncond_cond = [
            {"prompt": "", "seconds_total": float(sample_size / sample_rate)}
        ] * batch_size
        audio = pipeline.generate(
            conditioning=uncond_cond, steps=steps, seed=effective_seed
        )
    else:
        audio = generate_diffusion_uncond(
            pipeline._model,
            steps=steps,
            batch_size=batch_size,
            sample_size=input_sample_size,
            seed=effective_seed,
            device=pipeline._device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio_tensor,
            init_noise_level=init_noise_level,
            callback=_preview_callback if preview_every is not None else None,
        )

    audio_out = rearrange(audio, "b d n -> d (b n)")
    audio_int16 = (
        audio_out.to(torch.float32)
        .div(torch.max(torch.abs(audio_out)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )

    output_manager = OutputManager(project=project or _default_project or None)
    meta = GenerationMetadata(
        prompt="(unconditional)",
        model_name=_model_name,
        seed=effective_seed,
        steps=steps,
        cfg_scale=1.0,
        sampler_type=sampler_type,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        duration_seconds=audio_int16.shape[-1] / sample_rate,
        timestamp=__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    path, _ = output_manager.save_audio(audio_out.cpu(), meta, sample_rate)

    spectrogram = audio_spectrogram_image(audio_int16, sample_rate=sample_rate)
    output_mb = estimate_values_size_mb(audio, audio_out, audio_int16, init_audio_tensor)
    del audio
    del audio_out
    del audio_int16
    del init_audio_tensor
    _cleanup_after_generation(output_mb)
    return str(path), [spectrogram, *preview_images]


# ---------------------------------------------------------------------------
# Language model generation
# ---------------------------------------------------------------------------


def generate_lm(
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    batch_size: int = 1,
    project: str = "",
) -> tuple[str, list[Any]]:
    pipeline = _get_pipeline()
    _cleanup_before_generation()

    audio = pipeline._model.generate_audio(
        batch_size=batch_size,
        max_gen_len=sample_size // pipeline._model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True,
    )

    audio_out = rearrange(audio, "b d n -> d (b n)")
    audio_int16 = (
        audio_out.to(torch.float32)
        .div(torch.max(torch.abs(audio_out)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )

    output_manager = OutputManager(project=project or _default_project or None)
    meta = GenerationMetadata(
        prompt="(lm)",
        model_name=_model_name,
        seed=-1,
        steps=0,
        cfg_scale=1.0,
        sampler_type="lm",
        sigma_min=0.0,
        sigma_max=0.0,
        duration_seconds=audio_int16.shape[-1] / sample_rate,
        timestamp=__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    path, _ = output_manager.save_audio(audio_out.cpu(), meta, sample_rate)

    spectrogram = audio_spectrogram_image(audio_int16, sample_rate=sample_rate)
    output_mb = estimate_values_size_mb(audio, audio_out, audio_int16)
    del audio
    del audio_out
    del audio_int16
    _cleanup_after_generation(output_mb)
    return str(path), [spectrogram]


# ---------------------------------------------------------------------------
# ACE-Step generation
# ---------------------------------------------------------------------------


def generate_acestep(
    prompt: str,
    lyrics: str = "",
    negative_prompt: str = "",
    seconds_total: float = 30.0,
    steps: int = 8,
    cfg_scale: float = 7.0,
    seed: int = -1,
    project: str = "",
    audio_format: str = "wav",
    lora_reference: str = "",
    lora_scale: float = 1.0,
    lora_stack: str = "",
) -> tuple[str, list[Any], dict[str, Any]]:
    """Generate music with ACE-Step.

    Args:
        prompt:        Tags / style caption (e.g. ``"upbeat indie pop, energetic"``).
        lyrics:        Lyric text.  Leave blank or pass ``"[Instrumental]"`` for
                       instrumental output.
        negative_prompt: Text describing sounds or qualities to avoid. ACE-Step
                         applies this to the LM/thinking path when enabled.
        seconds_total: Duration of the generated audio in seconds.
        steps:         Number of diffusion steps (8 = turbo, 60 = full quality).
        cfg_scale:     Classifier-free guidance scale.
        seed:          RNG seed; -1 means random.
        project:       Project name for output routing.
        lora_reference: Optional registered adapter id/name or direct path.
        lora_scale:     LoRA strength from 0.0 to 1.0.
        lora_stack:     Additional comma/newline stack entries as adapter[:scale].

    Returns:
        ``(wav_path, [spectrogram_image], metadata_dict)``
    """
    from datetime import datetime

    pipeline = _get_pipeline()
    _cleanup_before_generation()

    print("=== ACE-Step generation ===")
    print(f"\tPrompt (tags): {prompt}")
    print(f"\tLyrics: {lyrics[:60]!r}{'...' if len(lyrics) > 60 else ''}")
    if negative_prompt:
        print(f"\tNegative prompt: {negative_prompt}")
    lora_metadata: dict[str, Any] = {}
    if (lora_reference and lora_reference.strip()) or (lora_stack and lora_stack.strip()):
        from anvil_audio.lora import resolve_lora_stack

        lora_items = resolve_lora_stack(
            lora_reference,
            primary_scale=float(lora_scale),
            stack=lora_stack,
        )
        apply_stack = getattr(pipeline, "apply_lora_stack", None)
        if callable(apply_stack):
            lora_status = apply_stack(lora_items)
        elif len(lora_items) == 1:
            apply_lora = getattr(pipeline, "apply_lora_adapter", None)
            if not callable(apply_lora):
                raise RuntimeError("Current ACE-Step pipeline does not support LoRA loading.")
            item = lora_items[0]
            lora_status = apply_lora(
                item.path,
                adapter_name=item.adapter_name,
                scale=float(item.scale),
            )
        else:
            raise RuntimeError("Current ACE-Step pipeline does not support LoRA stacking.")
        stack_metadata = [item.to_metadata() for item in lora_items]
        lora_metadata = {
            "lora_stack": {
                "adapters": stack_metadata,
                "status": lora_status,
            }
        }
        if len(stack_metadata) == 1:
            lora_metadata["lora"] = stack_metadata[0] | {"status": lora_status}
        print(f"\tLoRA stack: {len(stack_metadata)} adapter(s)")
    print(
        f"\tDuration: {seconds_total}s  |  Steps: {steps}  |  CFG: {cfg_scale}  |  Seed: {seed}"
    )

    effective_seed = (
        int(seed) if int(seed) != -1 else int(np.random.randint(0, 2**32 - 1))
    )

    conditioning = [
        {
            "prompt": prompt,
            "lyrics": lyrics,
            "negative_prompt": negative_prompt or "",
            "seconds_total": seconds_total,
        }
    ]

    import time as _time

    _gen_t0 = _time.perf_counter()
    audio = pipeline.generate(  # type: ignore[union-attr]
        conditioning,
        steps=steps,
        seed=effective_seed,
        cfg_scale=cfg_scale,
    )  # [1, C, T]
    _gen_duration = round(_time.perf_counter() - _gen_t0, 3)

    audio_item = audio[0]  # [C, T]
    audio_int16 = float_to_int16_audio(audio_item)
    length = int(sample_rate * seconds_total)
    audio_int16 = audio_int16[:, :length]

    output_manager = OutputManager(project=project or _default_project or None)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from ..core.output import GenerationMetadata as _GM

    meta = _GM(
        prompt=prompt,
        model_name=_model_name,
        seed=effective_seed,
        steps=steps,
        cfg_scale=float(cfg_scale),
        sampler_type="ode",
        sigma_min=0.0,
        sigma_max=0.0,
        duration_seconds=audio_int16.shape[-1] / sample_rate,
        timestamp=ts,
        negative_prompt=negative_prompt or "",
        seconds_start=0.0,
        seconds_total=float(seconds_total),
        generation_duration_seconds=_gen_duration,
        extra={"lyrics": lyrics} | lora_metadata,
    )
    path, _ = output_manager.save_audio(
        audio_int16.cpu(), meta, sample_rate, ext=audio_format
    )

    spectrogram = audio_spectrogram_image(audio_int16.cpu(), sample_rate=sample_rate)
    output_mb = estimate_values_size_mb(audio, audio_item, audio_int16)
    del audio
    del audio_item
    del audio_int16
    _cleanup_after_generation(output_mb)
    return str(path), [spectrogram], meta.to_dict()


def generate_unified(
    prompt: str,
    lyrics: str,
    negative_prompt: str,
    seconds_start: float,
    seconds_total: float,
    steps: int,
    preview_every: int,
    cfg_scale: float,
    seed: int,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    cfg_rescale: float,
    use_init: bool,
    init_audio: Any,
    init_noise_level: float,
    project: str,
    audio_format: str = "wav",
    lora_reference: str = "",
    lora_scale: float = 1.0,
    lora_stack: str = "",
) -> tuple[str, list[Any], dict[str, Any]]:
    """Route to the correct generation backend based on the currently loaded pipeline type."""
    global _last_generated_path
    if not prompt or not prompt.strip():
        import gradio as gr

        gr.Warning("Please enter a prompt before generating.")
        return None, [], None  # type: ignore[return-value]
    if _pipeline_type == "acestep":
        result = generate_acestep(
            prompt=prompt,
            lyrics=lyrics,
            negative_prompt=negative_prompt or "",
            seconds_total=seconds_total,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            project=project,
            audio_format=audio_format,
            lora_reference=lora_reference,
            lora_scale=lora_scale,
            lora_stack=lora_stack,
        )
    else:
        result = generate_cond(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            seconds_start=seconds_start,
            seconds_total=seconds_total,
            cfg_scale=cfg_scale,
            steps=steps,
            preview_every=preview_every,
            seed=seed,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            cfg_rescale=cfg_rescale,
            use_init=use_init,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            project=project,
            audio_format=audio_format,
        )
    _last_generated_path = result[0]
    return result


# ---------------------------------------------------------------------------
# Autoencoder / diffusion-prior passthrough (unchanged model logic)
# ---------------------------------------------------------------------------


def autoencoder_process(
    audio: Any,
    latent_noise: float,
    n_quantizers: int,
    project: str = "",
) -> str:
    pipeline = _get_pipeline()
    model = pipeline._model
    _cleanup_before_generation()

    device = pipeline._device
    dtype = next(model.parameters()).dtype

    in_sr, audio_np = audio
    audio_t = torch.from_numpy(audio_np).float().div(32767).to(device)
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)
    else:
        audio_t = audio_t.transpose(0, 1)

    audio_t = model.preprocess_audio_for_encoder(audio_t, in_sr).to(dtype)

    kwargs_enc: dict[str, Any] = {"chunked": False}
    kwargs_dec: dict[str, Any] = {"chunked": False}
    if n_quantizers > 0:
        kwargs_enc["n_quantizers"] = n_quantizers

    latents = model.encode_audio(audio_t, **kwargs_enc)
    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise
    audio_out = model.decode_audio(latents, **kwargs_dec)

    audio_out = rearrange(audio_out, "b d n -> d (b n)")
    audio_int16 = (
        audio_out.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    )

    output_manager = OutputManager(project=project or _default_project or None)
    meta = GenerationMetadata(
        prompt="(autoencoder)",
        model_name=_model_name,
        seed=-1,
        steps=0,
        cfg_scale=1.0,
        sampler_type="autoencoder",
        sigma_min=0.0,
        sigma_max=0.0,
        duration_seconds=audio_int16.shape[-1] / sample_rate,
        timestamp=__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    path, _ = output_manager.save_audio(audio_out.cpu(), meta, sample_rate)
    output_mb = estimate_values_size_mb(audio_t, latents, audio_out, audio_int16)
    del audio_t
    del latents
    del audio_out
    del audio_int16
    _cleanup_after_generation(output_mb)
    return str(path)


def diffusion_prior_process(
    audio: Any,
    steps: int,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    project: str = "",
) -> str:
    pipeline = _get_pipeline()
    model = pipeline._model
    _cleanup_before_generation()

    device = pipeline._device
    in_sr, audio_np = audio
    audio_t = torch.from_numpy(audio_np).float().div(32767).to(device)
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)
    elif audio_t.dim() == 2:
        audio_t = audio_t.transpose(0, 1)
    audio_t = audio_t.unsqueeze(0)

    audio_out = model.stereoize(
        audio_t,
        in_sr,
        steps,
        sampler_kwargs={
            "sampler_type": sampler_type,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
        },
    )
    audio_out = rearrange(audio_out, "b d n -> d (b n)")
    audio_int16 = (
        audio_out.to(torch.float32)
        .div(torch.max(torch.abs(audio_out)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )

    output_manager = OutputManager(project=project or _default_project or None)
    meta = GenerationMetadata(
        prompt="(diffusion-prior)",
        model_name=_model_name,
        seed=-1,
        steps=steps,
        cfg_scale=1.0,
        sampler_type=sampler_type,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        duration_seconds=audio_int16.shape[-1] / sample_rate,
        timestamp=__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    path, _ = output_manager.save_audio(audio_out.cpu(), meta, sample_rate)
    output_mb = estimate_values_size_mb(audio_t, audio_out, audio_int16)
    del audio_t
    del audio_out
    del audio_int16
    _cleanup_after_generation(output_mb)
    return str(path)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _model_load_ui(
    model_name: str,
    project: str,
    model_half: bool,
    device_str: str,
) -> str:
    """Called by the 'Load Model' button in the UI."""
    try:
        device = torch.device(device_str) if device_str else get_best_device()
        entry = registry.get_model(model_name)

        # ACE-Step models use a dedicated loading path.
        if (
            entry is not None
            and getattr(entry, "pipeline_type", "diffusion") == "acestep"
        ):
            load_acestep_model(entry=entry, device=device, project=project)
            return f"Loaded ACE-Step: {model_name} on {device}"

        # MLX Stable Audio models (Apple Silicon only).
        if (
            entry is not None
            and getattr(entry, "pipeline_type", "diffusion") == "mlx_diffusion"
        ):
            load_mlx_model(entry=entry, project=project)
            return f"Loaded MLX: {model_name} (Metal GPU)"

        # Diffusion models: resolve registry short-name → HuggingFace pretrained_name.
        # If the name isn't in the registry, treat it as a raw HF repo ID.
        resolved_pretrained = (
            entry.pretrained_name if entry and entry.pretrained_name else model_name
        )

        load_model(
            pretrained_name=resolved_pretrained,
            device=device,
            model_half=model_half,
            project=project,
        )
        return f"Loaded: {model_name} ({resolved_pretrained}) on {device}"
    except Exception as exc:
        return f"Error loading '{model_name}': {exc}"


def _model_load_ui_with_params(
    model_name: str,
    project: str,
    model_half: bool,
    device_str: str,
) -> tuple:
    """Like _model_load_ui but also returns registry default_params to update UI sliders.

    Returns (status, steps, cfg_scale, sampler_type, sigma_min, sigma_max) × 2 tabs = 11 values.
    ACE-Step models use their own defaults and 0.0 for the sigma fields.
    """
    import gradio as gr

    status = _model_load_ui(model_name, project, model_half, device_str)
    entry = registry.get_model(model_name)
    p = entry.resolved_params() if entry else {}

    updates = (
        gr.update(value=p.get("steps", 100)),
        gr.update(value=p.get("cfg_scale", 7.0)),
        gr.update(value=p.get("sampler_type", "dpmpp-3m-sde")),
        gr.update(value=p.get("sigma_min", 0.03)),
        gr.update(value=p.get("sigma_max", 500.0)),
    )
    # Return updates for both Generation and Inpainting tabs
    return (status,) + updates + updates


def _model_load_ui_with_acestep_params(
    model_name: str,
    project: str,
    model_half: bool,
    device_str: str,
) -> tuple:
    """Like _model_load_ui but also returns registry default_params for ACE-Step sliders.

    Returns (status, seconds_total_update, steps_update, cfg_scale_update).
    """
    import gradio as gr

    status = _model_load_ui(model_name, project, model_half, device_str)
    entry = registry.get_model(model_name)
    p = entry.resolved_params() if entry else {}

    return (
        status,
        gr.update(value=int(p.get("audio_duration", 60))),
        gr.update(value=int(p.get("steps", 50))),
        gr.update(value=float(p.get("cfg_scale", 4.0))),
    )


def _model_load_ui_unified(
    model_name: str,
    project: str,
    model_half: bool,
    device_str: str,
) -> tuple:
    import gradio as gr

    status = _model_load_ui(model_name, project, model_half, device_str)
    entry = registry.get_model(model_name)
    is_as = (
        entry is not None and getattr(entry, "pipeline_type", "diffusion") == "acestep"
    )
    p = entry.resolved_params() if entry else {}

    if is_as:
        max_dur = entry.max_duration if entry and entry.max_duration else 600.0
        dur_val = min(int(p.get("audio_duration", 60)), max_dur)
        return (
            status,
            gr.update(visible=True),  # lyrics_row
            gr.update(visible=True),  # neg_prompt_row
            gr.update(visible=True),  # intelligence_lyrics_row
            gr.update(visible=False),  # diffusion_controls
            gr.update(visible=True),  # lora_controls
            gr.update(value=dur_val, maximum=max_dur),
            gr.update(value=int(p.get("steps", 50))),
            gr.update(value=float(p.get("cfg_scale", 4.0))),
            gr.update(value="ode"),
            gr.update(value=0.0),
            gr.update(value=0.0),
            gr.update(visible=False),  # inpaint_content — ACE-Step doesn't support it
            gr.update(visible=True),  # inpaint_unsupported
        )

    # Diffusion model: max duration from registry entry or loaded model config
    if entry and entry.max_duration:
        max_dur = entry.max_duration
    else:
        # sample_size / sample_rate are updated by load_model() called inside _model_load_ui
        max_dur = sample_size / sample_rate if sample_rate else 240.0
    default_dur = min(p.get("seconds_total", max_dur), max_dur)
    return (
        status,
        gr.update(visible=False),  # lyrics_row
        gr.update(visible=True),  # neg_prompt_row
        gr.update(visible=False),  # intelligence_lyrics_row
        gr.update(visible=True),  # diffusion_controls
        gr.update(visible=False),  # lora_controls
        gr.update(maximum=max_dur, value=default_dur),
        gr.update(value=int(p.get("steps", 100))),
        gr.update(value=float(p.get("cfg_scale", 7.0))),
        gr.update(value=p.get("sampler_type", "dpmpp-3m-sde")),
        gr.update(value=float(p.get("sigma_min", 0.03))),
        gr.update(value=float(p.get("sigma_max", 500.0))),
        gr.update(visible=True),  # inpaint_content
        gr.update(visible=False),  # inpaint_unsupported
    )


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------


def create_uncond_sampling_ui(
    model_config: dict[str, Any], project_component: Any
) -> None:
    import gradio as gr

    generate_button = gr.Button("Generate", variant="primary", scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=500,
                    step=1,
                    value=100,
                    label="Steps",
                    info="More steps = higher quality but slower. 100 is a good default, 50 for quick drafts.",
                )
                seed_input = gr.Number(
                    label="Seed (-1 = random)",
                    value=-1,
                    precision=0,
                    info="Lock this number to reproduce the exact same output. -1 means random each time.",
                )

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(
                        [
                            "dpmpp-2m-sde",
                            "dpmpp-3m-sde",
                            "k-heun",
                            "k-lms",
                            "k-dpmpp-2s-ancestral",
                            "k-dpm-2",
                            "k-dpm-fast",
                        ],
                        label="Sampler type",
                        value="dpmpp-3m-sde",
                        allow_custom_value=True,
                        info="The algorithm used to generate audio. dpmpp-3m-sde is recommended for most use cases.",
                    )
                    sigma_min_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=0.03,
                        label="Sigma min",
                        info="Lower bound of the noise schedule. Leave at default unless you know what you're doing.",
                    )
                    sigma_max_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1000.0,
                        step=0.1,
                        value=500,
                        label="Sigma max",
                        info="Upper bound of the noise schedule. Leave at default unless you know what you're doing.",
                    )

            with gr.Accordion("Init audio", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use init audio")
                init_audio_input = gr.Audio(label="Init audio")
                init_noise_level_slider = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    step=0.01,
                    value=0.1,
                    label="Init noise level",
                    info="How much noise to add to the init audio before regenerating. Higher = more variation from the original.",
                )

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(
                label="Output spectrogram", show_label=False
            )
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(
                fn=lambda a: a, inputs=[audio_output], outputs=[init_audio_input]
            )

    generate_button.click(
        fn=generate_uncond,
        inputs=[
            steps_slider,
            seed_input,
            sampler_type_dropdown,
            sigma_min_slider,
            sigma_max_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
            project_component,
        ],
        outputs=[audio_output, audio_spectrogram_output],
        api_name="generate",
    )


def create_sampling_ui(
    model_config: dict[str, Any],
    project_component: Any,
    inpainting: bool = False,
) -> None:
    import gradio as gr

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(
                show_label=False,
                placeholder="Prompt",
                info="Describe the sound you want to generate — be specific about qualities like 'warm', 'sharp', 'reverberant', 'dry'.",
            )
            negative_prompt = gr.Textbox(
                show_label=False,
                placeholder="Negative prompt",
                info="Describe what you don't want in the output, e.g. 'music', 'reverb', 'noise'. Leave blank to skip.",
            )
        generate_button = gr.Button("Generate", variant="primary", scale=1)

    model_conditioning_config = model_config["model"].get("conditioning", None)
    has_seconds_start = False
    has_seconds_total = False
    seconds_total_val = 0.0
    seconds_itv = 0.5

    if model_conditioning_config is not None:
        for c in model_conditioning_config["configs"]:
            if c["id"] == "seconds_start":
                has_seconds_start = True
            if c["id"] == "seconds_total":
                has_seconds_total = True
                seconds_total_val = (
                    model_config["sample_size"] / model_config["sample_rate"]
                )
                seconds_total_val = int(seconds_total_val / seconds_itv) * seconds_itv

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row(visible=has_seconds_start or has_seconds_total):
                seconds_start_slider = gr.Slider(
                    minimum=0,
                    maximum=seconds_total_val,
                    step=seconds_itv,
                    value=0,
                    label="Seconds start",
                    visible=has_seconds_start,
                    info="Where in the model's audio timeline generation starts. Keep at 0 unless you want a specific placement.",
                )
                seconds_total_slider = gr.Slider(
                    minimum=0,
                    maximum=seconds_total_val,
                    step=seconds_itv,
                    value=seconds_total_val,
                    label="Seconds total",
                    visible=has_seconds_total,
                    info="How long the generated audio will be, in seconds. The output file is trimmed to this length.",
                )

            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=500,
                    step=1,
                    value=100,
                    label="Steps",
                    info="More steps = higher quality but slower. 100 is a good default, 50 for quick drafts.",
                )
                preview_every_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Preview Every",
                    info="Show a spectrogram preview every N steps while generating. 0 to disable — previews slow things down.",
                )
                cfg_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=25.0,
                    step=0.1,
                    value=7.0,
                    label="CFG scale",
                    info="How closely the output follows your prompt. Higher = more literal, lower = more creative. Start around 7.",
                )

            with gr.Row():
                seed_input = gr.Number(
                    label="Seed (-1 = random)",
                    value=-1,
                    precision=0,
                    info="Lock this number to reproduce the exact same output. -1 means random each time.",
                )

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(
                        [
                            "dpmpp-2m-sde",
                            "dpmpp-3m-sde",
                            "k-heun",
                            "k-lms",
                            "k-dpmpp-2s-ancestral",
                            "k-dpm-2",
                            "k-dpm-fast",
                        ],
                        label="Sampler type",
                        value="dpmpp-3m-sde",
                        allow_custom_value=True,
                        info="The algorithm used to generate audio. dpmpp-3m-sde is recommended for most use cases.",
                    )
                    sigma_min_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=0.03,
                        label="Sigma min",
                        info="Lower bound of the noise schedule. Leave at default unless you know what you're doing.",
                    )
                    sigma_max_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1000.0,
                        step=0.1,
                        value=500,
                        label="Sigma max",
                        info="Upper bound of the noise schedule. Leave at default unless you know what you're doing.",
                    )
                    cfg_rescale_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.01,
                        value=0.0,
                        label="CFG rescale amount",
                        info="Reduces CFG artifacts at high guidance scales. Try 0.7 if outputs sound distorted at high CFG. 0 = off.",
                    )

            if inpainting:
                with gr.Accordion("Inpainting", open=False):
                    sigma_max_slider.maximum = 1000
                    init_audio_checkbox = gr.Checkbox(
                        label="Do inpainting — regenerate only a portion of an existing file using the mask settings below"
                    )
                    init_audio_input = gr.Audio(
                        label="Init audio — the file to use as a starting point for inpainting or variation"
                    )
                    init_noise_level_slider = gr.Slider(
                        minimum=0.1,
                        maximum=100.0,
                        step=0.1,
                        value=80,
                        label="Init audio noise level",
                        visible=False,
                        info="How much noise to add before regenerating the masked region. 80 is a good starting point.",
                    )
                    mask_cropfrom_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=0,
                        label="Crop From %",
                        info="Where (%) in the source audio to start cropping the section you want to replace.",
                    )
                    mask_pastefrom_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=0,
                        label="Paste From %",
                        info="Where (%) in the output to paste the regenerated region.",
                    )
                    mask_pasteto_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=100,
                        label="Paste To %",
                        info="Where (%) in the output the pasted region ends.",
                    )
                    mask_maskstart_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=50,
                        label="Mask Start %",
                        info="Where (%) within the pasted region the inpainting mask begins.",
                    )
                    mask_maskend_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=100,
                        label="Mask End %",
                        info="Where (%) within the pasted region the inpainting mask ends.",
                    )
                    mask_softnessL_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=0,
                        label="Softmask Left Crossfade Length %",
                        info="How gradually (%) the mask fades in on the left edge. 0 = hard cut, higher = smoother blend.",
                    )
                    mask_softnessR_slider = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=0,
                        label="Softmask Right Crossfade Length %",
                        info="How gradually (%) the mask fades out on the right edge. 0 = hard cut, higher = smoother blend.",
                    )
                    mask_marination_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.0001,
                        value=0,
                        label="Marination level",
                        visible=False,
                    )

                    inputs = [
                        prompt,
                        negative_prompt,
                        seconds_start_slider,
                        seconds_total_slider,
                        cfg_scale_slider,
                        steps_slider,
                        preview_every_slider,
                        seed_input,
                        sampler_type_dropdown,
                        sigma_min_slider,
                        sigma_max_slider,
                        cfg_rescale_slider,
                        init_audio_checkbox,
                        init_audio_input,
                        init_noise_level_slider,
                        mask_cropfrom_slider,
                        mask_pastefrom_slider,
                        mask_pasteto_slider,
                        mask_maskstart_slider,
                        mask_maskend_slider,
                        mask_softnessL_slider,
                        mask_softnessR_slider,
                        mask_marination_slider,
                        gr.State(1),  # batch_size placeholder
                        project_component,
                    ]
            else:
                with gr.Accordion("Init audio", open=False):
                    init_audio_checkbox = gr.Checkbox(
                        label="Use init audio — upload audio to guide the generation, creating a variation rather than from scratch"
                    )
                    init_audio_input = gr.Audio(
                        label="Init audio — reference file for variation (higher init noise level = more deviation from this file)"
                    )
                    init_noise_level_slider = gr.Slider(
                        minimum=0.1,
                        maximum=100.0,
                        step=0.01,
                        value=0.1,
                        label="Init noise level",
                        info="How much noise to add to the init audio before regenerating. Higher = more variation from the original.",
                    )

                inputs = [
                    prompt,
                    negative_prompt,
                    seconds_start_slider,
                    seconds_total_slider,
                    cfg_scale_slider,
                    steps_slider,
                    preview_every_slider,
                    seed_input,
                    sampler_type_dropdown,
                    sigma_min_slider,
                    sigma_max_slider,
                    cfg_rescale_slider,
                    init_audio_checkbox,
                    init_audio_input,
                    init_noise_level_slider,
                    gr.State(None),  # mask_cropfrom
                    gr.State(None),  # mask_pastefrom
                    gr.State(None),  # mask_pasteto
                    gr.State(None),  # mask_maskstart
                    gr.State(None),  # mask_maskend
                    gr.State(None),  # mask_softnessL
                    gr.State(None),  # mask_softnessR
                    gr.State(None),  # mask_marination
                    gr.State(1),  # batch_size
                    project_component,
                ]

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(
                label="Output spectrogram", show_label=False
            )
            metadata_output = gr.JSON(label="Generation metadata")
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(
                fn=lambda a: a, inputs=[audio_output], outputs=[init_audio_input]
            )

    generate_button.click(
        fn=generate_cond,
        inputs=inputs,
        outputs=[audio_output, audio_spectrogram_output, metadata_output],
        api_name="generate",
    )

    return (
        steps_slider,
        cfg_scale_slider,
        sampler_type_dropdown,
        sigma_min_slider,
        sigma_max_slider,
    )


def create_acestep_ui(
    project_component: Any, default_params: dict | None = None
) -> tuple:
    """Build the Gradio UI panel for ACE-Step generation.

    Shows a Prompt (tags) field and a Lyrics field alongside the standard
    duration, steps, CFG, and seed controls.

    Args:
        project_component: The shared ``gr.Textbox`` for the project name.
        default_params:    Registry default_params dict; slider values are
                           initialised from ``steps``, ``cfg_scale``, and
                           ``audio_duration`` when present.

    Returns:
        ``(seconds_total_slider, steps_slider, cfg_scale_slider)`` — component
        references that the Load Model button handler updates with the newly
        loaded model's registry default_params.
    """
    import gradio as gr

    dp = default_params or {}
    default_steps = int(dp.get("steps", 50))
    default_cfg = float(dp.get("cfg_scale", 4.0))
    default_duration = int(dp.get("audio_duration", 60))

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(
                show_label=False,
                placeholder="Tags / style prompt (e.g. 'upbeat indie pop, electric guitar, energetic')",
                info=(
                    "Describe the style, genre, instruments, and mood.  "
                    "ACE-Step responds best to comma-separated tag lists rather than "
                    "full sentences."
                ),
            )
            lyrics = gr.Textbox(
                show_label=False,
                placeholder="Lyrics  (optional — leave blank or type '[Instrumental]' for no vocals)",
                lines=4,
                info=(
                    "Structure lyrics with section markers like [verse], [chorus], "
                    "[bridge].  Leave blank for an instrumental track."
                ),
            )
            negative_prompt = gr.Textbox(
                show_label=False,
                placeholder="Negative prompt",
                info=(
                    "Describe what you don't want in the output. For ACE-Step, "
                    "this controls the LM/thinking path when enabled."
                ),
            )
        generate_button = gr.Button("Generate", variant="primary", scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                seconds_total_slider = gr.Slider(
                    minimum=5,
                    maximum=240,
                    step=5,
                    value=default_duration,
                    label="Duration (seconds)",
                    info="Target audio length.  ACE-Step supports up to ~4 minutes.",
                )
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=150,
                    step=1,
                    value=default_steps,
                    label="Steps",
                    info="Diffusion steps.  50 = turbo (fast), 100 = full quality.",
                )
                cfg_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    step=0.5,
                    value=default_cfg,
                    label="CFG scale",
                    info="Guidance strength.  Higher = more literal prompt, lower = more creative.",
                )
                seed_input = gr.Number(
                    label="Seed (-1 = random)",
                    value=-1,
                    precision=0,
                    info="Lock to reproduce the exact same output.",
                )

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(
                label="Output spectrogram", show_label=False
            )
            metadata_output = gr.JSON(label="Generation metadata")

    generate_button.click(
        fn=generate_acestep,
        inputs=[
            prompt,
            lyrics,
            negative_prompt,
            seconds_total_slider,
            steps_slider,
            cfg_scale_slider,
            seed_input,
            project_component,
        ],
        outputs=[audio_output, audio_spectrogram_output, metadata_output],
        api_name="generate",
    )

    return seconds_total_slider, steps_slider, cfg_scale_slider


def create_txt2audio_ui(model_config: dict[str, Any], project_component: Any) -> Any:
    import gradio as gr

    with gr.Blocks():
        with gr.Tab("Generation"):
            gen_params = create_sampling_ui(model_config, project_component)
        with gr.Tab("Inpainting"):
            inpaint_params = create_sampling_ui(
                model_config, project_component, inpainting=True
            )
    return gen_params + inpaint_params


def create_diffusion_uncond_ui(
    model_config: dict[str, Any], project_component: Any
) -> Any:
    import gradio as gr

    with gr.Blocks() as ui:
        create_uncond_sampling_ui(model_config, project_component)
    return ui


def create_autoencoder_ui(model_config: dict[str, Any], project_component: Any) -> Any:
    import gradio as gr

    is_dac_rvq = (
        "model" in model_config
        and "bottleneck" in model_config["model"]
        and model_config["model"]["bottleneck"]["type"] in {"dac_rvq", "dac_rvq_vae"}
    )
    n_quantizers = (
        model_config["model"]["bottleneck"]["config"]["n_codebooks"]
        if is_dac_rvq
        else 0
    )

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        n_quantizers_slider = gr.Slider(
            minimum=1,
            maximum=n_quantizers,
            step=1,
            value=n_quantizers,
            label="# quantizers",
            visible=is_dac_rvq,
            info="Number of RVQ codebooks to use. Fewer = smaller file size, more lossy. Use max for best quality.",
        )
        latent_noise_slider = gr.Slider(
            minimum=0.0,
            maximum=10.0,
            step=0.001,
            value=0.0,
            label="Add latent noise",
            info="Add random noise to the encoded latent before decoding. 0 = clean reconstruction, higher = creative variation.",
        )
        process_button = gr.Button("Process", variant="primary", scale=1)
        process_button.click(
            fn=autoencoder_process,
            inputs=[
                input_audio,
                latent_noise_slider,
                n_quantizers_slider,
                project_component,
            ],
            outputs=output_audio,
            api_name="process",
        )
    return ui


def create_diffusion_prior_ui(
    model_config: dict[str, Any], project_component: Any
) -> Any:
    import gradio as gr

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        with gr.Row():
            steps_slider = gr.Slider(
                minimum=1,
                maximum=500,
                step=1,
                value=100,
                label="Steps",
                info="More steps = higher quality but slower. 100 is a good default, 50 for quick drafts.",
            )
            sampler_type_dropdown = gr.Dropdown(
                [
                    "dpmpp-2m-sde",
                    "dpmpp-3m-sde",
                    "k-heun",
                    "k-lms",
                    "k-dpmpp-2s-ancestral",
                    "k-dpm-2",
                    "k-dpm-fast",
                ],
                label="Sampler type",
                value="dpmpp-3m-sde",
                info="The algorithm used to generate audio. dpmpp-3m-sde is recommended for most use cases.",
            )
            sigma_min_slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                value=0.03,
                label="Sigma min",
                info="Lower bound of the noise schedule. Leave at default unless you know what you're doing.",
            )
            sigma_max_slider = gr.Slider(
                minimum=0.0,
                maximum=1000.0,
                step=0.1,
                value=500,
                label="Sigma max",
                info="Upper bound of the noise schedule. Leave at default unless you know what you're doing.",
            )
        process_button = gr.Button("Process", variant="primary", scale=1)
        process_button.click(
            fn=diffusion_prior_process,
            inputs=[
                input_audio,
                steps_slider,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                project_component,
            ],
            outputs=output_audio,
            api_name="process",
        )
    return ui


def create_lm_ui(model_config: dict[str, Any], project_component: Any) -> Any:
    import gradio as gr

    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(
            label="Output spectrogram", show_label=False
        )
        with gr.Row():
            temperature_slider = gr.Slider(
                minimum=0,
                maximum=5,
                step=0.01,
                value=1.0,
                label="Temperature",
                info="Controls randomness. Higher = more surprising/varied output, lower = more predictable. 1.0 is neutral.",
            )
            top_p_slider = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.95,
                label="Top p",
                info="Nucleus sampling threshold. Keeps only the most likely tokens summing to this probability. 0.95 is standard.",
            )
            top_k_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=0,
                label="Top k",
                info="Limits sampling to the top K most likely tokens at each step. 0 = disabled (use top_p instead).",
            )
        generate_button = gr.Button("Generate", variant="primary", scale=1)
        generate_button.click(
            fn=generate_lm,
            inputs=[temperature_slider, top_p_slider, top_k_slider, project_component],
            outputs=[output_audio, audio_spectrogram_output],
            api_name="generate",
        )
    return ui


def _list_recent_sidecars(project: str) -> list[tuple[str, str]]:
    """Return (stem_name, path_str) tuples for the 10 most recent JSON sidecars."""
    output_dir = Path.home() / "anvil-audio-outputs" / (project or "default")
    if not output_dir.exists():
        return []
    sidecars = [
        p for p in output_dir.rglob("*.json") if p.name != "batch_manifest.json"
    ]
    sidecars.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.stem, str(p)) for p in sidecars[:10]]


def _apply_preset(file_path: str | None, project: str) -> tuple:
    """Parse a sidecar JSON and return gr.update() values for all preset fields."""
    import gradio as gr

    _no_op = gr.update()
    _no_ops = (_no_op,) * 11

    if not file_path:
        return _no_ops

    try:
        with open(file_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        return (
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            _no_op,
            gr.update(value=f"⚠ Could not read preset: {exc}"),
        )

    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    seed = data.get("seed", -1)
    steps = data.get("steps", 100)
    cfg_scale = data.get("cfg_scale", 7.0)
    sampler_type = data.get("sampler_type", "dpmpp-3m-sde")
    sigma_min = data.get("sigma_min", 0.03)
    sigma_max = data.get("sigma_max", 500.0)
    lyrics = data.get("extra", {}).get("lyrics", "")

    raw_seconds_total = data.get("seconds_total", 0.0)
    seconds_total = (
        raw_seconds_total if raw_seconds_total > 0 else data.get("duration_seconds", 30)
    )

    preset_model = data.get("model_name", "")
    if preset_model and preset_model != _model_name:
        if registry.get_model(preset_model) is not None:
            status = f"ℹ This preset was made with **{preset_model}** — switch models to match."
        else:
            status = f"ℹ This preset was made with **{preset_model}**."
    else:
        status = ""

    return (
        gr.update(value=prompt),
        gr.update(value=lyrics),
        gr.update(value=negative_prompt),
        gr.update(value=seed),
        gr.update(value=seconds_total),
        gr.update(value=steps),
        gr.update(value=cfg_scale),
        gr.update(value=sampler_type),
        gr.update(value=sigma_min),
        gr.update(value=sigma_max),
        gr.update(value=status),
    )


def _enhance_prompt_ui(
    prompt: str,
    negative_prompt: str,
    seconds_total: float,
) -> tuple[str, str]:
    import gradio as gr

    try:
        from anvil_audio.intelligence import enhance_prompt

        package = enhance_prompt(
            prompt,
            mode="music" if _pipeline_type == "acestep" else "audio",
            duration_seconds=float(seconds_total or 60.0),
            negative_prompt=negative_prompt or "",
        )
        return package.prompt, package.negative_prompt
    except Exception as exc:
        gr.Warning(f"Prompt enhancement failed: {exc}")
        return prompt, negative_prompt


def _write_lyrics_ui(prompt: str, seconds_total: float) -> str:
    import gradio as gr

    try:
        from anvil_audio.intelligence import write_lyrics

        return write_lyrics(prompt, duration_seconds=float(seconds_total or 60.0))
    except Exception as exc:
        gr.Warning(f"Lyric writing failed: {exc}")
        return ""


def _prepare_song_ui(
    prompt: str,
    negative_prompt: str,
    seconds_total: float,
) -> tuple[str, str, str]:
    import gradio as gr

    try:
        from anvil_audio.intelligence import prepare_song_prompt

        package = prepare_song_prompt(
            prompt,
            duration_seconds=float(seconds_total or 60.0),
            negative_prompt=negative_prompt or "",
            write_vocals=True,
            enhance=True,
        )
        return package.prompt, package.negative_prompt, package.lyrics
    except Exception as exc:
        gr.Warning(f"Prompt/lyric preparation failed: {exc}")
        return prompt, negative_prompt, ""


def create_unified_txt2music_ui(
    project_component: Any,
    initial_model_type: str = "diffusion_cond",
    initial_params: dict | None = None,
    model_config: dict[str, Any] | None = None,
    initial_max_duration: float | None = None,
) -> tuple:
    """Build one panel covering both diffusion_cond and ACE-Step pipelines.

    Visibility of pipeline-specific rows is toggled at runtime by the Load Model
    button via _model_load_ui_unified.

    Returns:
        (lyrics_row, neg_prompt_row, intelligence_lyrics_row, diffusion_controls,
         lora_controls, seconds_total_slider, steps_slider, cfg_scale_slider,
         sampler_type_dropdown, sigma_min_slider, sigma_max_slider)
    """
    import gradio as gr

    dp = initial_params or {}
    is_acestep = initial_model_type == "acestep"
    is_diffusion = not is_acestep

    has_seconds_start = False
    if model_config is not None and is_diffusion:
        mc = model_config.get("model", {}).get("conditioning")
        if mc:
            for c in mc.get("configs", []):
                if c["id"] == "seconds_start":
                    has_seconds_start = True

    if is_acestep:
        default_steps = int(dp.get("steps", 50))
        default_cfg = float(dp.get("cfg_scale", 4.0))
        default_duration = int(dp.get("audio_duration", 60))
        default_sampler = "ode"
        default_sigma_min = 0.0
        default_sigma_max = 0.0
        slider_max = initial_max_duration or 600.0
    else:
        default_steps = int(dp.get("steps", 100))
        default_cfg = float(dp.get("cfg_scale", 7.0))
        if model_config is not None:
            raw = model_config.get("sample_size", 1920000) / model_config.get(
                "sample_rate", 32000
            )
            default_duration = int(raw / 0.5) * 0.5
        else:
            default_duration = 30.0
        default_sampler = dp.get("sampler_type", "dpmpp-3m-sde")
        default_sigma_min = float(dp.get("sigma_min", 0.03))
        default_sigma_max = float(dp.get("sigma_max", 500.0))
        if initial_max_duration is not None:
            slider_max = initial_max_duration
        elif model_config is not None:
            slider_max = model_config.get("sample_size", 1920000) / model_config.get(
                "sample_rate", 32000
            )
        else:
            slider_max = 240.0

    # 1. Prompt
    prompt = gr.Textbox(
        show_label=False,
        placeholder="Prompt",
        lines=3,
        max_lines=10,
        info="Describe the sound or music you want to generate.",
    )

    # 2. Lyrics (ACE-Step only) / Negative prompt
    with gr.Row(visible=is_acestep) as lyrics_row:
        lyrics = gr.Textbox(
            show_label=False,
            placeholder="Lyrics  (optional — leave blank or type '[Instrumental]' for no vocals)",
            lines=8,
            max_lines=24,
            info="Structure lyrics with section markers like [verse], [chorus], [bridge].",
        )
    with gr.Row(visible=True) as neg_prompt_row:
        negative_prompt = gr.Textbox(
            show_label=False,
            placeholder="Negative prompt",
            lines=2,
            max_lines=6,
            info="Describe what you don't want in the output. Leave blank to skip.",
        )

    with gr.Row():
        enhance_button = gr.Button("Enhance Prompt", variant="secondary")
        with gr.Row(visible=is_acestep) as intelligence_lyrics_row:
            write_lyrics_button = gr.Button("Write Lyrics", variant="secondary")
            prepare_song_button = gr.Button("Enhance + Lyrics", variant="secondary")

    with gr.Group(visible=is_acestep) as lora_controls:
        with gr.Accordion("ACE-Step LoRA", open=False):
            with gr.Row():
                lora_reference = gr.Dropdown(
                    choices=_lora_dropdown_choices(),
                    value="",
                    label="Adapter",
                    allow_custom_value=True,
                    filterable=True,
                    info=(
                        "Pick a registered adapter or paste a PEFT/LoKr path. "
                        "Use refresh after importing a new adapter."
                    ),
                    scale=3,
                )
                lora_scale = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                    label="LoRA scale",
                    info="Adapter strength. 1.0 is full strength; lower values blend with the base model.",
                )
                lora_refresh = gr.Button("↻ Refresh", variant="secondary", scale=1)
            lora_stack = gr.Textbox(
                value="",
                label="Additional adapters",
                lines=2,
                placeholder="adapter-two:0.5, /path/to/adapter-three:0.25",
            )

            lora_refresh.click(
                fn=_refresh_lora_dropdown,
                inputs=[lora_reference],
                outputs=[lora_reference],
                show_progress="hidden",
            )

    # 3. Generation controls — single row
    with gr.Row():
        seconds_start_slider = gr.Slider(
            minimum=0,
            maximum=240,
            step=0.5,
            value=0,
            label="Seconds start",
            visible=(is_diffusion and has_seconds_start),
            info="Where in the audio timeline generation starts.",
        )
        seconds_total_slider = gr.Slider(
            minimum=0,
            maximum=slider_max,
            step=1,
            value=min(default_duration, slider_max),
            label="Duration (seconds)",
            info="Target audio length in seconds.",
        )
        steps_slider = gr.Slider(
            minimum=1,
            maximum=500,
            step=1,
            value=default_steps,
            label="Steps",
            info="More steps = higher quality but slower.",
        )
        cfg_scale_slider = gr.Slider(
            minimum=0,
            maximum=30,
            step=0.1,
            value=default_cfg,
            label="CFG scale",
            info="How closely the output follows your prompt.",
        )
        seed_input = gr.Number(
            label="Seed (-1 = random)",
            value=-1,
            precision=0,
            info="Lock this number to reproduce the exact same output.",
        )

    # Diffusion-only advanced controls (hidden for ACE-Step)
    with gr.Group(visible=is_diffusion) as diffusion_controls:
        with gr.Accordion("Sampler params", open=False):
            with gr.Row():
                preview_every_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Preview Every",
                    info="Show a spectrogram preview every N steps. 0 to disable.",
                )
                sampler_type_dropdown = gr.Dropdown(
                    [
                        "dpmpp-2m-sde",
                        "dpmpp-3m-sde",
                        "k-heun",
                        "k-lms",
                        "k-dpmpp-2s-ancestral",
                        "k-dpm-2",
                        "k-dpm-fast",
                    ],
                    label="Sampler type",
                    value=default_sampler,
                    allow_custom_value=True,
                    info="The algorithm used to generate audio.",
                )
                sigma_min_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.01,
                    value=default_sigma_min,
                    label="Sigma min",
                    info="Lower bound of the noise schedule.",
                )
                sigma_max_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1000.0,
                    step=0.1,
                    value=default_sigma_max,
                    label="Sigma max",
                    info="Upper bound of the noise schedule.",
                )
                cfg_rescale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1,
                    step=0.01,
                    value=0.0,
                    label="CFG rescale amount",
                    info="Reduces CFG artifacts at high guidance scales.",
                )

        with gr.Accordion("Init audio", open=False):
            init_audio_checkbox = gr.Checkbox(
                label="Use init audio — upload audio to guide the generation",
            )
            init_audio_input = gr.Audio(
                label="Init audio — reference file for variation",
            )
            init_noise_level_slider = gr.Slider(
                minimum=0.1,
                maximum=100.0,
                step=0.01,
                value=0.1,
                label="Init noise level",
                info="How much noise to add to the init audio before regenerating.",
            )

    # 4. Generate button + format selector
    with gr.Row():
        generate_button = gr.Button("Generate", variant="primary", scale=4)
        audio_format_dropdown = gr.Dropdown(
            ["wav", "mp3", "ogg"],
            value="wav",
            label="Format",
            scale=1,
            min_width=80,
            info="wav = lossless · mp3 = compressed · ogg = game engines (Roblox, Unity, Godot)",
        )

    # 5. Output
    with gr.Row():
        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(
                label="Output spectrogram", show_label=False
            )
        with gr.Column():
            metadata_output = gr.JSON(label="Generation metadata")
            send_to_init_button = gr.Button("Send to init audio", variant="secondary")
            send_to_init_button.click(
                fn=lambda a: a,
                inputs=[audio_output],
                outputs=[init_audio_input],
            )

    # 6. Load Preset (below output, collapsed by default)
    with gr.Accordion("Load Preset", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                preset_recent = gr.Dropdown(
                    choices=_list_recent_sidecars(_default_project or "default"),
                    label="Load Recent",
                    value=None,
                    interactive=True,
                    info="10 most recent generations from the current project.",
                )
                preset_refresh_btn = gr.Button("↻ Refresh recent", variant="secondary")
            with gr.Column(scale=2):
                preset_file = gr.File(
                    label="Upload sidecar (.json)",
                    file_types=[".json"],
                    type="filepath",
                )
        preset_status = gr.Markdown("")

    generate_button.click(
        fn=generate_unified,
        inputs=[
            prompt,
            lyrics,
            negative_prompt,
            seconds_start_slider,
            seconds_total_slider,
            steps_slider,
            preview_every_slider,
            cfg_scale_slider,
            seed_input,
            sampler_type_dropdown,
            sigma_min_slider,
            sigma_max_slider,
            cfg_rescale_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
            project_component,
            audio_format_dropdown,
            lora_reference,
            lora_scale,
            lora_stack,
        ],
        outputs=[audio_output, audio_spectrogram_output, metadata_output],
        api_name="generate",
    )
    enhance_button.click(
        fn=_enhance_prompt_ui,
        inputs=[prompt, negative_prompt, seconds_total_slider],
        outputs=[prompt, negative_prompt],
    )
    write_lyrics_button.click(
        fn=_write_lyrics_ui,
        inputs=[prompt, seconds_total_slider],
        outputs=[lyrics],
    )
    prepare_song_button.click(
        fn=_prepare_song_ui,
        inputs=[prompt, negative_prompt, seconds_total_slider],
        outputs=[prompt, negative_prompt, lyrics],
    )

    _preset_outputs = [
        prompt,
        lyrics,
        negative_prompt,
        seed_input,
        seconds_total_slider,
        steps_slider,
        cfg_scale_slider,
        sampler_type_dropdown,
        sigma_min_slider,
        sigma_max_slider,
        preset_status,
    ]

    preset_file.change(
        fn=_apply_preset,
        inputs=[preset_file, project_component],
        outputs=_preset_outputs,
    )
    preset_recent.change(
        fn=_apply_preset,
        inputs=[preset_recent, project_component],
        outputs=_preset_outputs,
    )
    preset_refresh_btn.click(
        fn=lambda proj: gr.update(
            choices=_list_recent_sidecars(proj or "default"), value=None
        ),
        inputs=[project_component],
        outputs=[preset_recent],
    )

    return (
        lyrics_row,
        neg_prompt_row,
        intelligence_lyrics_row,
        diffusion_controls,
        lora_controls,
        seconds_total_slider,
        steps_slider,
        cfg_scale_slider,
        sampler_type_dropdown,
        sigma_min_slider,
        sigma_max_slider,
    )


# ---------------------------------------------------------------------------
# Top-level UI factory
# ---------------------------------------------------------------------------


def create_ui(
    model_config_path: str | None = None,
    ckpt_path: str | None = None,
    pretrained_name: str | None = None,
    pretransform_ckpt_path: str | None = None,
    model_half: bool = False,
    tmp_dir: str = "",
    device: torch.device | None = None,
    project: str = "",
    model_name: str | None = None,
) -> Any:
    """Build and return the Gradio ``Blocks`` interface.

    Args:
        model_config_path:      Path to a local JSON model config.
        ckpt_path:              Path to a local checkpoint.
        pretrained_name:        HuggingFace Hub repo ID.
        pretransform_ckpt_path: Optional separate pretransform checkpoint.
        model_half:             Use float16 inference.
        tmp_dir:                Legacy parameter (ignored; output manager handles paths).
        device:                 Target device.  Auto-detected if ``None``.
        project:                Default project name for output routing.
        model_name:             Registry name of the model being loaded
                                (used to detect ACE-Step pipeline types).

    Returns:
        A ``gradio.Blocks`` instance ready for ``.queue()`` and ``.launch()``.
    """
    import gradio as gr

    # Determine whether this is an ACE-Step model via the registry.
    _entry = registry.get_model(model_name) if model_name else None
    is_acestep = (
        _entry is not None
        and getattr(_entry, "pipeline_type", "diffusion") == "acestep"
    )

    model_type: str
    loaded_config: dict[str, Any]

    if is_acestep:
        assert _entry is not None  # guaranteed by is_acestep condition
        load_acestep_model(entry=_entry, device=device, project=project)
        # Synthetic config so the rest of the function can branch on model_type.
        model_type = "acestep"
        loaded_config = {"model_type": "acestep"}
        initial_status = f"Loaded ACE-Step: {model_name}"
    else:
        assert exists(pretrained_name) ^ (
            exists(model_config_path) and exists(ckpt_path)
        ), (
            "Provide either pretrained_name or (model_config_path + ckpt_path), not both."
        )

        model_config_dict: dict[str, Any] | None = None
        if exists(model_config_path):
            with open(model_config_path) as fh:  # type: ignore[arg-type]
                model_config_dict = json.load(fh)

        _, loaded_config = load_model(
            model_config=model_config_dict,
            model_ckpt_path=ckpt_path,
            pretrained_name=pretrained_name,
            pretransform_ckpt_path=pretransform_ckpt_path,
            device=device,
            model_half=model_half,
            project=project,
        )
        model_type = loaded_config["model_type"]
        initial_status = f"Loaded: {pretrained_name or 'custom'}"

    registered_models = [e.name for e in registry.list_models()]
    initial_model_value = (
        model_name
        if model_name in registered_models
        else (pretrained_name if pretrained_name in registered_models else None)
    )

    custom_theme_css = _build_custom_theme_css()

    with gr.Blocks(title="Stable Audio Tools") as interface:
        # ---- Global controls (outside tabs) ----
        with gr.Row():
            with gr.Column(scale=3):
                project_textbox = gr.Textbox(
                    label="Project",
                    placeholder="default",
                    value=project,
                    info="Keeps your files organized by project. Outputs go to ~/anvil-audio-outputs/{project}/. Leave blank for 'default'.",
                )
                gr.HTML(_github_star_html())
            with gr.Column(scale=3):
                model_dropdown = gr.Dropdown(
                    choices=registered_models,
                    value=initial_model_value,
                    label="Registered model",
                    info="Choose a registered model by name. Click Load to switch without restarting. Add your own in ~/.anvil-audio/registry.yaml.",
                )
                model_half_checkbox = gr.Checkbox(
                    label="Half precision (fp16) — cuts memory in half with minimal quality loss, recommended on GPUs",
                    value=model_half,
                )
            with gr.Column(scale=2):
                device_textbox = gr.Textbox(
                    label="Device",
                    value=str(device) if device else str(get_best_device()),
                    info="Device to run generation on. 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' as a last resort (very slow).",
                )
                load_model_btn = gr.Button("Load Model", variant="secondary")

        model_status = gr.Textbox(
            label="Model status",
            value=initial_status,
            interactive=False,
        )

        with gr.Accordion("Appearance", open=True):
            with gr.Row():
                theme_dropdown = gr.Dropdown(
                    choices=_theme_dropdown_choices(),
                    value=_THEME_DEFAULT_VALUE,
                    label="Theme preset",
                    info="Runtime theme accents inspired by Gradio's bundled themes. The built-in system/light/dark setting still controls brightness.",
                    elem_classes=["anvil-theme-picker"],
                    scale=2,
                )
                theme_status = gr.Markdown(_theme_markdown(_THEME_DEFAULT_VALUE))

            theme_dropdown.change(
                fn=None,
                inputs=[theme_dropdown],
                outputs=[theme_dropdown, theme_status],
                js=_THEME_APPLY_JS,
                show_progress="hidden",
            )
            interface.load(
                fn=None,
                outputs=[theme_dropdown, theme_status],
                js=_THEME_LOAD_JS,
                show_progress="hidden",
            )

        gr.Markdown("---")

        # ---- Model-type-specific UI + Edit tab ----
        _btn_inputs = [
            model_dropdown,
            project_textbox,
            model_half_checkbox,
            device_textbox,
        ]

        from .edit_tab import create_edit_tab

        with gr.Tabs():
            with gr.Tab("Generate"):
                if model_type in {
                    "diffusion_cond",
                    "diffusion_cond_inpaint",
                    "acestep",
                }:
                    # Unified panel handles both diffusion and ACE-Step with show/hide
                    _mc = loaded_config if model_type != "acestep" else None
                    _init_params = _entry.resolved_params() if _entry else {}
                    _max_dur = _entry.max_duration if _entry else None
                    param_comps = create_unified_txt2music_ui(
                        project_textbox,
                        initial_model_type=model_type,
                        initial_params=_init_params,
                        model_config=_mc,
                        initial_max_duration=_max_dur,
                    )
                else:
                    # Non-text2music model types keep their existing dedicated UIs
                    if model_type == "diffusion_uncond":
                        create_diffusion_uncond_ui(loaded_config, project_textbox)
                    elif model_type in {"autoencoder", "diffusion_autoencoder"}:
                        create_autoencoder_ui(loaded_config, project_textbox)
                    elif model_type == "diffusion_prior":
                        create_diffusion_prior_ui(loaded_config, project_textbox)
                    elif model_type == "lm":
                        create_lm_ui(loaded_config, project_textbox)

            if model_type in {"diffusion_cond", "diffusion_cond_inpaint", "acestep"}:
                with gr.Tab("Inpainting"):
                    # Always render the full inpainting UI so it's ready when the user
                    # switches from ACE-Step to a diffusion model.  Use a minimal fallback
                    # config when the initial model is ACE-Step so the sliders initialise
                    # with reasonable defaults.
                    _inpaint_cfg = (
                        loaded_config
                        if not is_acestep
                        else {
                            "model_type": "diffusion_cond",
                            "sample_rate": 44100,
                            "sample_size": 2076672,
                            "model": {
                                "conditioning": {
                                    "configs": [
                                        {"id": "seconds_start"},
                                        {"id": "seconds_total"},
                                    ]
                                }
                            },
                        }
                    )
                    with gr.Group(visible=not is_acestep) as _inpaint_content:
                        create_sampling_ui(
                            _inpaint_cfg, project_textbox, inpainting=True
                        )
                    with gr.Group(visible=is_acestep) as _inpaint_unsupported:
                        gr.Markdown(
                            "**Inpainting is not supported by the current model.**  \n"
                            "Load a Stable Audio diffusion model "
                            "(e.g. `stable-audio-open-1.0`) to use this tab."
                        )

            with gr.Tab("Edit"):
                create_edit_tab(
                    project_component=project_textbox,
                    last_path_getter=lambda: _last_generated_path,
                )

        # Wire the Load Model button after all tab components are defined.
        if model_type in {"diffusion_cond", "diffusion_cond_inpaint", "acestep"}:
            load_model_btn.click(
                fn=_model_load_ui_unified,
                inputs=_btn_inputs,
                outputs=[
                    model_status,
                    *param_comps,
                    _inpaint_content,
                    _inpaint_unsupported,
                ],
            )
        else:
            load_model_btn.click(
                fn=_model_load_ui,
                inputs=_btn_inputs,
                outputs=[model_status],
            )

    setattr(interface, _THEME_CSS_ATTR, custom_theme_css)
    return interface
