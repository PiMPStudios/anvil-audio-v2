# Stable Audio, MLX, and Model Registry

Stable Audio generation, Apple Silicon MLX acceleration, and custom registry entries.

## Stable Audio Models

### CLI

```bash
# Use a registered model by name
anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"

# List all registered models
anvil --list-models

# Batch generation from a YAML file
anvil generate --model stable-audio-open-1.0 --cond-yaml-path batch.yaml --output-dir ./out

# Legacy path (local config + checkpoint)
anvil generate --model-config config.json --ckpt-path model.ckpt \
    --prompt "rain on a tin roof" --output-dir ./out
```

Multi-GPU generation is supported via Accelerate.

### Gradio web UI

```bash
# Load by registry name (recommended)
python run_gradio.py --model stable-audio-open-1.0

# Load from HuggingFace Hub directly
python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0

# No args — loads the first model from your registry
python run_gradio.py

# Route outputs to a named project folder
python run_gradio.py --model stable-audio-open-1.0 --project sfx-pack-v1
```

#### Presets & Reproducibility

Every generation saves a JSON sidecar alongside the audio containing the full parameters — prompt, seed, steps, CFG scale, model, duration, and for edits the full effects chain. These sidecars double as presets: the **Load Recent** dropdown at the bottom of the Generate tab shows your last 10 generations from the current project, and selecting one pre-populates all fields instantly. **Load Preset** accepts any `.json` sidecar via drag-and-drop, so you can share settings with collaborators or reload a preset from a different project. Tweak any field after loading and hit Generate to create a variation.

---

## MLX Acceleration (Apple Silicon)

On M1/M2/M3/M4 Macs, Anvil can run Stable Audio inference through
[mlx-audiogen](https://github.com/jasonvassallo/mlx-audio-generate), which ports the DiT,
VAE, and T5 conditioner to Apple's native MLX framework. This runs directly on the Metal GPU
without going through PyTorch MPS.

**Benchmark — 30-second clip, `stable-audio-open-1.0`:**

| Backend | Time |
|---|---|
| PyTorch MPS | ~61 s |
| MLX (native Metal) | ~31 s |

### Enable MLX

```bash
pip install mlx-audiogen
```

That's it. Once installed, two new models appear in the registry automatically:

| Model name | Source model |
|---|---|
| `stable-audio-open-small-mlx` | `stabilityai/stable-audio-open-small` |
| `stable-audio-open-1.0-mlx` | `stabilityai/stable-audio-open-1.0` |

On first use, Anvil downloads the original HuggingFace weights and converts them to MLX
safetensors format. Converted weights are cached at:

```text
~/.cache/anvil-audio/mlx-weights/<model-slug>/
```

Subsequent loads skip conversion and go straight to inference.

### Usage

```bash
# CLI
anvil generate --model stable-audio-open-1.0-mlx --prompt "rain on leaves"

# Gradio — select from the model dropdown
python run_gradio.py --model stable-audio-open-1.0-mlx
```

MLX models use a rectified-flow sampler (`euler` or `rk4`). The `sigma_max` range is
`[0.01, 2.0]` — values outside this range (e.g. the PyTorch default of 500.0) are
automatically clamped to `1.0`.

### Requirements

- macOS on Apple Silicon (M1 or later)
- `pip install mlx-audiogen`

mlx-audiogen is an optional dependency — Anvil works normally on all platforms without it.
The MLX model entries only appear in the registry on Apple Silicon with mlx-audiogen installed.

### User registry — custom MLX weights

If you have pre-converted weights in a custom directory, point to them via `mlx_weights_dir`:

```yaml
- name: my-mlx-model
  pipeline_type: mlx_diffusion
  pretrained_name: stabilityai/stable-audio-open-small
  mlx_weights_dir: /path/to/my/converted/weights
  default_params:
    steps: 100
    cfg_scale: 7.0
    sampler_type: euler
    sigma_max: 1.0
```

---

## User Registry

Add your own models to `~/.anvil-audio/registry.yaml`. The file is a YAML list — entries
with the same name as a built-in will override it.

### Stable Audio / diffusion models

```yaml
- name: my-sfx-model
  pretrained_name: myorg/my-sfx-model        # HuggingFace Hub
  default_params:
    steps: 100
    cfg_scale: 7.0

- name: local-vae-dit
  model_config_path: /path/to/config.json
  ckpt_path: /path/to/model.ckpt
  pretransform_ckpt_path: /path/to/vae.ckpt
```

### MLX models (Apple Silicon)

```yaml
- name: my-mlx-model
  pipeline_type: mlx_diffusion
  pretrained_name: stabilityai/stable-audio-open-small
  # Optional: point to a directory with pre-converted MLX safetensors.
  # Omit to use the auto-convert cache at ~/.cache/anvil-audio/mlx-weights/
  mlx_weights_dir: /path/to/converted/weights
  default_params:
    steps: 100
    cfg_scale: 7.0
    sampler_type: euler
    sigma_max: 1.0
```

### ACE-Step models

```yaml
- name: my-acestep-finetune
  pipeline_type: acestep
  model_config_path: acestep-v15-sft          # checkpoint variant name
  # Built-in models use ~/.cache/anvil-audio/acestep automatically.
  # Set this only when using a local ACE-Step checkout or custom checkpoint root.
  acestep_project_root: /path/to/ACE-Step-1.5
  # Optional: override the LM checkpoint.
  # Omit to use the built-in default (1.7B for turbo, 4B for sft).
  lm_model_path: acestep-5Hz-lm-4B
  default_params:
    steps: 50
    cfg_scale: 7.5
    audio_duration: 60
    shift: 3.0
    lm_cfg_scale: 2.0
    thinking: false
    dcw_enabled: false
```

XL entries use the same schema, but XL checkpoints are intentionally opt-in.
Install them first with `acestep-download --dir "$HOME/.cache/anvil-audio/acestep/checkpoints" --model <checkpoint>`.
If an XL checkpoint is missing, Anvil refuses to auto-download it during model
load and prints the explicit install command.

---
