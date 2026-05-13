# Anvil Audio

> **A pluggable studio tool for AI audio generation. Swap models, keep your workflow.**

Anvil Audio is a refactored and extended fork of
[`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) by Stability AI.
It turns a single-model inference codebase into a clean, swappable-component platform where
models, conditioners, and compressors are first-class abstractions.

Supports **Stable Audio** diffusion models (Stability AI),
**[ACE-Step](https://github.com/ace-step/ACE-Step-1.5)** music-generation models (ACE Studio /
StepFun), and **MLX-accelerated Stable Audio** models on Apple Silicon — all through a unified
registry, CLI, and Gradio UI.

---

## What's New in Anvil

- **Pluggable pipeline architecture** — `BasePipeline`, `BaseGenerator`, `BaseCompressor`, `BaseConditioner` ABCs; swap any component without touching the rest of your workflow.
- **Named model registry** — `anvil generate --model stable-audio-open-1.0 --prompt "..."` loads the right pipeline automatically; add your own entries in `~/.anvil-audio/registry.yaml`.
- **ACE-Step support** — optional integration with [ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) for full-song music generation with lyrics and style tags. Optionally installed by `install.sh` — no manual repo clone required.
- **MLX acceleration** — Apple Silicon users can install `mlx-audiogen` to get native Metal GPU inference for Stable Audio models (~2x faster than PyTorch MPS); weights are auto-converted and cached on first use.
- **Output management** — collision-free timestamped filenames, JSON metadata sidecars with `generation_duration_seconds`, batch manifests, and project-scoped folders under `~/anvil-audio-outputs/`.
- **MPS / CUDA / CPU auto-detection** — runs on Apple Silicon, NVIDIA GPUs, or CPU with no flags needed.
- **`anvil generate` CLI** — multi-GPU via Accelerate, wav/flac/mp3/ogg output, batch YAML conditions, per-run seed control; `anvil --list-models` works at the top level.
- **Gradio web UI** — project name, seed input, live metadata panel, model
  dropdown with hot-reload, device field, and runtime theme presets.
- **Built-in audio editor** — post-processing tab with normalize, trim, fade, time stretch, pitch shift, EQ, and reverb; non-destructive exports with full effects sidecar.
- **Local prompt intelligence** — optional MLX Llama prompt enhancement,
  negative-prompt suggestions, and duration-aware lyric writing on Apple
  Silicon.
- **Automated training datasets** — `anvil dataset` can build local or
  YouTube-sourced clip datasets, write captions, generate
  `character_sheet.json`, and emit a training-ready `dataset_config.json`.
- **ACE-Step LoRA workflows** — import PEFT/LoKr adapters, apply them during
  generation, and wrap ACE-Step's corrected `training_v2` preprocess/train
  flow from the `anvil lora` CLI.
- **MCP server** — expose all generation and editing capabilities to Claude and other MCP clients over stdio; models are cached between calls.
- **Python 3.12 / 3.13** — uses modern union syntax, `slots=True` dataclasses, and lowercase generics throughout.

---

## Requirements

- **Python 3.13** — recommended (3.12 also supported)
- Python 3.14+ is too new for several ML dependencies and will cause build failures
- PyTorch 2.0 or later

---

## Install

> **Python 3.13 is recommended.** Check with `python3 --version`.
> Install via `brew install python@3.13` (macOS) or from [python.org](https://python.org) (Windows).

### macOS / Linux

```bash
git clone https://github.com/PiMPStudios/anvil-audio-v2.git
cd anvil-audio-v2
python3.13 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The script detects your platform, installs the right PyTorch build, adds
`pytest` for local verification, installs `yt-dlp` for dataset building,
enables MLX acceleration, local prompt intelligence, and the lightweight MLX
Whisper runtime on Apple Silicon, and optionally installs ACE-Step for music
generation — all in one step. Whisper model weights are still downloaded lazily
the first time transcription is used. No separate repo clones required.

### Windows

```powershell
git clone https://github.com/PiMPStudios/anvil-audio-v2.git
cd anvil-audio-v2
python -m venv .venv
.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Manual install (advanced)

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install .                      # core install
# Apple Silicon only: MLX acceleration + local intelligence
pip install mlx-audiogen mlx-lm
pip install yt-dlp                 # optional: YouTube dataset builder
pip install lightning-whisper-mlx  # Apple Silicon vocal transcription runtime
# or: pip install openai-whisper   # optional: cross-platform local transcription
pip install 'anvil-audio[acestep]' # optional: ACE-Step music generation
pip install pytest                 # optional: run the local test suite
```

### Verify your setup

```bash
anvil setup
.venv/bin/python -m pytest
```

`anvil setup` prints your platform, which optional packages are active, and what models are registered. The `pytest` command runs the local smoke tests.

---

## Documentation

- [Dataset and LoRA workflow](docs/datasets.md) covers local/YouTube dataset
  creation, vocal transcription, Qwen embedding QA, ACE-Step preprocessing, and
  training.
- [Stable Audio Open notes](docs/Stable_Audio_Open.md) cover the inherited
  Stable Audio Open 1.0 generation scripts and HuggingFace access notes.
- [Model internals](docs/diffusion.md), [conditioning](docs/conditioning.md),
  [autoencoders](docs/autoencoders.md), and [pretransforms](docs/pretransforms.md)
  are inherited Stable Audio technical references for people changing model
  architecture or training code.
- [Third-party notices](THIRD_PARTY_NOTICES.md) lists upstream code and model
  dependencies.

---

## Quick Start

The fastest path to generating audio:

```bash
# 1. Clone and install (see Install above)
# 2. Choose a built-in model, or add your own in ~/.anvil-audio/registry.yaml
# 3. Launch the Gradio UI — loads your selected or first registered model
python run_gradio.py
```

Or from the CLI in one command:

```bash
anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"
```

---

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

## ACE-Step Music Generation (optional)

[ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) is an open-source full-song music generation
model that supports style tags and full lyric input. Anvil integrates it through the same
registry and UI as Stable Audio — no separate server or app required, and no manual repo clone needed.

ACE-Step is **optional**. If you don't install it, all other Anvil functionality works as normal.

### LM thinking

ACE-Step ships a separate 5 Hz LM that produces structured `audio_codes` fed
into the DiT. Anvil can initialise that LM automatically, but the built-in SFT
entry defaults to the direct DiT conditioning path because that is the
known-good local baseline for blank-lyrics SFT generation.

Anvil initialises the LM automatically using the checkpoint specified in the registry
entry. The built-in entries default to:

| Model | LM checkpoint |
|-------|---------------|
| `acestep-v1.5-turbo` | `acestep-5Hz-lm-1.7B` (lighter, faster) |
| `acestep-v1.5-sft` | `acestep-5Hz-lm-4B` (heavier, better quality) |

Both checkpoints are downloaded automatically from HuggingFace on first use.

You can override the LM checkpoint for a specific registry entry via `lm_model_path` in
`registry.yaml` (see [ACE-Step models](#ace-step-models) below), or set a global fallback
for all ACE-Step models via an environment variable:

```bash
export ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
```

If the LM fails to initialise (missing checkpoint, memory constraints, etc.) Anvil
falls back gracefully to DiT-only generation and prints a warning. You can enable
LM thinking explicitly with registry defaults or per-call overrides when you want
ACE-Step to generate semantic code hints and metadata.

### Apple Silicon acceleration

On macOS, Anvil automatically sets `ACESTEP_LM_BACKEND=mlx` before initializing ACE-Step,
enabling the MLX backend for the 5 Hz LM (unless you've already set the variable yourself).
The DiT and VAE also run via native MLX on Apple Silicon automatically.
To compare against the non-MLX DiT/VAE backend, launch with:

```bash
ANVIL_ACESTEP_USE_MLX_DIT=0 python run_gradio.py
```

If you run the ACE-Step MCP server or API separately, set the variable in your shell or MCP
`env` block:

```bash
export ACESTEP_LM_BACKEND=mlx
```

### Built-in registry entries

Anvil registers two ACE-Step variants automatically and stores their downloaded
weights under `~/.cache/anvil-audio/acestep/checkpoints`:

| Model name | Description | Steps | Notes |
|---|---|---|---|
| `acestep-v1.5-turbo` | Fast generation | 8 | Good for drafts and quick iteration |
| `acestep-v1.5-sft` | Full quality | 50 | Direct DiT conditioning, guidance 7.5, shift 3.0, experimental DCW off |

The SFT built-in intentionally disables ACE-Step's experimental DCW sampler
correction and LM thinking by default. Keep those options disabled for the
normal SFT path; enable them explicitly only when you want to experiment with
ACE-Step's LM/code-hint path.

### Optional XL checkpoints

ACE-Step v1.5 XL DiT checkpoints (`acestep-v15-xl-base`,
`acestep-v15-xl-sft`, and `acestep-v15-xl-turbo`) are large optional downloads.
Anvil does **not** register or auto-download them by default. If you add an XL
entry to `~/.anvil-audio/registry.yaml` and the checkpoint is missing, Anvil
stops before ACE-Step can start a surprise download.

Install an XL checkpoint explicitly before using it:

```bash
acestep-download \
    --dir "$HOME/.cache/anvil-audio/acestep/checkpoints" \
    --model acestep-v15-xl-turbo
```

Use one of these model names with `--model`: `acestep-v15-xl-turbo`,
`acestep-v15-xl-sft`, or `acestep-v15-xl-base`. Avoid `acestep-download --all`
unless you intentionally want every optional ACE-Step submodel.

Then register the XL model in your user registry:

```yaml
- name: acestep-v1.5-xl-turbo
  pipeline_type: acestep
  model_config_path: acestep-v15-xl-turbo
  lm_model_path: acestep-5Hz-lm-1.7B
  max_duration: 600
  default_params:
    steps: 8
    cfg_scale: 1.0
    audio_duration: 60
    sampler_type: ode
    shift: 3.0
    thinking: true
    use_cot_metas: true
    use_cot_caption: false
    use_cot_language: true
    dcw_enabled: false
    velocity_norm_threshold: 0.0
    velocity_ema_factor: 0.0
    sigma_min: 0.0
    sigma_max: 0.0
    # Optional load-time memory controls for large XL checkpoints:
    # offload_to_cpu: true
    # offload_dit_to_cpu: true
    # quantization: int8_weight_only
```

For XL SFT, use the SFT checkpoint name and the direct SFT defaults:

```yaml
- name: acestep-v1.5-xl-sft
  pipeline_type: acestep
  model_config_path: acestep-v15-xl-sft
  max_duration: 600
  default_params:
    steps: 50
    cfg_scale: 7.5
    audio_duration: 60
    sampler_type: ode
    shift: 3.0
    thinking: false
    use_cot_metas: false
    use_cot_caption: false
    use_cot_language: false
    dcw_enabled: false
    velocity_norm_threshold: 0.0
    velocity_ema_factor: 0.0
    sigma_min: 0.0
    sigma_max: 0.0
```

### CLI

```bash
# Single prompt (instrumental)
anvil generate --model acestep-v1.5-turbo \
    --prompt "indie pop, acoustic guitar, warm vocals, upbeat" \
    --negative-prompt "muddy mix, harsh clipping, distorted vocals"

# Batch generation with lyrics (see example file)
anvil generate --model acestep-v1.5-turbo \
    --cond-yaml-path example/generation/acestep_conditions.yaml \
    --output-dir ./out

# Generate with a registered LoRA adapter
anvil generate --model acestep-v1.5-sft \
    --prompt "my_style, anthemic rock, live drums, gritty guitars" \
    --lora my-style \
    --lora-scale 0.8 \
    --seconds-total 60
```

Batch YAML format — each entry supports `prompt`, `negative_prompt`, `lyrics`,
and `seconds_total`:

```yaml
tracks:
  indie_pop:
    prompt: "indie pop, acoustic guitar, warm vocals, upbeat, sunny afternoon"
    lyrics: |
      [verse]
      Walking down the open road
      Sunlight through the trees
      [chorus]
      This is where I start again
    seconds_total: 30.0

  electronic_instrumental:
    prompt: "electronic, synthwave, driving bass, retro 80s, cinematic"
    negative_prompt: "muddy low end, harsh treble, clipping"
    lyrics: "[Instrumental]"
    seconds_total: 45.0
```

### Gradio web UI

```bash
# Turbo (fast, good for iteration)
python run_gradio.py --model acestep-v1.5-turbo

# Full quality
python run_gradio.py --model acestep-v1.5-sft
```

The ACE-Step UI adds a **Lyrics** field below the prompt. Leave it blank or enter
`[Instrumental]` for tracks with no vocals. Structure lyrics with section markers
like `[verse]`, `[chorus]`, `[bridge]`.

The **Negative prompt** field is available for every model. For ACE-Step, it is
passed to ACE-Step's LM/thinking path as `lm_negative_prompt`; direct DiT-only
SFT generation keeps it in metadata, but upstream ACE-Step does not expose a
separate DiT negative-prompt control.

### Local prompt intelligence

On Apple Silicon, Anvil can use a local MLX Llama model to enhance prompts,
suggest negative prompts, and write duration-aware lyrics. The default model is
`mlx-community/Llama-3.2-3B-Instruct-4bit`. First use downloads the model into
Anvil Audio's local cache:

```text
~/.cache/anvil-audio/llm/llama-3.2-3b-instruct-4bit/
```

You can override the model with `ANVIL_LLM_MODEL`, `ANVIL_LLM_MODEL_PATH`, or
the CLI `--model` / `--intelligence-model` flags.

CLI helper:

```bash
anvil enhance-prompt \
    --prompt "anthemic alternative rock with emotional male vocal" \
    --duration 60
```

Generate with automatic prompt enhancement and lyric writing:

```bash
anvil generate --model acestep-v1.5-sft \
    --prompt "anthemic alternative rock with emotional male vocal" \
    --seconds-total 60 \
    --enhance-prompt \
    --write-lyrics
```

In Gradio, use **Enhance Prompt**, **Write Lyrics**, or **Enhance + Lyrics**
above the generation controls.

---

## Automated Training Datasets

`anvil dataset` prepares reviewable clip datasets for future LoRA training. It
can split local audio or authorized YouTube sources into fixed-length WAV clips,
write JSON sidecars with prompts, create `captions.json`, summarize the set into
`character_sheet.json`, and emit a `dataset_config.json` compatible with the
existing `audio_dir` training loader.

Use this only with material you own or are authorized to train on.

Build from local audio:

```bash
anvil dataset build-local ./my-source-audio \
    --name my_band_style \
    --clips 80 \
    --clip-length 35 \
    --style-hint "anthemic alternative rock, live drums, gritty guitars" \
    --caption-mode heuristic
```

Build from a YouTube channel, playlist, or video:

```bash
anvil dataset build-youtube "https://www.youtube.com/playlist?list=..." \
    --name my_channel_style \
    --tracks 12 \
    --clips 80 \
    --clip-length 35 \
    --style-hint "cinematic alternative rock, emotional male vocal" \
    --caption-mode llm
```

`--caption-mode heuristic` is fast and deterministic. `--caption-mode llm` uses
the same local MLX Llama intelligence model as prompt enhancement to polish each
caption and the final character sheet. Both modes keep the generated metadata
editable so bad clips or captions can be removed before training.

For vocal datasets, add local Whisper transcription hints:

```bash
anvil dataset build-local ./my-source-audio \
    --name my_vocal_style \
    --clips 80 \
    --clip-length 35 \
    --style-hint "dark blues, smoky male vocal, raw guitar" \
    --caption-mode llm \
    --transcribe-vocals
```

`--transcribe-vocals` only runs on clips whose source title or style hint looks
vocal-focused. Use `--transcribe-all` when the source metadata is weak and you
want every clip checked. This is local-only. On Apple Silicon, `install.sh`
installs the lightweight `lightning-whisper-mlx` runtime and the selected
Whisper model downloads lazily on first use. Cross-platform manual installs can
use `openai-whisper` instead.

Before training, run the embedding QA pass to find duplicate captions, semantic
outliers, weak coverage, and low-confidence clips:

```bash
anvil dataset qa ./datasets/my_channel_style_YYYYMMDD_HHMMSS
```

This uses the local `Qwen3-Embedding-0.6B` checkpoint from the Anvil ACE-Step
cache when present, otherwise `Qwen/Qwen3-Embedding-0.6B` from HuggingFace. It
writes `dataset_qa_report.json` and `dataset_qa_report.md` in the dataset
folder so you can prune or recaption clips before LoRA training.

Output layout:

```text
datasets/my_channel_style_YYYYMMDD_HHMMSS/
  clips/
    clip_0001.wav
    clip_0001.json
  captions.json
  character_sheet.json
  dataset_manifest.json
  dataset_config.json
  sources/
```

The per-clip JSON sidecars contain the training `prompt`, source metadata, basic
audio analysis, tags, negative tags, and confidence. The generated
`dataset_config.json` can be passed to training code that consumes
`audio_dir` datasets.

---

## ACE-Step LoRA Adapters

Anvil can register and apply ACE-Step LoRA adapters without bundling the
adapter weights into this repo. Local adapter metadata lives at:

```text
~/.cache/anvil-audio/lora/adapters/
```

Import a PEFT LoRA directory or LoKr safetensors file:

```bash
anvil lora import-local ./lora-runs/my_style/final --name my-style
```

Import from HuggingFace:

```bash
anvil lora import-hf username/my-style-acestep-lora --name my-style
```

List and inspect registered adapters:

```bash
anvil lora list
anvil lora info my-style
```

Use an adapter from the CLI:

```bash
anvil generate --model acestep-v1.5-sft \
    --prompt "my_style, polished alternative rock, live drums" \
    --lora my-style \
    --lora-scale 0.75 \
    --seconds-total 60
```

Use an adapter in Gradio by loading an ACE-Step model, opening the
**ACE-Step LoRA** accordion, and entering a registered adapter id/name or a
direct PEFT/LoKr path.

Anvil does not scan other applications for adapters. Keep this repo's LoRA path
explicit: train an adapter here, import a local PEFT/LoKr folder, or import a
standard HuggingFace adapter repo when you have one you want to use.

### Training a LoRA

The full automated flow is:

```bash
# 1. Build reviewable clips and captions
anvil dataset build-youtube "https://www.youtube.com/playlist?list=..." \
    --name my_style \
    --tracks 12 \
    --clips 80 \
    --clip-length 35 \
    --style-hint "anthemic alternative rock, live drums" \
    --caption-mode llm

# 2. Review duplicate captions, semantic outliers, and low-confidence clips
anvil dataset qa ./datasets/my_style_YYYYMMDD_HHMMSS

# 3. Convert clips into ACE-Step training tensors
anvil lora preprocess ./datasets/my_style_YYYYMMDD_HHMMSS \
    --output-dir ./tensors/my_style \
    --model-variant sft \
    --precision fp32 \
    --custom-tag my_style

# 4. Train with ACE-Step's corrected training_v2 fixed LoRA trainer
anvil lora train ./tensors/my_style \
    --output-dir ./lora-runs/my_style \
    --model-variant sft \
    --epochs 20 \
    --rank 64 \
    --alpha 128

# 5. Register the final adapter for generation
anvil lora import-local ./lora-runs/my_style/final --name my-style
```

`anvil lora train` writes an inference-ready PEFT adapter under
`<output-dir>/final/`, which ACE-Step can load directly.

On Apple Silicon, add `--basic-loop` if Lightning Fabric fails with MPS AMP
gradient-scaler errors. This uses ACE-Step's own non-Fabric training loop.
Keep preprocessing at `--precision fp32`; lower precision can produce non-finite
conditioning tensors on the Apple path.

---

## Troubleshooting

`anvil setup` is the first thing to run when something looks off. It reports
whether ACE-Step, MLX Stable Audio, local prompt intelligence, and MLX vocal
transcription are importable in the current virtual environment.

Common startup warnings:

- `bitsandbytes not installed. Using standard AdamW.` is expected on macOS and
  only affects optimizer selection for training.
- `torchao` compatibility warnings can be ignored unless you are actively using
  ACE-Step quantization.
- `mx.metal.device_info is deprecated` is an upstream MLX warning and does not
  indicate failed generation.

If an XL model refuses to load, install the checkpoint explicitly with
`acestep-download --dir "$HOME/.cache/anvil-audio/acestep/checkpoints" --model <checkpoint>`.
Anvil blocks surprise XL downloads because those checkpoints are large.

---

## Audio Editor

The Gradio UI includes a built-in **Edit** tab for quick post-processing without
leaving Anvil. It works with any loaded model and any audio file — not just
files generated in the current session.

Available tools:

| Tool | What it does |
|---|---|
| Normalize | Peak or LUFS loudness targeting |
| Trim silence | Strips quiet sections from the edges |
| Fade in / fade out | Linear ramp from/to silence |
| Loop / clip | Trim the audio to a start/end range |
| Time stretch | Speed up or slow down without changing pitch |
| Pitch shift | Transpose up or down in semitones |
| EQ | Low shelf, peak mid, high shelf |
| Reverb | Room size, damping, wet/dry mix |

Effects are applied in a fixed chain (trim → clip → stretch/pitch → EQ →
reverb → fade → normalize), which keeps results predictable regardless of the
order you adjust knobs.

**Typical workflow:**

1. Generate on the **Generate** tab
2. Switch to **Edit**
3. Click **Load Last Generation** — the output loads automatically
4. Adjust effects; click **Preview** to hear the result
5. Click **Export** when satisfied

Export creates a new file via the output manager — the original is never
touched. The JSON sidecar for the exported file records the source path and
the full effects chain so you can always trace what was applied and replay it.

You can also drag any audio file into the source field to edit files from
outside Anvil.

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

## MCP Server

Anvil exposes its full capabilities as an [MCP](https://modelcontextprotocol.io) server so
Claude and other MCP clients can generate and edit audio directly without the Gradio UI or
manual CLI commands.

### Install

```bash
pip install mcp
```

The `mcp` package is not installed by default. Everything else is already a dependency.

### Available tools

| Tool | What it does |
|---|---|
| `generate_audio` | Generate a clip from a prompt; auto-selects model if not specified |
| `batch_generate` | Generate multiple clips in one call |
| `edit_audio` | Post-process a file with normalize, trim, EQ, reverb, etc. |
| `list_models` | All registered models with type, limits, and loaded status |
| `get_model_info` | Full details for one model |
| `list_recent_outputs` | Recent output files with their metadata, newest-first |
| `get_generation_metadata` | Read the sidecar for any output file |
| `list_projects` | Project folders under `~/anvil-audio-outputs/` |
| `set_active_project` | Set a default project so you don't repeat it every call |

All `generate_audio` and `batch_generate` responses include `generation_duration_seconds` —
the wall-clock time from the start of inference to the file being written. This lets you
compare backends directly (e.g. PyTorch MPS vs MLX) without any external timing.

Models are loaded lazily on first use and cached between calls — switching between
two models during a session only pays the load cost once per model.

### Claude Desktop config

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`
(create the file if it doesn't exist):

```json
{
  "mcpServers": {
    "anvil-audio": {
      "command": "/path/to/anvil-audio-v2/.venv/bin/python",
      "args": ["-m", "anvil_audio.mcp_server"]
    }
  }
}
```

Replace `/path/to/anvil-audio-v2` with the absolute path to your clone.

### Claude Code config

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "anvil-audio": {
      "command": "/path/to/anvil-audio-v2/.venv/bin/python",
      "args": ["-m", "anvil_audio.mcp_server"],
      "type": "stdio"
    }
  }
}
```

### Example session

Once configured, Claude can generate and edit audio directly:

```text
You:    Generate a short thunderstorm ambience clip
Claude: [calls generate_audio(prompt="thunderstorm ambience, rain, distant thunder", duration_seconds=20)]
        Generated: ~/anvil-audio-outputs/default/20260401_181907_thunderstorm_...wav
        Generation time: 31.2 s

You:    Add a slight fade in and normalize it to -14 LUFS
Claude: [calls edit_audio(file_path="...", fade_in=2.0, normalize=True,
                          normalize_target_db=-14, normalize_lufs=True)]
        Exported: ~/anvil-audio-outputs/default/20260401_181942_edit_...wav
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

## `run_gradio.py` flags

| Flag | Description |
|------|-------------|
| `--model` | Registry model name (e.g. `stable-audio-open-1.0`, `acestep-v1.5-turbo`) |
| `--pretrained-name` | HuggingFace Hub repo ID (e.g. `stabilityai/stable-audio-open-1.0`) |
| `--model-config` | Local model config JSON (ignored if `--model` or `--pretrained-name` set) |
| `--ckpt-path` | Local checkpoint (ignored if `--model` or `--pretrained-name` set) |
| `--pretransform-ckpt-path` | Optional separate VAE checkpoint |
| `--username` / `--password` | Gradio auth |
| `--model-half` | Use float16 inference |
| `--device` | `cuda`, `mps`, or `cpu` (auto-detects if omitted) |
| `--project` | Outputs go to `~/anvil-audio-outputs/{project}/` |
| `--share` | Create a public Gradio share URL |

---

## `anvil generate` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model NAME` | — | Registry model name |
| `--list-models` | — | Print registry and exit (also works as `anvil --list-models`) |
| `--model-config PATH` | — | Legacy: local JSON config |
| `--ckpt-path PATH` | — | Legacy: local checkpoint |
| `--pretransform-ckpt-path PATH` | — | Separate VAE checkpoint |
| `--prompt TEXT` | — | Single text prompt |
| `--cond-yaml-path PATH` | — | Batch YAML conditions file |
| `--seconds-start` | `0.0` | Start time (seconds) |
| `--seconds-total` | `30.0` | Duration (seconds) |
| `--negative-prompt` | blank | Text describing sounds or qualities to avoid |
| `--enhance-prompt` | off | Enhance prompt and negative prompt |
| `--write-lyrics` | off | Write duration-aware ACE-Step lyrics |
| `--intelligence-model` | default | LLM path or HuggingFace repo |
| `--lora` | blank | ACE-Step adapter id/name or direct PEFT/LoKr path |
| `--lora-scale` | `1.0` | ACE-Step adapter strength |
| `--lora-adapter-name` | blank | Optional runtime adapter name |
| `--output-dir` | `./output` | Output directory |
| `--format` | `wav` | `wav`, `flac`, `mp3`, or `ogg` |
| `--clip-length` | off | Clip to `seconds_total` |
| `--sample-steps` | pipeline default | Diffusion / inference steps |
| `--cfg-scale` | pipeline default | CFG guidance scale |
| `--sampler-type` | pipeline default | Sampler type |
| `--sigma-min` / `--sigma-max` | pipeline default | Noise schedule bounds |
| `--n-sample-per-cond` | `1` | Samples per condition |
| `--batch-size` | `10` | Items per GPU batch |
| `--seed` | `-1` (random) | RNG seed |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

---

## `anvil dataset` flags

| Command / Flag | Default | Description |
| --- | --- | --- |
| `build-local SOURCE_DIR` | - | Build from a folder of audio files |
| `build-youtube URL` | - | Download authorized YouTube audio with `yt-dlp` |
| `qa DATASET_DIR` | - | Run Qwen embedding QA on captions |
| `--name` | `anvil_dataset` | Dataset name in manifests |
| `--output-dir` | timestamped `./datasets/...` | Output dataset directory |
| `--clips` | `40` | Maximum clips to write |
| `--clip-length` | `35` | Clip length in seconds |
| `--min-clip-length` | `8` | Skip source files shorter than this |
| `--stride` | clip length | Seconds between clip starts |
| `--sample-rate` | `48000` | Output sample rate |
| `--channels` | `2` | Output channel count |
| `--style-hint` | blank | Style context added to captions |
| `--caption-mode` | `heuristic` | `heuristic`, `llm`, or `off` |
| `--llm-model` | default | LLM path/repo for caption cleanup |
| `--transcribe-vocals` | off | Add local Whisper hints to likely vocal clips |
| `--transcribe-all` | off | Transcribe every clip |
| `--transcription-backend` | `auto` | `lightning-whisper-mlx` or `whisper` |
| `--transcription-model` | backend default | Whisper model name |
| `--transcription-language` | auto | Optional source language code |
| `--tracks` | unlimited | YouTube-only max source videos/tracks |
| `--delete-downloads` | off | Delete raw downloads after clips are written |
| `--quiet-ytdlp` | off | Pass `--quiet` to `yt-dlp` |
| `--embedding-model` | local Qwen cache | QA-only embedding model path/repo |
| `--duplicate-threshold` | `0.9` | QA-only duplicate similarity cutoff |
| `--cluster-threshold` | `0.78` | QA-only cluster similarity cutoff |
| `--outlier-threshold` | `0.55` | QA-only outlier neighbor cutoff |

---

## `anvil lora` flags

| Command / Flag | Default | Description |
| --- | --- | --- |
| `list` | - | List registered adapters |
| `info REF` | - | Show adapter metadata or resolve a direct path |
| `import-local PATH` | - | Register a PEFT adapter dir or LoKr safetensors |
| `import-hf REPO_ID` | - | Download and register a HuggingFace adapter |
| `write-dataset-json DATASET_DIR` | - | Convert an Anvil dataset to ACE-Step JSON |
| `preprocess DATASET_DIR` | - | Build ACE-Step `.pt` tensors |
| `train TENSOR_DIR` | - | Run ACE-Step corrected LoRA training |
| `--name` | inferred | Adapter display name |
| `--base-model` | `acestep-v1.5` | Compatibility note for adapter metadata |
| `--checkpoint-dir` | Anvil ACE-Step cache | ACE-Step checkpoints root |
| `--model-variant` | `sft` | `turbo`, `base`, `sft`, or custom folder name |
| `--precision` | `fp32` for preprocess | Preprocess/train precision |
| `--custom-tag` | blank | Trigger tag prepended during preprocessing |
| `--output-dir` | required | Tensor or training output directory |
| `--epochs` | `100` | Training epochs |
| `--rank` / `--alpha` | `64` / `128` | LoRA rank and alpha |
| `--basic-loop` | off | Use ACE-Step's non-Fabric loop for MPS AMP issues |
| `--dry-run` | off | Print the ACE-Step training command |

---

## Container Setup

Build a Docker image and optionally convert to Singularity for HPC clusters:

```bash
NAME=anvil-audio
docker build -t ${NAME} -f ./container/anvil-audio.Dockerfile .

# Convert to Singularity
singularity build anvil-audio.sif docker-daemon://anvil-audio
```

---

## Backlog

- [ ] PyPI package (`pip install anvil-audio`)
- [ ] Contribution guidelines
- [ ] More audio augmentations

---

## Licensing

Anvil Audio is MIT licensed. It builds on several open-source projects and optional model
weights with their own licenses — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
full attributions.
