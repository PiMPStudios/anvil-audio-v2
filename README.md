# Anvil Audio

> **A pluggable studio tool for AI audio generation. Swap models, keep your workflow.**

Anvil Audio is a refactored and extended fork of
[`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) by Stability AI.
It turns a single-model inference codebase into a clean, swappable-component platform where
models, conditioners, and compressors are first-class abstractions.

Supports **Stable Audio** diffusion models, **[ACE-Step](https://github.com/ace-step/ACE-Step-1.5)**
music-generation models with lyrics and LoRA adapters, and **MLX-accelerated Stable Audio**
models on Apple Silicon through one registry, CLI, Gradio UI, and MCP server.

---

## What's New in Anvil

- **Pluggable pipeline architecture** - `BasePipeline`, `BaseGenerator`, `BaseCompressor`, and `BaseConditioner` abstractions.
- **Named model registry** - use built-ins or add your own entries in `~/.anvil-audio/registry.yaml`.
- **ACE-Step support** - optional full-song generation with lyrics, XL checkpoints, and LoRA adapter workflows.
- **MLX acceleration** - native Metal inference for Stable Audio on Apple Silicon, with auto-converted cached weights.
- **Local prompt intelligence** - MLX Llama prompt enhancement, negative prompts, and duration-aware lyrics on Apple Silicon.
- **Automated datasets and LoRA training** - build local/YouTube datasets, run QA, separate stems, package cloud jobs, and train adapters.
- **Gradio studio UI** - generation, inpainting, editing, projects, metadata, model hot-reload, themes, and LoRA controls.
- **Audio editor** - normalize, trim, fade, stretch, pitch shift, EQ, reverb, and non-destructive exports.
- **MCP server** - generate and edit audio directly from Claude and other MCP clients, with model cache controls.
- **Output management** - timestamped files, JSON sidecars, batch manifests, and project folders under `~/anvil-audio-outputs/`.

---

## Requirements

- **Python 3.13** recommended; Python 3.12 also supported
- Python 3.14+ is too new for several ML dependencies
- PyTorch 2.0 or later

---

## Install

### macOS / Linux

```bash
git clone https://github.com/PiMPStudios/anvil-audio-v2.git
cd anvil-audio-v2
python3.13 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer detects your platform, installs the right PyTorch build, adds local test tooling,
installs `yt-dlp` for dataset building, enables MLX acceleration and prompt intelligence on Apple
Silicon, and optionally installs ACE-Step for music generation. No separate ACE-Step repo clone is
required.

### Windows

```powershell
git clone https://github.com/PiMPStudios/anvil-audio-v2.git
cd anvil-audio-v2
python -m venv .venv
.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Verify

```bash
anvil setup
.venv/bin/python -m pytest
```

`anvil setup` reports which optional runtimes are available: ACE-Step, MLX Stable Audio, prompt
intelligence, MCP, and local transcription.

See [Installation](docs/install.md) for manual dependency installation and
advanced setup notes.

---

## Quick Start

Launch the Gradio studio:

```bash
python run_gradio.py
```

Generate from the CLI:

```bash
anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"
```

Generate music with ACE-Step:

```bash
anvil generate --model acestep-v1.5-sft \
    --prompt "dark blues noir rock, raw guitar, smoky male vocal" \
    --lyrics "[Verse 1]\nRain on the window\n[Chorus]\nCarry me home" \
    --seconds-total 60
```

Use a registered LoRA adapter:

```bash
anvil generate --model acestep-v1.5-xl-sft \
    --prompt "dark blues noir rock, raw guitar, cinematic atmosphere" \
    --lora dark-blues-xl-sft \
    --lora-scale 0.8 \
    --seconds-total 60
```

---

## Core Workflows

| Workflow | Start here |
|---|---|
| Installation and manual dependency setup | [Installation](docs/install.md) |
| Stable Audio and custom registry models | [Stable Audio, MLX, and Model Registry](docs/stable-audio-and-mlx.md) |
| ACE-Step music generation, XL checkpoints, LoRA adapters | [ACE-Step Music and LoRA](docs/ace-step.md) |
| Dataset creation, captions, source separation, QA | [Datasets](docs/datasets.md) and [Dataset and Training Workflows](docs/training-workflows.md) |
| Cloud LoRA training and model-training notes | [Cloud training and model notes](docs/cloud-training-and-model-notes.md) |
| MCP setup and tool reference | [MCP Server](docs/mcp.md) |
| Audio editing tools | [Audio Editor](docs/audio-editor.md) |
| CLI flags | [CLI Reference](docs/cli-reference.md) |
| Common warnings and setup checks | [Troubleshooting](docs/troubleshooting.md) |

Additional planning docs:

- [Dataset separation plan](docs/dataset-separation-plan.md)
- [Gradio Training Studio plan](docs/gradio-training-studio-plan.md)

Inherited Stable Audio technical references:

- [Stable Audio Open](docs/Stable_Audio_Open.md)
- [Diffusion](docs/diffusion.md)
- [Conditioning](docs/conditioning.md)
- [Autoencoders](docs/autoencoders.md)
- [Pretransforms](docs/pretransforms.md)

---

## MCP Server

The MCP runtime is included in the default install:

```bash
.venv/bin/python -m anvil_audio.mcp_server
```

ACE-Step PEFT LoRAs can run through the native MLX DiT path on Apple Silicon.
Use `--no-mlx-dit` only when you specifically need the PyTorch backend, such as
for LoKr/LyCORIS adapters or backend comparisons:

```bash
.venv/bin/python -m anvil_audio.mcp_server --no-mlx-dit
```

MCP tools include generation, batch generation, editing, LoRA listing, model inspection, project
management, memory status, and model unloading. See [MCP Server](docs/mcp.md) for Claude Desktop and
Claude Code config examples.

---

## Output Layout

Generated and edited files are written under:

```text
~/anvil-audio-outputs/{project}/
```

Every output includes a JSON sidecar with prompt, model, seed, generation time, edit chain, LoRA
metadata, and any workflow-specific extras. These sidecars can be loaded back into the Gradio UI as
presets.

---

## Backlog

- [ ] PyPI package (`pip install anvil-audio`)
- [ ] Contribution guidelines
- [ ] More audio augmentations
- [ ] Screenshot gallery once project screenshots are ready

---

## Licensing

Anvil Audio is MIT licensed. It builds on several open-source projects and optional model weights
with their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attributions.
