# Documentation Guide

Start with the root [README](../README.md) for install, quick start, and the high-level workflow map.

Use these docs when you want more detail:

- [Stable Audio, MLX, and Model Registry](stable-audio-and-mlx.md) - Stable Audio generation,
  Apple Silicon MLX acceleration, and custom `~/.anvil-audio/registry.yaml` entries.
- [Installation](install.md) - platform install commands, manual dependency install, and setup
  verification.
- [ACE-Step Music and LoRA](ace-step.md) - ACE-Step music generation, LM thinking, XL checkpoints,
  prompt intelligence, LoRA adapter usage, and adapter training commands.
- [Datasets](datasets.md) - current Anvil workflow for local/YouTube dataset creation, vocal
  transcription, Qwen embedding QA, ACE-Step LoRA preprocessing, and LoRA training.
- [Dataset and Training Workflows](training-workflows.md) - compact command-oriented dataset build
  reference moved out of the root README.
- [Cloud training and model notes](cloud-training-and-model-notes.md) - cloud job packaging, SSH
  runner notes, burst GPU training design, ACE-Step component fine-tuning, and future automation.
- [MCP Server](mcp.md) - MCP tools, Claude Desktop/Claude Code config, LoRA backend notes, and memory
  controls.
- [Audio Editor](audio-editor.md) - Gradio Edit tab tools and non-destructive export behavior.
- [CLI Reference](cli-reference.md) - Gradio, generation, dataset, cloud, LoRA, and container flags.
- [Troubleshooting](troubleshooting.md) - common warnings and setup checks.
- [Dataset separation plan](dataset-separation-plan.md) - branch-level plan for source separation,
  stem sidecars, stem-aware QA, and training-bundle exports.
- [Gradio Training Studio plan](gradio-training-studio-plan.md) - proposed UI plan for dataset review,
  source separation, cloud training, and adapter review inside Gradio.
- [Stable Audio Open](Stable_Audio_Open.md) - inherited Stable Audio Open 1.0 generation notes and
  HuggingFace access guidance.
- [Diffusion](diffusion.md), [conditioning](conditioning.md), [autoencoders](autoencoders.md), and
  [pretransforms](pretransforms.md) - inherited Stable Audio architecture references for deeper model
  work.

Most users only need the README, ACE-Step guide, MCP guide, and dataset guide. The architecture docs
are useful when changing model configs, training internals, or inherited Stable Audio components.
