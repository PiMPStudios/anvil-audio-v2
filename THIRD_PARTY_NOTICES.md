# Third-Party Notices

Anvil Audio incorporates code and optionally depends on model weights from the following
projects. Each entry lists the project, its license, and how Anvil uses it.

---

## stable-audio-tools

**Project:** stable-audio-tools
**URL:** https://github.com/Stability-AI/stable-audio-tools
**Author:** Stability AI
**License:** MIT

Anvil Audio is a fork of stable-audio-tools. The core model architecture, training loop,
diffusion pipeline, and Gradio UI are derived from this codebase. Significant portions of
the original code remain in `anvil_audio/` in refactored form.

---

## friendly-stable-audio-tools

**Project:** friendly-stable-audio-tools
**URL:** https://github.com/yukara-ikemiya/friendly-stable-audio-tools
**Author:** Yukara Ikemiya
**License:** MIT

An earlier refactor of stable-audio-tools that Anvil builds on. Structural improvements
and abstractions introduced in this fork informed the pluggable pipeline architecture in
Anvil.

---

## ACE-Step

**Project:** ACE-Step
**URL:** https://github.com/ace-step/ACE-Step-1.5
**Author:** ACE Studio and StepFun
**License:** Apache 2.0

**Optional dependency.** The installer can install ACE-Step from its GitHub package; users
can also provide a local ACE-Step checkout explicitly. When present, Anvil wraps ACE-Step's
`AceStepHandler` through the `ACEStepPipeline` adapter (`anvil_audio/pipelines/acestep.py`),
integrating it into the Anvil registry, CLI batch generation, output manager, Gradio UI,
and LoRA workflow. ACE-Step source is not vendored into this repository.

---

## mlx-audiogen

**Project:** mlx-audiogen
**URL:** https://github.com/jasonvassallo/mlx-audio-generate
**Author:** Jason Vassallo
**License:** Apache 2.0

**Optional dependency.** `install.sh` installs it on Apple Silicon; manual installs can use
`pip install mlx-audiogen`. When present on Apple Silicon (M1/M2/M3/M4), Anvil uses
mlx-audiogen's `StableAudioPipeline` and `convert_stable_audio` via the `MLXDiffusionPipeline`
adapter
(`anvil_audio/pipelines/mlx_diffusion.py`) to run Stable Audio inference on Apple's native
MLX framework. Converted weights are cached locally — no mlx-audiogen source is bundled
with Anvil.

---

## Stable Audio Open model weights

**Project:** Stable Audio Open (1.0 and Small)
**URL:** https://huggingface.co/stabilityai/stable-audio-open-1.0
**Author:** Stability AI
**License:** [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE)

**Optional / downloaded on demand.** Model weights are not bundled with Anvil. They are
downloaded from HuggingFace Hub when requested via `--pretrained-name` or the built-in
registry entries. Use of these weights is governed by the Stability AI Community License,
not the MIT license that covers Anvil's code.

---

## ACE-Step model weights

**Project:** ACE-Step v1.5 checkpoints (turbo and SFT)
**URL:** https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B
**Author:** ACE Studio and StepFun
**License:** Apache 2.0

**Optional / downloaded on demand.** Model weights are not bundled with Anvil. They are
downloaded from HuggingFace Hub by ACE-Step's `initialize_service` call on first use.

---

## Local prompt and dataset intelligence models

**Projects:** mlx-lm, Llama 3.2 3B Instruct MLX quantizations, Qwen3-Embedding-0.6B,
and local Whisper runtimes such as lightning-whisper-mlx or openai-whisper.

**Optional / downloaded on demand.** These are used for prompt enhancement, lyric writing,
dataset caption cleanup, embedding QA, and optional vocal transcription. Runtime packages
are installed through pip extras or `install.sh`; model weights are cached locally on first
use and remain governed by their upstream model cards and licenses.
