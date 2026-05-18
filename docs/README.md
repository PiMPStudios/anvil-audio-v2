# Documentation Guide

Start with the root [README](../README.md) for install, model registry, Gradio,
ACE-Step generation, prompt intelligence, and command reference.

Use these docs when you want more detail:

- [Datasets](datasets.md) - current Anvil workflow for local/YouTube dataset
  creation, vocal transcription, Qwen embedding QA, ACE-Step LoRA preprocessing,
  and LoRA training.
- [Cloud training and model notes](cloud-training-and-model-notes.md) - cloud
  job packaging, SSH runner notes, burst GPU training design, ACE-Step component
  fine-tuning, and future training automation.
- [Dataset separation plan](dataset-separation-plan.md) - branch-level plan for
  source separation, stem sidecars, stem-aware QA, and training-bundle exports.
- [Stable Audio Open](Stable_Audio_Open.md) - inherited Stable Audio Open 1.0
  generation notes and HuggingFace access guidance.
- [Diffusion](diffusion.md), [conditioning](conditioning.md),
  [autoencoders](autoencoders.md), and [pretransforms](pretransforms.md) -
  inherited Stable Audio architecture references for deeper model work.

Most users only need the README plus the dataset guide. The architecture docs
are useful when changing model configs, training internals, or inherited Stable
Audio components.
