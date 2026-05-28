# Cloud Training and Model Notes

This note captures the working design ideas around cloud LoRA training, dataset
preparation, source separation, and deeper ACE-Step training paths. It is not a
current feature contract. Treat it as the scratchpad for future Anvil work that
is too useful to leave buried in chat history.

## Goal

Anvil Audio should stay useful on a local Mac, but heavier training should be
able to burst onto rented GPU hardware when the local machine is too busy,
memory-constrained, or slow for repeated experiments.

The useful end state is:

```bash
anvil cloud train \
    --provider digitalocean \
    --gpu h200 \
    --dataset ~/Documents/Anvil/TrainingDatasets/dark-blues \
    --base-model acestep-v1.5-sft \
    --recipe lora-balanced \
    --max-hours 6 \
    --destroy-on-complete
```

That command should package the dataset, create a temporary GPU box, install the
training stack, run the job, collect adapters and logs, then destroy the box.

## Why Cloud GPUs

Apple Silicon is very capable for local generation and smaller experiments, but
LoRA training competes with the same machine used for everything else. A rented
GPU instance gives Anvil a clean training lane:

- keep the laptop responsive during long jobs
- iterate on multiple LoRA recipes in one session
- test SFT versus Turbo training without waiting overnight
- run validation generations on every checkpoint
- keep cloud cost bounded with hard runtime limits and auto-destroy

The first provider target should be a simple API-driven VM provider.
DigitalOcean GPU Droplets are worth watching because they expose normal SSH
machines, API/CLI creation, and H200 shapes, but availability can be limited.
RunPod is the first implemented API launcher. Other providers such as Lambda,
Vast, or CoreWeave can be considered later behind the same abstraction.

If DigitalOcean has the GPU capacity you need and you create an account from
this project, this referral link helps offset Anvil training costs:
[m.do.co/c/aa4e66218064](https://m.do.co/c/aa4e66218064).

## Cloud Runner Shape

A reliable cloud runner should be built from boring steps before it becomes a
single button.

```text
anvil cloud doctor
anvil cloud package
anvil cloud run-ssh
anvil cloud runpod launch
anvil cloud runpod terminate
```

The first implemented slice is provider-agnostic packaging plus an SSH runner.
That means Anvil can prepare a portable job folder locally, then run it on any
GPU host that exposes normal SSH. DigitalOcean, RunPod, Lambda, Vast, and other
providers can all plug in later because the provider-specific part is reduced to
"give me an SSH machine with a GPU."

```bash
anvil cloud doctor

anvil cloud search --gpu h200 --max-price 4 --min-vram-gb 80

anvil dataset export-training-bundle ~/Developer/TrainingDatasets/datasets/Dark_Bluesy \
    --include full-mix,instrumental

anvil cloud package \
    ~/Developer/TrainingDatasets/datasets/Dark_Bluesy/training_bundle.json \
    --output-dir ~/Developer/TrainingDatasets/cloud-jobs/dark-blues-h200 \
    --primary-asset instrumental \
    --model-variant sft \
    --recipe lora-balanced \
    --max-hours 6

export RUNPOD_API_KEY=...
anvil cloud runpod launch ~/Developer/TrainingDatasets/cloud-jobs/dark-blues-h200 \
    --gpu-type "NVIDIA H200" \
    --dry-run

anvil cloud run-ssh ~/Developer/TrainingDatasets/cloud-jobs/dark-blues-h200 \
    --host ubuntu@203.0.113.10 \
    --dry-run
```

`anvil cloud doctor` checks for local `ssh`, `rsync`, `bash`, and `git`.
`anvil cloud search` queries GPUFindr's read-only public catalog so users can
see whether suitable GPUs exist before creating provider accounts or entering
billing details. `anvil cloud runpod launch` prepares a RunPod
`podFindAndDeployOnDemand` GraphQL request and requires `RUNPOD_API_KEY` unless
it is run with `--dry-run`. If you create a RunPod account from this project,
the referral link
[runpod.io?ref=sox5p475](https://runpod.io?ref=sox5p475) helps offset Anvil
training costs. The default RunPod launch uses template
`runpod-torch-v280`, matching the deploy URLs returned by GPUFindr for RunPod
H200/H100/A100 offers. Add `--minimal` when RunPod's allocator rejects the
extra disk or minimum machine constraints even though the UI still shows
featured GPUs. For `--gpu-type`, use the `gpuId` value from
`runpodctl gpu list`; for example, H200 SXM is `NVIDIA H200`, and A100 SXM is
`NVIDIA A100-SXM4-80GB`. Remove `--dry-run` when the remote commands look
right. The runner uploads the job with `rsync`, runs `scripts/bootstrap.sh`,
then runs `scripts/run_training.sh`. Passing `--collect` also runs
`scripts/collect.sh` and syncs a slim bundle back into `remote_artifacts/`:
the final adapter, logs, job manifests, a checkpoint listing, and a small
`anvil_cloud_results.tar.gz`. Full epoch checkpoint state remains on the
remote host unless you copy it manually.

The generated bootstrap creates a system-site-packages venv so managed GPU
images can keep their baked-in CUDA/PyTorch stack. It installs Anvil from the
selected branch, installs ACE-Step's training dependencies separately, and then
installs ACE-Step with dependency resolution disabled so ACE-Step's exact torch
pin does not replace the provider image's working torch build. The Anvil
package is force-refreshed from the selected Git ref so reusable pods do not
keep stale code just because the package version number stayed the same. Remote uploads
preserve `.venv`, `work`, `outputs`, and `logs`, so a later `--skip-bootstrap`
training run does not erase the environment created by bootstrap. Bootstrap
also verifies the required ACE-Step checkpoints before training. Every job
needs the shared main bundle for `vae` and text components, and non-turbo jobs
also fetch their selected DiT checkpoint, such as `acestep-v15-sft` for SFT.
Set `ANVIL_SKIP_CHECKPOINT_DOWNLOAD=1` only when the remote checkpoint cache is
already populated or mounted somewhere else.

Useful RunPod follow-ups:

```bash
anvil cloud runpod status POD_ID
anvil cloud runpod terminate POD_ID --dry-run
anvil cloud runpod terminate POD_ID
```

Each job package has this shape:

```text
cloud-job/
  job.json
  training_bundle.json
  inputs/
    dataset/
      captions.json
      dataset_manifest.json
      clips/...
      stems/...
  logs/
  outputs/
  scripts/
    bootstrap.sh
    run_training.sh
    collect.sh
  work/
```

`anvil cloud package` rewrites `inputs/dataset/captions.json` so the selected
`--primary-asset` becomes the actual training file. That is what lets a future
run train against full mixes, instrumental stems, or vocal stems without
manually editing captions. Instrumental jobs can keep the default
`--training-lyrics "[Instrumental]"`; vocal-stem jobs should pass a short
non-instrumental marker such as
`--training-lyrics "vocal stem, expressive vocal performance"`. If the source
dataset includes reviewed transcripts, add `--training-lyrics-source transcript`
so remote preprocessing uses per-clip lyrics text before falling back to the
generic marker.

Expected behavior:

- estimate hourly cost before launch
- require `--max-hours` for any paid run
- create clearly tagged cloud instances
- store the remote instance ID locally
- verify CUDA and `nvidia-smi`
- install Anvil, ACE-Step, and training dependencies idempotently without
  replacing the provider image's CUDA/PyTorch stack
- sync dataset/config to the remote machine
- run training inside `tmux`, `screen`, or a supervised service
- stream logs locally
- checkpoint periodically
- generate fixed validation samples after checkpoints
- collect LoRA adapters, configs, logs, and sample audio
- destroy the instance by default on success or failure
- provide panic commands such as `anvil cloud list` and
  `anvil cloud destroy-all-anvil`

## GPU Notes

For ACE-Step LoRA work, a single H200 is the practical starting point. It has a
large memory pool, high bandwidth, and the safest CUDA software path. The point
is not that smaller GPUs are useless. It is that H200 reduces the chance that a
training experiment turns into VRAM triage.

Practical ordering for this project:

| GPU | Notes |
| --- | --- |
| H200 | Best first cloud target for ACE-Step LoRA and deeper experiments. |
| L40S | Good CUDA-compatible value option for moderate LoRA runs. |
| RTX 6000 Ada | Workstation-class 48 GB option, usable for smaller training. |
| RTX 4000 Ada | Useful for inference and small experiments, likely cramped. |
| MI350X | Huge memory, but ROCm compatibility makes it a later experiment. |
| B200/B300 | Newer Blackwell-class hardware, excellent but likely overkill. |

Start with one GPU. Multi-GPU only helps once the training code can actually use
it efficiently.

## Dataset Reality

The hard part is not only compute. It is dataset quality.

For LoRA:

- minutes to a few hours of clean audio can be useful
- 80 clips at 10 to 35 seconds is enough for early style tests
- repeated/bad clips can overfit the adapter quickly
- captions matter more than they seem
- loss is only a signal; listening tests decide whether the adapter is good

For full or deeper fine-tuning:

- hundreds or thousands of hours become more relevant
- rights and licensing matter much more
- caption consistency becomes a core training problem
- the risk of catastrophic forgetting increases
- evaluation needs fixed prompt suites, not only loss curves

Clean sources include owned recordings, commissioned material, opt-in artists,
licensed catalogs, and open datasets whose licenses allow the intended use.
Creative Commons does not always mean unrestricted use, especially for
commercial training or redistribution.

## YouTube Dataset Workflow

The existing dataset tooling can build clips from local files or authorized
YouTube sources. A future production-grade flow should be:

1. Download allowed source audio with `yt-dlp`.
2. Normalize loudness and sample rate.
3. Split into clips with overlap controls.
4. Reject silence, clipping, noise, and bad segments.
5. Optionally separate stems.
6. Transcribe vocal sections when useful.
7. Generate captions with style, instrumentation, vocal, and mix notes.
8. Run embedding QA for duplicates and outliers.
9. Write `character_sheet.json`, `captions.json`, and training config.
10. Preprocess into ACE-Step tensors.
11. Train LoRA and generate validation samples.

The dataset builder should make review easy. It should not pretend every clip is
good just because it was downloaded successfully.

## Demucs and Source Separation

Demucs is a music source separation tool. It takes a finished mix and splits it
into stems such as:

```text
song.wav
  vocals.wav
  drums.wav
  bass.wav
  other.wav
```

In Anvil, source separation would help with dataset focus:

- train on full mixes for an overall channel/style adapter
- train on instrumental stems to reduce voice imprinting
- inspect vocal stems for transcription and vocal-character captions
- reject clips where stems are obviously broken or noisy
- build separate analysis reports for vocals, drums, bass, and accompaniment

Demucs stems are not perfect. They can contain bleed, artifacts, or missing
transients, so they should be treated as review/filtering aids rather than
perfect ground truth.

Useful references:

- <https://github.com/facebookresearch/demucs>
- <https://github.com/nomadkaraoke/python-audio-separator>

## LoRA Versus Deeper Training

LoRA is the right first move for Anvil because it is cheaper, smaller, and
reversible. It teaches a base model a style, character, or production tendency
without rewriting the whole model.

```text
LoRA:
"Bias this existing model toward this sound."

Full fine-tune:
"Change the model weights so the model itself behaves differently."

Training from scratch:
"Build a new model from the ground up."
```

For most user workflows, LoRA is enough:

- genre/style packs
- production tone
- vocal delivery tendencies
- instrumentation bias
- channel or project identity

Full fine-tuning becomes interesting when LoRA cannot move the model far enough
or when the target behavior should become native to the model instead of loaded
as an adapter.

## ACE-Step Components

ACE-Step has at least two major pieces worth thinking about separately.

### DiT Generator

The DiT path renders the audio. Fine-tuning or adapting this side is most useful
when the goal is sound:

- timbre
- mix character
- vocal texture
- instrument tone
- production style
- overall audio quality

This is the natural target for early LoRA work.

### 5 Hz LM Planner

The 5 Hz LM is the planner path. It works at a compressed musical-token rate
instead of raw audio sample rate, and it can create or influence the plan that
the DiT renders.

Fine-tuning the planner is useful when the goal is musical behavior:

- better song structure
- better verse/chorus/bridge pacing
- better prompt-to-arrangement alignment
- better genre interpretation
- better metadata or caption reasoning
- more useful `thinking`/CoT behavior

Simple framing:

```text
Fine-tune DiT:
"Make the audio sound more like this."

Fine-tune 5 Hz LM:
"Make the model plan songs more like this."

Fine-tune both:
"Plan this kind of song better, then render it in that style better."
```

The planner can be fine-tuned, but it is a deeper project than DiT LoRA because
the training data needs to include or derive the semantic code targets the
planner emits.

## Training Experiment Loop

Cloud training is only useful if it improves iteration. A good experiment should
produce more than a checkpoint.

Each run should save:

- dataset manifest
- `character_sheet.json`
- training config
- adapter/checkpoints
- loss curve
- console logs
- fixed validation prompts
- generated validation audio
- a short run report with subjective notes

The comparison loop should answer:

- Did it learn the target style?
- Did it overfit?
- Did vocals get too close to a reference voice?
- Did prompt control get worse?
- Did the adapter work better on SFT or Turbo?
- Which checkpoint actually sounded best?

## Generation Trace Panel

Anvil already writes JSON sidecars, but the long-term idea is to make those
sidecars useful while judging generations instead of only after the fact. The
trace panel would show how a song moved through the ACE-Step path and make A/B
comparison less guessy.

Useful views:

- direct DiT versus 5 Hz LM thinking path
- whether LM thinking, CoT metadata, CoT caption, or DCW were enabled
- model, LoRA, seed, sampler, steps, CFG, duration, and negative prompt
- enhanced prompt versus original prompt
- generated lyrics versus user-provided lyrics
- inferred metadata, language, tempo, key, or structure when available
- checkpoint or LoRA run that produced the output
- validation sample group for the same prompt

The panel should support side-by-side comparison of two or more generation
sidecars:

- highlight changed parameters
- show prompt and lyric diffs
- show waveform/loudness summaries
- show spectrogram snapshots if available
- play A/B audio without leaving the page
- mark a preferred take and attach a short listening note

This would be especially useful for the XL-SFT/thinking/DCW experiments where
small registry changes can dramatically change the output. The goal is not a
deep model debugger at first. It is a practical listening lab that answers,
"what changed between the good one and the wrecked one?"

## Product Ideas

These are likely future Anvil features:

- cloud training runner
- cloud training Gradio tab
- dataset review UI
- source separation during dataset prep
- fixed prompt A/B evaluation
- automatic validation renders per checkpoint
- LoRA recipe presets
- generation trace panel for ACE-Step pathing, sidecar diffing, and A/B audio
  comparison
- Apple Music/local library export helper
- release-kit generator for cover art, titles, descriptions, and tags

## Personal Use Case

One strong use case is a private music workshop: create lawful or private style
datasets, train local adapters, generate music for personal listening, and keep
the output in a local library. The same tooling can still be built with clean
defaults so that public or commercial use remains easier to reason about later.

The guiding principle is creator-friendly power: make it easy to build a private
radio station around a taste or aesthetic without turning the tool into a
copyright mess by default.
