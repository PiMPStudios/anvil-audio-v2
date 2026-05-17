# Dataset Separation Plan

This branch starts the source-separation layer for Anvil datasets. The goal is
to make training inputs trustworthy before adding cloud GPU orchestration.

## Objective

Build a local, reviewable workflow that can turn source tracks into
training-ready clips with optional stems, captions, QA reports, and stable
metadata.

The first milestone is not H200 automation. It is proving that Anvil can produce
datasets that are worth sending to an H200 later.

## Scope

In scope:

- add an Anvil source-separation abstraction
- integrate `python-audio-separator` as the first backend
- support Demucs-compatible models through that backend
- separate existing dataset clips into stems
- cache separated stems so repeated QA does not recompute work
- write stem metadata into clip sidecars
- extend dataset QA to understand stem availability and basic stem health
- document training-bundle expectations for local and cloud runs

Out of scope for this branch:

- DigitalOcean API integration
- remote H200 bootstrap
- automatic cloud training jobs
- full Gradio review UI
- maintaining a custom Demucs fork

## Proposed CLI

Separate clips in an existing dataset:

```bash
anvil dataset separate ./datasets/dark-blues \
    --backend audio-separator \
    --mode four-stem \
    --model auto
```

Create instrumental-only training material:

```bash
anvil dataset separate ./datasets/dark-blues \
    --backend audio-separator \
    --mode instrumental \
    --model auto
```

Run QA with stem checks:

```bash
anvil dataset qa ./datasets/dark-blues --include-stems
```

Export a stable local training bundle:

```bash
anvil dataset export-training-bundle ./datasets/dark-blues \
    --profile acestep-lora \
    --include full-mix,instrumental
```

## Dataset Layout

The dataset schema should be versioned before training and cloud automation
depend on it.

```text
datasets/name_YYYYMMDD_HHMMSS/
  clips/
    clip_0001.wav
    clip_0001.json
  stems/
    clip_0001/
      vocals.wav
      instrumental.wav
      drums.wav
      bass.wav
      other.wav
      separation.json
  captions.json
  character_sheet.json
  dataset_manifest.json
  dataset_qa_report.md
  dataset_qa_report.json
  training_bundle.json
```

Every dataset-level manifest should include:

```json
{
  "anvil_dataset_version": "1.0"
}
```

Every stem run should write `separation.json` with:

- backend name and version
- model name
- source clip path
- output stems
- sample rate
- duration
- peak/RMS summary
- elapsed time
- any warnings

## Backend Shape

Suggested module layout:

```text
anvil_audio/separation/
  __init__.py
  base.py
  audio_separator_backend.py
  registry.py
```

The Anvil layer should own stable concepts:

- `SeparationBackend`
- `SeparationRequest`
- `SeparationResult`
- `StemInfo`
- `SeparationMode`

The backend should hide dependency-specific details such as model filenames,
temporary output names, and package-specific flags.

## Separation Modes

Initial modes:

| Mode | Outputs | Use |
| --- | --- | --- |
| `instrumental` | `vocals.wav`, `instrumental.wav` | Reduce voice imprinting for production-style LoRAs. |
| `four-stem` | `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav` | Analyze arrangement and build richer captions. |
| `vocals` | `vocals.wav` | Transcription and vocal-character analysis. |

`full-mix` remains the original clip and does not require separation.

## QA Checks

Stem-aware QA should flag:

- missing expected stems
- stems with near-silence
- stems with clipping
- duration mismatch against the source clip
- failed separation runs
- suspiciously loud vocal bleed in instrumental mode
- repeated source clips or near-duplicate captions

The report should advise review instead of deleting automatically.

## Dependency Strategy

Use `python-audio-separator` as the first integration path because it exposes
multiple model families while still giving access to Demucs-style workflows.
Keep it optional until dataset separation is requested.

Demucs should be treated as an engine/model choice rather than something Anvil
maintains directly. A direct Demucs backend can be added later if the wrapper
blocks progress.

## Success Criteria

The branch is ready for broader testing when:

- `anvil dataset separate` works on an existing dataset
- stem output is deterministic and cached
- sidecars record enough information to reproduce separation
- QA reports stem health without breaking existing QA
- existing dataset build and LoRA tests still pass
- docs explain the workflow clearly

Once this layer is reliable, the next branch can build the cloud training runner
around the stable training bundle instead of around raw source files.
