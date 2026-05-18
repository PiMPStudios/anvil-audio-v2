# Datasets

This is the current Anvil workflow for building reviewable audio datasets and
turning them into ACE-Step LoRA training inputs.

Anvil has two dataset paths:

- `anvil dataset` creates reviewable local clip datasets for LoRA work.
- The legacy Stable Audio loaders still support `audio_dir` and S3
  WebDataset configs for older training experiments.

## Anvil LoRA Dataset Builder

Build a local dataset from audio files:

```bash
anvil dataset build-local ./source-audio \
    --name my_style \
    --clips 80 \
    --clip-length 35 \
    --style-hint "anthemic alternative rock, live drums" \
    --caption-mode heuristic
```

Build from an authorized YouTube video, playlist, or channel:

```bash
anvil dataset build-youtube "https://www.youtube.com/playlist?list=..." \
    --name my_channel_style \
    --tracks 12 \
    --clips 80 \
    --clip-length 35 \
    --caption-mode llm
```

Only train on material you own or are authorized to train on.

YouTube source downloads may start as whatever container YouTube provides, but
Anvil writes generated dataset clips as WAV files under `clips/`. That keeps
source separation, QA, and training preprocessing on a predictable audio format.

## Optional Vocal Transcription

For vocal-focused datasets, add local Whisper hints while building clips:

```bash
anvil dataset build-local ./source-audio \
    --name my_vocal_style \
    --clips 80 \
    --clip-length 35 \
    --style-hint "dark blues, smoky male vocal, raw guitar" \
    --caption-mode llm \
    --transcribe-vocals
```

`--transcribe-vocals` uses source text and `--style-hint` to decide which clips
are likely vocal-forward. Use `--transcribe-all` to force transcription for
every clip. The feature is optional and local-only. On Apple Silicon,
`bash install.sh` installs the lightweight `lightning-whisper-mlx` runtime, and
the selected Whisper model downloads lazily on first transcription. Manual
installs can use:

```bash
pip install lightning-whisper-mlx  # Apple Silicon
# or
pip install openai-whisper         # local PyTorch Whisper
```

Generated clip sidecars and `captions.json` include `transcript` and
`transcription` fields when text is detected. Captions get a compact
`lyric hint` phrase rather than a full transcript dump, which keeps LoRA labels
focused.

## Output Layout

```text
datasets/my_style_YYYYMMDD_HHMMSS/
  clips/
    clip_0001.wav
    clip_0001.json
  captions.json
  character_sheet.json
  dataset_manifest.json
  dataset_config.json
  sources/
```

The clip sidecars and `captions.json` contain the prompt/caption, tags,
negative tags, source metadata, clip timing, and deterministic audio analysis.
`character_sheet.json` summarizes the dataset style so you can review the
result before training.

## Embedding QA

Run a caption-embedding QA pass before LoRA preprocessing:

```bash
anvil dataset qa ./datasets/my_style_YYYYMMDD_HHMMSS
```

This uses `Qwen3-Embedding-0.6B` to cluster captions, flag near duplicates,
surface semantic outliers, list low-confidence captions, and write both
`dataset_qa_report.json` and `dataset_qa_report.md` into the dataset folder.
The default model resolution prefers the Anvil ACE-Step checkpoint cache at
`~/.cache/anvil-audio/acestep/checkpoints/Qwen3-Embedding-0.6B`, then falls
back to `Qwen/Qwen3-Embedding-0.6B` on HuggingFace.

Useful options:

```bash
anvil dataset qa ./datasets/my_style_YYYYMMDD_HHMMSS \
    --duplicate-threshold 0.9 \
    --cluster-threshold 0.78 \
    --outlier-threshold 0.55 \
    --device auto
```

Treat this report as a review aid, not an automatic delete list. Outliers are
often exactly what you want if the adapter is supposed to cover multiple
substyles.

## Optional Source Separation

For datasets that mix vocals and instruments, separate clips into stems before
LoRA preprocessing:

```bash
anvil dataset separate ./datasets/my_style_YYYYMMDD_HHMMSS \
    --mode instrumental \
    --backend audio-separator
```

This uses the optional `audio-separator` package and writes cached stems under
`stems/` without replacing the original clips:

```text
datasets/my_style_YYYYMMDD_HHMMSS/
  clips/
    clip_0001.wav
    clip_0001.json
  stems/
    clip_0001/
      vocals.wav
      instrumental.wav
      separation.json
```

For older or imported datasets whose `clips/` are still `.m4a`, `.webm`, or
another compressed format, `anvil dataset separate` normalizes each non-WAV clip
to `stems/<clip>/source.wav` before calling `audio-separator`. New datasets
built with `anvil dataset build-youtube` should already have WAV clips.

Install the separation backend in an isolated tool environment when you need it.
`audio-separator` currently wants newer `numpy`/`protobuf` versions than some
core Anvil audio dependencies, so do not install it directly into the Anvil venv
unless you are deliberately testing dependency changes.

```bash
python3.13 -m venv ~/.cache/anvil-audio/tools/audio-separator
~/.cache/anvil-audio/tools/audio-separator/bin/pip install \
    'audio-separator>=0.44.1' \
    onnxruntime
export ANVIL_AUDIO_SEPARATOR_BIN=~/.cache/anvil-audio/tools/audio-separator/bin/audio-separator
```

Useful modes:

| Mode | Stems | Use |
| --- | --- | --- |
| `instrumental` | `vocals`, `instrumental` | Build production-style datasets with less voice imprinting. |
| `four-stem` | `vocals`, `drums`, `bass`, `other` | Review arrangement and richer caption clues. |
| `vocals` | `vocals` | Pull vocal-only material for transcription or review. |

Run a tiny smoke test before processing a large dataset:

```bash
anvil dataset separate ./datasets/my_style_YYYYMMDD_HHMMSS \
    --mode instrumental \
    --limit 2
```

Use `--force` to recompute stems after changing models or settings. The
separation metadata is written into each clip sidecar, `captions.json`, and
`dataset_manifest.json` so future QA and training-bundle work can reuse it.

After separation, include stem health checks in QA:

```bash
anvil dataset qa ./datasets/my_style_YYYYMMDD_HHMMSS --include-stems
```

Stem-aware QA flags missing stem files, near-silent stems, possible clipping,
and duration mismatches against the source clip.

## Caption Audit and Repair

If QA reports many duplicate or low-confidence captions, audit the caption file:

```bash
anvil dataset captions ./datasets/my_style_YYYYMMDD_HHMMSS
```

To deterministically rebuild weak captions from source titles, audio analysis,
style hints, and stem metadata, run repair mode as a dry run first:

```bash
anvil dataset captions ./datasets/my_style_YYYYMMDD_HHMMSS \
    --repair \
    --style-hint "dark blues, slow guitar, atmospheric vocals"
```

Then write the repaired captions when the report looks right:

```bash
anvil dataset captions ./datasets/my_style_YYYYMMDD_HHMMSS \
    --repair \
    --write \
    --style-hint "dark blues, slow guitar, atmospheric vocals"
```

This updates `captions.json`, matching clip sidecars, and records
`caption_repair` metadata in `dataset_manifest.json`.

## Training Bundle Export

After clips, captions, optional stems, and QA look reasonable, export a stable
bundle for local training or future cloud runners:

```bash
anvil dataset export-training-bundle ./datasets/my_style_YYYYMMDD_HHMMSS \
    --profile acestep-lora \
    --include full-mix,instrumental
```

This writes `training_bundle.json` with the selected assets, captions, tags,
negative tags, timing, and warnings for missing optional assets. Use `--strict`
when a missing requested asset should fail the export.

## Portable Cloud Jobs

`anvil cloud package` turns a training bundle into a self-contained folder that
can be uploaded to any SSH GPU host:

```bash
anvil cloud package ./datasets/my_style_YYYYMMDD_HHMMSS/training_bundle.json \
    --output-dir ./cloud-jobs/my_style_h200 \
    --primary-asset instrumental \
    --model-variant sft \
    --recipe lora-balanced \
    --max-hours 6
```

The package copies the selected assets under `inputs/dataset`, rewrites
`captions.json` so the selected `--primary-asset` is the training file, and
writes remote scripts for bootstrap, training, and collection.

On managed CUDA images, the generated `bootstrap.sh` keeps the image's existing
PyTorch install by creating a system-site-packages venv. ACE-Step is installed
after its non-torch training dependencies with dependency resolution disabled,
which avoids torch pin conflicts on provider images such as RunPod's PyTorch
templates. Bootstrap also downloads any required ACE-Step checkpoints that are
missing on the remote cache. For SFT LoRA training, that means the shared main
bundle, including `vae`, plus `acestep-v15-sft`.

Preview an SSH run before touching the remote box:

```bash
anvil cloud doctor
anvil cloud search --gpu h200 --max-price 4 --min-vram-gb 80

export RUNPOD_API_KEY=...
anvil cloud runpod launch ./cloud-jobs/my_style_h200 \
    --gpu-type "NVIDIA H200" \
    --dry-run

anvil cloud run-ssh ./cloud-jobs/my_style_h200 \
    --host ubuntu@203.0.113.10 \
    --dry-run
```

Remove `--dry-run` when the commands look right. Add `--collect` to package and
sync `outputs/` plus `logs/` back after training.

`anvil cloud search` uses GPUFindr's public read-only catalog. It is only a
provider discovery step. The RunPod adapter can create pods through RunPod's
GraphQL API once `RUNPOD_API_KEY` is set. Keep `--dry-run` on until the launch
request looks right, use `gpuId` values from `runpodctl gpu list`, and
terminate pods promptly when training finishes.

## ACE-Step LoRA Preprocessing

ACE-Step LoRA training needs preprocessed tensor files. Convert an Anvil dataset
like this:

```bash
anvil lora preprocess ./datasets/my_style_20260512_140000 \
    --output-dir ./tensors/my_style \
    --model-variant sft \
    --precision fp32 \
    --custom-tag my_style
```

This writes `acestep_dataset.json` into the dataset folder, then delegates to
ACE-Step's `training_v2` preprocessing pipeline to produce `.pt` tensors.
Anvil validates the tensor files after preprocessing and fails fast if any
non-finite conditioning values are written. The generated ACE-Step sample
filenames are index-prefixed so stem-heavy datasets with many files named
`instrumental.wav` or `vocals.wav` do not collide inside ACE-Step's temporary
preprocessing outputs.

Then run training from those tensors:

```bash
anvil lora train ./tensors/my_style \
    --output-dir ./lora-runs/my_style \
    --model-variant sft \
    --epochs 20
```

On Apple Silicon, add `--basic-loop` if Lightning Fabric hits MPS AMP gradient
scaler errors before the first optimizer step. Keep preprocessing at
`--precision fp32`; lower precision can poison the training tensors on the
Apple path.

Once the final adapter exists, register it with:

```bash
anvil lora import-local ./lora-runs/my_style/final --name my-style
```

After import, select it in Gradio under the **ACE-Step LoRA** accordion or pass
it to `anvil generate --lora my-style`.

## Legacy Stable Audio Dataset Config

Some inherited Stable Audio training code still consumes a JSON dataset config.
`anvil dataset` writes a starter `dataset_config.json` automatically:

```json
{
  "dataset_type": "audio_dir",
  "datasets": [
    {
      "id": "my_style",
      "path": "/path/to/datasets/my_style/clips"
    }
  ],
  "random_crop": true
}
```

For S3 WebDataset experiments, use:

```json
{
  "dataset_type": "s3",
  "datasets": [
    {
      "id": "s3-test",
      "s3_path": "s3://my-bucket/datasets/webdataset/audio/"
    }
  ],
  "random_crop": true
}
```

## Custom Metadata

Legacy Stable Audio training can add a `custom_metadata_module` to the dataset
config. The module must define `get_custom_metadata(info, audio)` and return a
dictionary whose values are merged into the training metadata.

```json
{
  "dataset_type": "audio_dir",
  "datasets": [
    {
      "id": "my_audio",
      "path": "/path/to/audio/dataset/"
    }
  ],
  "custom_metadata_module": "/path/to/custom_metadata.py",
  "random_crop": true
}
```

Example module:

```python
def get_custom_metadata(info, audio):
    return {"prompt": info["relpath"]}
```
