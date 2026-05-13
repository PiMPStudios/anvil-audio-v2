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
non-finite conditioning values are written.

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
