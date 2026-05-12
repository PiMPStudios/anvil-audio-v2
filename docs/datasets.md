# Datasets

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

## ACE-Step LoRA Preprocessing

ACE-Step LoRA training needs preprocessed tensor files. Convert an Anvil dataset
like this:

```bash
anvil lora preprocess ./datasets/my_style_20260512_140000 \
    --output-dir ./tensors/my_style \
    --model-variant sft \
    --custom-tag my_style
```

This writes `acestep_dataset.json` into the dataset folder, then delegates to
ACE-Step's `training_v2` preprocessing pipeline to produce `.pt` tensors.

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
