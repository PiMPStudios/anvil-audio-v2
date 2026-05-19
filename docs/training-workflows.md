# Dataset and Training Workflows

Quick command reference for building reviewable datasets. See `datasets.md` for the deeper dataset guide.

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
