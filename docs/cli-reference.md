# CLI Reference

Command and flag reference for Gradio, generation, datasets, cloud jobs, LoRA, and containers.

## `run_gradio.py` flags

| Flag | Description |
|------|-------------|
| `--model` | Registry model name (e.g. `stable-audio-open-1.0`, `acestep-v1.5-turbo`) |
| `--pretrained-name` | HuggingFace Hub repo ID (e.g. `stabilityai/stable-audio-open-1.0`) |
| `--model-config` | Local model config JSON (ignored if `--model` or `--pretrained-name` set) |
| `--ckpt-path` | Local checkpoint (ignored if `--model` or `--pretrained-name` set) |
| `--pretransform-ckpt-path` | Optional separate VAE checkpoint |
| `--username` / `--password` | Gradio auth |
| `--model-half` | Use float16 inference |
| `--device` | `cuda`, `mps`, or `cpu` (auto-detects if omitted) |
| `--project` | Outputs go to `~/anvil-audio-outputs/{project}/` |
| `--share` | Create a public Gradio share URL |

---

## `anvil generate` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model NAME` | — | Registry model name |
| `--list-models` | — | Print registry and exit (also works as `anvil --list-models`) |
| `--model-config PATH` | — | Legacy: local JSON config |
| `--ckpt-path PATH` | — | Legacy: local checkpoint |
| `--pretransform-ckpt-path PATH` | — | Separate VAE checkpoint |
| `--prompt TEXT` | — | Single text prompt |
| `--cond-yaml-path PATH` | — | Batch YAML conditions file |
| `--seconds-start` | `0.0` | Start time (seconds) |
| `--seconds-total` | `30.0` | Duration (seconds) |
| `--negative-prompt` | blank | Text describing sounds or qualities to avoid |
| `--enhance-prompt` | off | Enhance prompt and negative prompt |
| `--write-lyrics` | off | Write duration-aware ACE-Step lyrics |
| `--intelligence-model` | default | LLM path or HuggingFace repo |
| `--lora` | blank | ACE-Step adapter id/name or direct PEFT/LoKr path |
| `--lora-scale` | `1.0` | ACE-Step adapter strength |
| `--lora-adapter-name` | blank | Optional runtime adapter name |
| `--lora-stack` | repeatable | Additional ACE-Step adapter as `id-or-path[:scale]` |
| `--output-dir` | `./output` | Output directory |
| `--format` | `wav` | `wav`, `flac`, `mp3`, or `ogg` |
| `--clip-length` | off | Clip to `seconds_total` |
| `--sample-steps` | pipeline default | Diffusion / inference steps |
| `--cfg-scale` | pipeline default | CFG guidance scale |
| `--sampler-type` | pipeline default | Sampler type |
| `--sigma-min` / `--sigma-max` | pipeline default | Noise schedule bounds |
| `--n-sample-per-cond` | `1` | Samples per condition |
| `--batch-size` | `10` | Items per GPU batch |
| `--seed` | `-1` (random) | RNG seed |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

---

## `anvil dataset` flags

| Command / Flag | Default | Description |
| --- | --- | --- |
| `build-local SOURCE_DIR` | - | Build from a folder of audio files |
| `build-youtube URL` | - | Download authorized YouTube audio with `yt-dlp` |
| `qa DATASET_DIR` | - | Run Qwen embedding QA on captions |
| `separate DATASET_DIR` | - | Separate clips into cached vocals/instrumental/stems |
| `captions DATASET_DIR` | - | Audit duplicate and low-confidence captions |
| `export-training-bundle DATASET_DIR` | - | Write `training_bundle.json` for local/cloud training |
| `--name` | `anvil_dataset` | Dataset name in manifests |
| `--output-dir` | timestamped `./datasets/...` | Output dataset directory |
| `--clips` | `40` | Maximum clips to write |
| `--clip-length` | `35` | Clip length in seconds |
| `--min-clip-length` | `8` | Skip source files shorter than this |
| `--stride` | clip length | Seconds between clip starts |
| `--sample-rate` | `48000` | Output sample rate |
| `--channels` | `2` | Output channel count |
| `--style-hint` | blank | Style context added to captions |
| `--caption-mode` | `heuristic` | `heuristic`, `llm`, or `off` |
| `--llm-model` | default | LLM path/repo for caption cleanup |
| `--transcribe-vocals` | off | Add local Whisper hints to likely vocal clips |
| `--transcribe-all` | off | Transcribe every clip |
| `--transcription-backend` | `auto` | `lightning-whisper-mlx` or `whisper` |
| `--transcription-model` | backend default | Whisper model name |
| `--transcription-language` | auto | Optional source language code |
| `--tracks` | unlimited | YouTube-only max source videos/tracks |
| `--delete-downloads` | off | Delete raw downloads after clips are written |
| `--quiet-ytdlp` | off | Pass `--quiet` to `yt-dlp` |
| `--embedding-model` | local Qwen cache | QA-only embedding model path/repo |
| `--include-stems` | off | QA-only stem health checks |
| `--duplicate-threshold` | `0.9` | QA-only duplicate similarity cutoff |
| `--cluster-threshold` | `0.78` | QA-only cluster similarity cutoff |
| `--outlier-threshold` | `0.55` | QA-only outlier neighbor cutoff |
| `--mode` | `instrumental` | Separation mode: `instrumental`, `four-stem`, or `vocals` |
| `--backend` | `audio-separator` | Separation backend |
| `--model` | `auto` | Separation model filename |
| `--force` | off | Recompute cached stems |
| `--limit` | unlimited | Separation smoke-test limit |
| `--repair` | off | Caption command repairs weak duplicates |
| `--write` | off | Caption repair writes files instead of dry-running |
| `--include` | `full-mix` | Bundle assets: `full-mix`, `vocals`, `instrumental`, `drums`, `bass`, `other` |
| `--profile` | `acestep-lora` | Training bundle profile |
| `--strict` | off | Fail bundle export if a requested asset is missing |

---

## `anvil cloud` flags

| Command / Flag | Default | Description |
| --- | --- | --- |
| `doctor` | - | Check local `bash`, `ssh`, `rsync`, and `git` availability |
| `search` | - | Search GPUFindr's public GPU availability catalog |
| `package TRAINING_BUNDLE` | - | Build a portable SSH training job folder |
| `run-ssh JOB_DIR` | - | Upload/bootstrap/train/collect on an existing SSH GPU host |
| `runpod launch JOB_DIR` | - | Create a RunPod pod for a packaged job |
| `runpod status POD_ID` | - | Show pod status and SSH host/port hints |
| `runpod terminate POD_ID` | - | Terminate a RunPod pod |
| `--gpu` | blank | Search GPU name filter, e.g. `h200` |
| `--source` | blank | Search provider filter, e.g. `runpod` |
| `--max-price` | unlimited | Search maximum hourly price |
| `--min-vram-gb` | `0` | Search minimum VRAM |
| `--output-dir` | required | Cloud package destination |
| `--primary-asset` | `full-mix` | Training asset copied into the remote dataset |
| `--model-variant` | `sft` | ACE-Step variant for remote preprocess/train |
| `--recipe` | `lora-balanced` | Remote LoRA recipe preset |
| `--max-hours` | `6` | Runtime budget written into `job.json` |
| `--training-lyrics` | `[Instrumental]` | Lyrics marker passed to remote LoRA preprocessing |
| `--training-lyrics-source` | `constant` | Use `transcript` to fill sample lyrics from per-clip transcripts |
| `--host` | required | SSH host for `run-ssh`, e.g. `ubuntu@203.0.113.10` |
| `--port` | `22` | SSH port |
| `--identity-file` | default SSH config | Optional SSH private key |
| `--skip-bootstrap` | off | Reuse an already bootstrapped remote job folder |
| `--no-train` | off | Upload/bootstrap only |
| `--collect` | off | Sync a slim final-adapter bundle plus logs after training |
| `--gpu-type` | required | RunPod `gpuId`, e.g. `NVIDIA H200` |
| `--cloud-type` | `ALL` | RunPod cloud type: `ALL`, `SECURE`, or `COMMUNITY` |
| `--minimal` | off | Send a UI-like minimal RunPod launch request |
| `--dry-run` | off | Print cloud/SSH/API commands without spending money |

---

## `anvil lora` flags

| Command / Flag | Default | Description |
| --- | --- | --- |
| `list` | - | List registered adapters |
| `info REF` | - | Show adapter metadata or resolve a direct path |
| `import-local PATH` | - | Register a PEFT adapter dir or LoKr safetensors |
| `import-hf REPO_ID` | - | Download and register a HuggingFace adapter |
| `write-dataset-json DATASET_DIR` | - | Convert an Anvil dataset to ACE-Step JSON |
| `preprocess DATASET_DIR` | - | Build ACE-Step `.pt` tensors |
| `train TENSOR_DIR` | - | Run ACE-Step corrected LoRA training |
| `--name` | inferred | Adapter display name |
| `--base-model` | `acestep-v1.5` | Compatibility note for adapter metadata |
| `--checkpoint-dir` | Anvil ACE-Step cache | ACE-Step checkpoints root |
| `--model-variant` | `sft` | `turbo`, `base`, `sft`, or custom folder name |
| `--precision` | `fp32` for preprocess | Preprocess/train precision |
| `--custom-tag` | blank | Trigger tag prepended during preprocessing |
| `--lyrics-source` | `constant` | Use `transcript` to prefer per-clip lyrics/transcript metadata |
| `--output-dir` | required | Tensor or training output directory |
| `--epochs` | `100` | Training epochs |
| `--rank` / `--alpha` | `64` / `128` | LoRA rank and alpha |
| `--basic-loop` | off | Use ACE-Step's non-Fabric loop for MPS AMP issues |
| `--dry-run` | off | Print the ACE-Step training command |

---

## Container Setup

Build a Docker image and optionally convert to Singularity for HPC clusters:

```bash
NAME=anvil-audio
docker build -t ${NAME} -f ./container/anvil-audio.Dockerfile .

# Convert to Singularity
singularity build anvil-audio.sif docker-daemon://anvil-audio
```

---
