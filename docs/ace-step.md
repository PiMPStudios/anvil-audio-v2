# ACE-Step Music and LoRA

ACE-Step setup, XL checkpoints, local prompt intelligence, and adapter usage/training.

## ACE-Step Music Generation (optional)

[ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) is an open-source full-song music generation
model that supports style tags and full lyric input. Anvil integrates it through the same
registry and UI as Stable Audio — no separate server or app required, and no manual repo clone needed.

ACE-Step is **optional**. If you don't install it, all other Anvil functionality works as normal.

### LM thinking

ACE-Step ships a separate 5 Hz LM that produces structured `audio_codes` fed
into the DiT. Anvil can initialise that LM automatically when a generation uses
thinking/COT/DCW options, but the built-in SFT entry defaults to the direct DiT
conditioning path because that is the known-good local baseline for blank-lyrics
SFT generation and avoids keeping the extra LM resident in memory.

When the LM path is configured, Anvil lazy-loads the LM from the checkpoint
specified in the registry entry the first time the LM path is actually needed.
The built-in entries default to:

| Model | LM checkpoint |
|-------|---------------|
| `acestep-v1.5-turbo` | `acestep-5Hz-lm-1.7B` (lighter, faster) |
| `acestep-v1.5-sft` | `acestep-5Hz-lm-4B` (heavier, better quality) |

Both checkpoints are downloaded automatically from HuggingFace on first use.

You can override the LM checkpoint for a specific registry entry via `lm_model_path` in
`registry.yaml` (see [ACE-Step models](stable-audio-and-mlx.md#ace-step-models) below), or set a global fallback
for all ACE-Step models via an environment variable:

```bash
export ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
```

If the LM fails to initialise (missing checkpoint, memory constraints, etc.) Anvil
falls back gracefully to DiT-only generation and prints a warning. You can enable
LM thinking explicitly with registry defaults or per-call overrides when you want
ACE-Step to generate semantic code hints and metadata.

### Apple Silicon acceleration

On macOS, Anvil automatically sets `ACESTEP_LM_BACKEND=mlx` before initializing ACE-Step,
enabling the MLX backend for the 5 Hz LM (unless you've already set the variable yourself).
The DiT and VAE also run via native MLX on Apple Silicon automatically.
To compare against the non-MLX DiT/VAE backend, launch with:

```bash
ANVIL_ACESTEP_USE_MLX_DIT=0 python run_gradio.py
```

If you run the ACE-Step MCP server or API separately, set the variable in your shell or MCP
`env` block:

```bash
export ACESTEP_LM_BACKEND=mlx
```

### Built-in registry entries

Anvil registers two ACE-Step variants automatically and stores their downloaded
weights under `~/.cache/anvil-audio/acestep/checkpoints`:

| Model name | Description | Steps | Notes |
|---|---|---|---|
| `acestep-v1.5-turbo` | Fast generation | 8 | Good for drafts and quick iteration |
| `acestep-v1.5-sft` | Full quality | 50 | Direct DiT conditioning, guidance 7.5, shift 3.0, experimental DCW off |

The SFT built-in intentionally disables ACE-Step's experimental DCW sampler
correction and LM thinking by default. Keep those options disabled for the
normal SFT path; enable them explicitly only when you want to experiment with
ACE-Step's LM/code-hint path.

### Optional XL checkpoints

ACE-Step v1.5 XL DiT checkpoints (`acestep-v15-xl-base`,
`acestep-v15-xl-sft`, and `acestep-v15-xl-turbo`) are large optional downloads.
Anvil does **not** register or auto-download them by default. If you add an XL
entry to `~/.anvil-audio/registry.yaml` and the checkpoint is missing, Anvil
stops before ACE-Step can start a surprise download.

Install an XL checkpoint explicitly before using it:

```bash
acestep-download \
    --dir "$HOME/.cache/anvil-audio/acestep/checkpoints" \
    --model acestep-v15-xl-turbo
```

Use one of these model names with `--model`: `acestep-v15-xl-turbo`,
`acestep-v15-xl-sft`, or `acestep-v15-xl-base`. Avoid `acestep-download --all`
unless you intentionally want every optional ACE-Step submodel.

Then register the XL model in your user registry:

```yaml
- name: acestep-v1.5-xl-turbo
  pipeline_type: acestep
  model_config_path: acestep-v15-xl-turbo
  lm_model_path: acestep-5Hz-lm-1.7B
  max_duration: 600
  default_params:
    steps: 8
    cfg_scale: 1.0
    audio_duration: 60
    sampler_type: ode
    shift: 3.0
    thinking: true
    use_cot_metas: true
    use_cot_caption: false
    use_cot_language: true
    dcw_enabled: false
    velocity_norm_threshold: 0.0
    velocity_ema_factor: 0.0
    sigma_min: 0.0
    sigma_max: 0.0
    # Optional load-time memory controls for large XL checkpoints:
    # offload_to_cpu: true
    # offload_dit_to_cpu: true
    # quantization: int8_weight_only
```

For XL SFT, use the SFT checkpoint name and the direct SFT defaults:

```yaml
- name: acestep-v1.5-xl-sft
  pipeline_type: acestep
  model_config_path: acestep-v15-xl-sft
  max_duration: 600
  default_params:
    steps: 50
    cfg_scale: 7.5
    audio_duration: 60
    sampler_type: ode
    shift: 3.0
    thinking: false
    use_cot_metas: false
    use_cot_caption: false
    use_cot_language: false
    dcw_enabled: false
    velocity_norm_threshold: 0.0
    velocity_ema_factor: 0.0
    sigma_min: 0.0
    sigma_max: 0.0
```

### CLI

```bash
# Single prompt (instrumental)
anvil generate --model acestep-v1.5-turbo \
    --prompt "indie pop, acoustic guitar, warm vocals, upbeat" \
    --negative-prompt "muddy mix, harsh clipping, distorted vocals"

# Batch generation with lyrics (see example file)
anvil generate --model acestep-v1.5-turbo \
    --cond-yaml-path example/generation/acestep_conditions.yaml \
    --output-dir ./out

# Generate with a registered LoRA adapter
anvil generate --model acestep-v1.5-sft \
    --prompt "my_style, anthemic rock, live drums, gritty guitars" \
    --lora my-style \
    --lora-scale 0.8 \
    --seconds-total 60

# Stack multiple PEFT LoRAs
anvil generate --model acestep-v1.5-sft \
    --prompt "my_style, roomy live band, gritty guitars" \
    --lora my-style \
    --lora-scale 0.75 \
    --lora-stack room-tone:0.25 \
    --seconds-total 60
```

Batch YAML format — each entry supports `prompt`, `negative_prompt`, `lyrics`,
and `seconds_total`:

```yaml
tracks:
  indie_pop:
    prompt: "indie pop, acoustic guitar, warm vocals, upbeat, sunny afternoon"
    lyrics: |
      [verse]
      Walking down the open road
      Sunlight through the trees
      [chorus]
      This is where I start again
    seconds_total: 30.0

  electronic_instrumental:
    prompt: "electronic, synthwave, driving bass, retro 80s, cinematic"
    negative_prompt: "muddy low end, harsh treble, clipping"
    lyrics: "[Instrumental]"
    seconds_total: 45.0
```

### Gradio web UI

```bash
# Turbo (fast, good for iteration)
python run_gradio.py --model acestep-v1.5-turbo

# Full quality
python run_gradio.py --model acestep-v1.5-sft
```

The ACE-Step UI adds a **Lyrics** field below the prompt. Leave it blank or enter
`[Instrumental]` for tracks with no vocals. Structure lyrics with section markers
like `[verse]`, `[chorus]`, `[bridge]`.

The **Negative prompt** field is available for every model. For ACE-Step, it is
passed to ACE-Step's LM/thinking path as `lm_negative_prompt`; direct DiT-only
SFT generation keeps it in metadata, but upstream ACE-Step does not expose a
separate DiT negative-prompt control.

### Local prompt intelligence

On Apple Silicon, Anvil can use a local MLX Llama model to enhance prompts,
suggest negative prompts, and write duration-aware lyrics. The default model is
`mlx-community/Llama-3.2-3B-Instruct-4bit`. First use downloads the model into
Anvil Audio's local cache:

```text
~/.cache/anvil-audio/llm/llama-3.2-3b-instruct-4bit/
```

You can override the model with `ANVIL_LLM_MODEL`, `ANVIL_LLM_MODEL_PATH`, or
the CLI `--model` / `--intelligence-model` flags.

CLI helper:

```bash
anvil enhance-prompt \
    --prompt "anthemic alternative rock with emotional male vocal" \
    --duration 60
```

Generate with automatic prompt enhancement and lyric writing:

```bash
anvil generate --model acestep-v1.5-sft \
    --prompt "anthemic alternative rock with emotional male vocal" \
    --seconds-total 60 \
    --enhance-prompt \
    --write-lyrics
```

In Gradio, use **Enhance Prompt**, **Write Lyrics**, or **Enhance + Lyrics**
above the generation controls.

---

## ACE-Step LoRA Adapters

Anvil can register and apply ACE-Step LoRA adapters without bundling the
adapter weights into this repo. Local adapter metadata lives at:

```text
~/.cache/anvil-audio/lora/adapters/
```

Import a PEFT LoRA directory or LoKr safetensors file:

```bash
anvil lora import-local ./lora-runs/my_style/final --name my-style
```

Import from HuggingFace:

```bash
anvil lora import-hf username/my-style-acestep-lora --name my-style
```

List and inspect registered adapters:

```bash
anvil lora list
anvil lora info my-style
```

Use an adapter from the CLI:

```bash
anvil generate --model acestep-v1.5-sft \
    --prompt "my_style, polished alternative rock, live drums" \
    --lora my-style \
    --lora-scale 0.75 \
    --seconds-total 60
```

Use an adapter in Gradio by loading an ACE-Step model, opening the
**ACE-Step LoRA** accordion, and selecting a registered adapter from the
dropdown. You can still paste a direct PEFT/LoKr path as a custom value. Add
more PEFT adapters in the additional-adapters box with `adapter:scale` entries.

On Apple Silicon, PEFT LoRA directories can run on Anvil's native MLX DiT path.
Anvil reads the PEFT weights and applies the tiny LoRA A/B matrices directly to
matching MLX Linear projections, so the large XL DiT can stay in MLX instead of
falling back to PyTorch/MPS. LoKr/LyCORIS adapters still use ACE-Step's PyTorch
DiT backend:

```bash
ANVIL_ACESTEP_USE_MLX_DIT=0 python ./run_gradio.py
ANVIL_ACESTEP_USE_MLX_DIT=0 anvil generate --model acestep-v1.5-sft \
    --prompt "my_style, dark blues noir rock" \
    --lora my-style
```

If MLX DiT is active and an unsupported adapter format is selected, Anvil fails
loudly instead of pretending the adapter affected the audio.

Anvil does not scan other applications for adapters. Keep this repo's LoRA path
explicit: train an adapter here, import a local PEFT/LoKr folder, or import a
standard HuggingFace adapter repo when you have one you want to use.

### Training a LoRA

The full automated flow is:

```bash
# 1. Build reviewable clips and captions
anvil dataset build-youtube "https://www.youtube.com/playlist?list=..." \
    --name my_style \
    --tracks 12 \
    --clips 80 \
    --clip-length 35 \
    --style-hint "anthemic alternative rock, live drums" \
    --caption-mode llm

# 2. Review duplicate captions, semantic outliers, and low-confidence clips
anvil dataset qa ./datasets/my_style_YYYYMMDD_HHMMSS

# 3. Convert clips into ACE-Step training tensors
anvil lora preprocess ./datasets/my_style_YYYYMMDD_HHMMSS \
    --output-dir ./tensors/my_style \
    --model-variant sft \
    --precision fp32 \
    --custom-tag my_style

# 4. Train with ACE-Step's corrected training_v2 fixed LoRA trainer
anvil lora train ./tensors/my_style \
    --output-dir ./lora-runs/my_style \
    --model-variant sft \
    --epochs 20 \
    --rank 64 \
    --alpha 128

# 5. Register the final adapter for generation
anvil lora import-local ./lora-runs/my_style/final --name my-style
```

`anvil lora train` writes an inference-ready PEFT adapter under
`<output-dir>/final/`, which ACE-Step can load directly.

On Apple Silicon, add `--basic-loop` if Lightning Fabric fails with MPS AMP
gradient-scaler errors. This uses ACE-Step's own non-Fabric training loop.
Keep preprocessing at `--precision fp32`; lower precision can produce non-finite
conditioning tensors on the Apple path.

### Portable Cloud Jobs

When the dataset is ready, export a bundle and package it for any SSH-accessible
GPU host:

```bash
anvil cloud doctor

anvil cloud search --gpu h200 --max-price 4 --min-vram-gb 80

anvil dataset export-training-bundle ./datasets/my_style_YYYYMMDD_HHMMSS \
    --include full-mix,instrumental

anvil cloud package ./datasets/my_style_YYYYMMDD_HHMMSS/training_bundle.json \
    --output-dir ./cloud-jobs/my_style_h200 \
    --primary-asset instrumental \
    --model-variant sft \
    --recipe lora-balanced \
    --max-hours 6

export RUNPOD_API_KEY=...
anvil cloud runpod launch ./cloud-jobs/my_style_h200 \
    --gpu-type "NVIDIA H200" \
    --dry-run

anvil cloud run-ssh ./cloud-jobs/my_style_h200 \
    --host ubuntu@203.0.113.10 \
    --dry-run
```

The cloud package includes `job.json`, copied training assets, rewritten
captions, and `bootstrap.sh`/`run_training.sh` scripts. Remove `--dry-run` once
the SSH and `rsync` commands look right. Provider API launchers can be added
later without changing the job format.

When packaging vocal-stem LoRA jobs, use `--primary-asset vocals` and set
`--training-lyrics` to a short vocal marker instead of leaving the default
`[Instrumental]` marker in place.

`anvil cloud search` uses GPUFindr's read-only public GPU catalog to show live
provider availability and pricing before you create an account somewhere.
`anvil cloud runpod launch` uses RunPod's GraphQL pod API; keep `--dry-run` on
until the request looks right, then remove it to create a pod. If you create a
RunPod account from the docs, this referral link helps offset Anvil training
costs: [runpod.io?ref=sox5p475](https://runpod.io?ref=sox5p475). The default
RunPod launch uses template `runpod-torch-v280`, matching RunPod's Torch 2.8
deploy URL. Add `--minimal` if RunPod reports supply constraints even though
the UI shows featured GPUs. For `--gpu-type`, use the `gpuId` value from
`runpodctl gpu list`; for example, H200 SXM is `NVIDIA H200`. Use
`anvil cloud runpod terminate POD_ID` when the run is done.

Cloud bootstrap reuses the provider image's CUDA/PyTorch stack via a
system-site-packages venv, then installs Anvil and ACE-Step without letting
ACE-Step replace the image's torch build. This avoids torch resolver conflicts
on managed GPU images. Remote uploads preserve `.venv`, `work`, `outputs`, and
`logs` so rerunning with `--skip-bootstrap` does not delete the remote runtime
state. Bootstrap also ensures the required ACE-Step checkpoints exist before
training: the shared main bundle for `vae`/text components plus the selected
DiT variant, such as `acestep-v15-sft` for `--model-variant sft`.

---
