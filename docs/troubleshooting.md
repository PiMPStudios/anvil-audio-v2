# Troubleshooting

Common setup and runtime warnings.

`anvil setup` is the first thing to run when something looks off. It reports
whether ACE-Step, MLX Stable Audio, local prompt intelligence, and MLX vocal
transcription are importable in the current virtual environment.

Common startup warnings:

- `bitsandbytes not installed. Using standard AdamW.` is expected on macOS and
  only affects optimizer selection for training.
- `torchao` compatibility warnings can be ignored unless you are actively using
  ACE-Step quantization.
- `mx.metal.device_info is deprecated` is an upstream MLX warning and does not
  indicate failed generation.

If an XL model refuses to load, install the checkpoint explicitly with
`acestep-download --dir "$HOME/.cache/anvil-audio/acestep/checkpoints" --model <checkpoint>`.
Anvil blocks surprise XL downloads because those checkpoints are large.

For long Gradio sessions, Anvil flushes unused Python, torch, and MLX caches
before generation and after large outputs. The default large-output cleanup
threshold is `128` MB. Tune it with `ANVIL_GRADIO_LARGE_OUTPUT_CLEANUP_MB`, or
set it to `0` to disable that trigger. Optional pressure thresholds use the
`ANVIL_GRADIO_MEMORY_*` prefix, for example
`ANVIL_GRADIO_MEMORY_RSS_LIMIT_MB` or
`ANVIL_GRADIO_MEMORY_SYSTEM_AVAILABLE_LIMIT_MB`.

---
