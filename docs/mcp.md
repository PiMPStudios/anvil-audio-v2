# MCP Server

Configure Anvil Audio as an MCP server for Claude and other MCP clients.

Anvil exposes its full capabilities as an [MCP](https://modelcontextprotocol.io) server so
Claude and other MCP clients can generate and edit audio directly without the Gradio UI or
manual CLI commands.

The MCP runtime is included in the default install. If you installed with
`bash install.sh` or `pip install -e .`, the server is ready to run with:

```bash
.venv/bin/python -m anvil_audio.mcp_server
```

For ACE-Step LoRA generation on Apple Silicon, start the MCP server with
`--no-mlx-dit`. PEFT/LoKr adapters currently patch ACE-Step's PyTorch DiT, so
the server needs the PyTorch DiT backend when a LoRA is selected:

```bash
.venv/bin/python -m anvil_audio.mcp_server --no-mlx-dit
```

Use `--use-mlx-dit` to explicitly force the native MLX DiT/VAE path for normal
non-LoRA ACE-Step generation. These flags set `ANVIL_ACESTEP_USE_MLX_DIT` for
the MCP server process before any model is loaded.

## Available tools

| Tool | What it does |
|---|---|
| `prepare_music_prompt` | Enhance prompt, negatives, and lyrics |
| `generate_audio` | Generate a clip from a prompt; auto-selects model if not specified |
| `batch_generate` | Generate multiple clips in one call |
| `edit_audio` | Post-process a file with normalize, trim, EQ, reverb, etc. |
| `list_lora_adapters` | List registered LoRA adapters for ACE-Step generation |
| `list_models` | All registered models with type, limits, and loaded status |
| `get_model_info` | Full details for one model |
| `get_memory_status` | Report MCP process memory, accelerator cache stats, loaded models, and LoRA state |
| `unload_models` | Drop cached model pipelines from the MCP process and flush memory caches |
| `list_recent_outputs` | Recent output files with their metadata, newest-first |
| `get_generation_metadata` | Read the sidecar for any output file |
| `list_projects` | Project folders under `~/anvil-audio-outputs/` |
| `set_active_project` | Set a default project so you don't repeat it every call |

All `generate_audio` and `batch_generate` responses include `generation_duration_seconds` —
the wall-clock time from the start of inference to the file being written. This lets you
compare backends directly (e.g. PyTorch MPS vs MLX) without any external timing.

ACE-Step generation tools also accept `lora`, `lora_scale`, and
`lora_adapter_name`. They also accept per-call `use_mlx_dit`. Use
`list_lora_adapters` to discover registered adapter IDs, then pass the adapter
id or a direct PEFT/LoKr path:

```text
generate_audio(
  prompt="dark blues noir rock, raw guitar, smoky male vocal",
  model="acestep-v1.5-sft",
  lora="dark-blues-h200-sft",
  lora_scale=0.75,
  use_mlx_dit=false,
  negative_prompt="muddy mix, harsh treble, weak drums"
)
```

For LoRA calls, MCP automatically defaults `use_mlx_dit` to `false` when you do
not specify it, because current PEFT/LoKr adapters apply to ACE-Step's PyTorch
DiT. Explicitly pass `use_mlx_dit=true` only for non-LoRA ACE-Step calls where
you want the native MLX DiT/VAE backend.

Models are loaded lazily on first use and cached between calls. The MCP server
keeps a small LRU cache of loaded pipelines so desktop memory does not climb
forever as you switch models. The default keeps two resident pipelines; set
`ANVIL_MCP_MAX_PIPELINES=0` to disable automatic eviction, or set a different
count if your machine has more headroom. You can also set
`ANVIL_MCP_IDLE_TIMEOUT_SECONDS` to evict models that have not been used for a
while. ACE-Step keeps separate cached instances for the default, MLX DiT, and
PyTorch DiT backend variants when those variants are requested.

If an ACE-Step pipeline has a LoRA loaded, a later MCP generation with no
`lora` argument explicitly disables the active adapter before generating. This
keeps A/B tests honest when comparing a tuned adapter against the base model.
Use `get_memory_status(flush=true)` to inspect loaded model/backends and clear
unused torch/MLX caches without unloading weights. Use `unload_models()` to
drop all cached pipelines, or target one backend:

```text
unload_models(model="acestep-v1.5-xl-sft", backend="torch_dit")
```

## Claude Desktop config

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`
(create the file if it doesn't exist):

```json
{
  "mcpServers": {
    "anvil-audio": {
      "command": "/path/to/anvil-audio-v2/.venv/bin/python",
      "args": ["-m", "anvil_audio.mcp_server", "--no-mlx-dit"]
    }
  }
}
```

Replace `/path/to/anvil-audio-v2` with the absolute path to your clone. Remove
`--no-mlx-dit` if you do not need ACE-Step LoRA generation through MCP.

## Claude Code config

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "anvil-audio": {
      "command": "/path/to/anvil-audio-v2/.venv/bin/python",
      "args": ["-m", "anvil_audio.mcp_server", "--no-mlx-dit"],
      "type": "stdio"
    }
  }
}
```

## Example session

Once configured, Claude can generate and edit audio directly:

```text
You:    Generate a short thunderstorm ambience clip
Claude: [calls generate_audio(prompt="thunderstorm ambience, rain, distant thunder", duration_seconds=20)]
        Generated: ~/anvil-audio-outputs/default/20260401_181907_thunderstorm_...wav
        Generation time: 31.2 s

You:    Add a slight fade in and normalize it to -14 LUFS
Claude: [calls edit_audio(file_path="...", fade_in=2.0, normalize=True,
                          normalize_target_db=-14, normalize_lufs=True)]
        Exported: ~/anvil-audio-outputs/default/20260401_181942_edit_...wav
```

---
