import json
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from anvil_audio.mlx_lora import (
    apply_lora_stack_to_mlx_decoder,
    load_peft_lora_adapter,
    normalize_peft_module_path,
    parse_peft_lora_key,
)


def test_parse_peft_lora_key_normalizes_acestep_prefixes():
    key = (
        "base_model.model.layers.3.self_attn.q_proj"
        ".lora_A.default.weight"
    )

    assert parse_peft_lora_key(key) == ("layers.3.self_attn.q_proj", "A")
    assert normalize_peft_module_path("base_model.model.decoder.layers.0.self_attn.k_proj") == (
        "layers.0.self_attn.k_proj"
    )


def test_load_peft_lora_adapter_pairs_weights_and_scaling(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "lora_alpha": 8}),
        encoding="utf-8",
    )
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": (
                np.ones((2, 4), dtype=np.float32)
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": (
                np.ones((6, 2), dtype=np.float32)
            ),
            "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": (
                np.zeros((6, 4), dtype=np.float32)
            ),
        },
        adapter_dir / "adapter_model.safetensors",
    )

    adapter = load_peft_lora_adapter(adapter_dir, adapter_name="style", scale=0.5)

    assert adapter.name == "style"
    assert len(adapter.modules) == 1
    module = adapter.modules[0]
    assert module.module_path == "layers.0.self_attn.q_proj"
    assert module.rank == 2
    assert module.alpha == 8
    assert module.scale == 2.0
    assert module.user_scale == 0.5


def test_apply_lora_stack_to_mlx_decoder_wraps_linear_module(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "lora_alpha": 2}),
        encoding="utf-8",
    )
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": (
                np.ones((2, 4), dtype=np.float32)
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": (
                np.ones((6, 2), dtype=np.float32)
            ),
        },
        adapter_dir / "adapter_model.safetensors",
    )
    linear = nn.Linear(4, 6, bias=False)
    linear.weight = mx.zeros((6, 4))
    decoder = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=linear))]
    )

    status = apply_lora_stack_to_mlx_decoder(
        decoder,
        [{"path": str(adapter_dir), "adapter_name": "style", "scale": 0.5}],
    )
    output = decoder.layers[0].self_attn.q_proj(mx.ones((1, 1, 4)))

    assert status["loaded"] is True
    assert status["active"] is True
    assert status["applied_modules"] == 1
    assert mx.allclose(output, mx.full((1, 1, 6), 4.0)).item()


def test_load_peft_lora_adapter_accepts_bfloat16_safetensors(tmp_path):
    import torch
    from safetensors.torch import save_file as save_torch_file

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "lora_alpha": 2}),
        encoding="utf-8",
    )
    save_torch_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                torch.ones((2, 4), dtype=torch.bfloat16)
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": (
                torch.ones((6, 2), dtype=torch.bfloat16)
            ),
        },
        adapter_dir / "adapter_model.safetensors",
    )

    adapter = load_peft_lora_adapter(adapter_dir)

    assert len(adapter.modules) == 1
    assert adapter.modules[0].lora_a.dtype == np.float32
    assert adapter.modules[0].lora_b.dtype == np.float32
