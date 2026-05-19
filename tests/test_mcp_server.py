import json
import importlib
from pathlib import Path

import torch

from anvil_audio import mcp_server
from anvil_audio.core.output import OutputManager
from anvil_audio.lora import import_local_adapter


def _write_fake_peft_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"placeholder")


def test_mcp_lists_registered_lora_adapters(monkeypatch, tmp_path):
    monkeypatch.setenv("ANVIL_AUDIO_LORA_DIR", str(tmp_path / "lora-cache"))
    source = tmp_path / "adapter"
    _write_fake_peft_adapter(source)
    import_local_adapter(source, name="Dark Blues")

    adapters = mcp_server.list_lora_adapters()

    assert adapters[0]["id"] == "dark-blues"
    assert adapters[0]["loadable"] is True
    assert adapters[0]["format"] == "peft"


def test_mcp_generate_applies_lora_and_records_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("ANVIL_AUDIO_LORA_DIR", str(tmp_path / "lora-cache"))
    monkeypatch.setattr(OutputManager, "DEFAULT_BASE", tmp_path / "outputs")
    source = tmp_path / "adapter"
    _write_fake_peft_adapter(source)
    import_local_adapter(source, name="Dark Blues")

    class FakeEntry:
        name = "acestep-v1.5-sft"
        pipeline_type = "acestep"
        max_duration = 60.0

        def resolved_params(self):
            return {"steps": 4, "cfg_scale": 6.0, "sampler_type": "ode"}

    class FakePipeline:
        sample_rate = 10

        def __init__(self):
            self.lora_calls = []

        def apply_lora_adapter(self, path, adapter_name=None, scale=1.0):
            self.lora_calls.append((path, adapter_name, scale))
            return {"enabled": True, "scale": scale}

        def generate(self, *_args, **_kwargs):
            return torch.zeros(1, 1, 20)

    fake_pipeline = FakePipeline()
    monkeypatch.setattr(mcp_server.registry, "get_model", lambda _name: FakeEntry())
    seen_pipeline_kwargs = {}

    def fake_get_pipeline(_name, **kwargs):
        seen_pipeline_kwargs.update(kwargs)
        return fake_pipeline

    monkeypatch.setattr(mcp_server, "_get_pipeline", fake_get_pipeline)

    result = mcp_server._run_generate(
        prompt="dark blues",
        model_name="acestep-v1.5-sft",
        duration=1.0,
        steps=None,
        cfg_scale=None,
        seed=123,
        fmt="wav",
        project="mcp-test",
        lyrics="[Instrumental]",
        negative_prompt="muddy mix",
        lora="dark-blues",
        lora_scale=0.75,
        lora_adapter_name="style",
    )

    assert "error" not in result
    assert seen_pipeline_kwargs["use_mlx_dit"] is False
    assert fake_pipeline.lora_calls[0][1:] == ("style", 0.75)
    metadata = result["metadata"]
    assert metadata["extra"]["lora"]["registry_id"] == "dark-blues"
    assert metadata["extra"]["lora"]["scale"] == 0.75
    assert metadata["extra"]["lyrics"] == "[Instrumental]"
    assert metadata["extra"]["use_mlx_dit"] is False


def test_mcp_generate_accepts_per_call_mlx_dit_override(monkeypatch):
    captured = {}

    def fake_run_generate(**kwargs):
        captured.update(kwargs)
        return {"path": "/tmp/test.wav"}

    monkeypatch.setattr(mcp_server, "_run_generate", fake_run_generate)
    monkeypatch.setattr(mcp_server, "_clamp_duration", lambda _model, duration: duration)

    result = mcp_server.generate_audio(
        prompt="dark blues",
        model="acestep-v1.5-sft",
        use_mlx_dit=False,
    )

    assert result == {"path": "/tmp/test.wav"}
    assert captured["use_mlx_dit"] is False


def test_mcp_pipeline_cache_separates_mlx_dit_backends(monkeypatch):
    calls = []

    class FakeEntry:
        pipeline_type = "acestep"

    def fake_load_pipeline(name, **kwargs):
        calls.append((name, kwargs.get("use_mlx_dit")))
        return object()

    registry_module = importlib.import_module("anvil_audio.core.registry")
    monkeypatch.setattr(mcp_server.registry, "get_model", lambda _name: FakeEntry())
    monkeypatch.setattr(registry_module, "load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(mcp_server, "_pipeline_cache", {})

    first = mcp_server._get_pipeline("acestep-v1.5-sft", use_mlx_dit=False)
    second = mcp_server._get_pipeline("acestep-v1.5-sft", use_mlx_dit=True)
    third = mcp_server._get_pipeline("acestep-v1.5-sft", use_mlx_dit=False)

    assert first is third
    assert first is not second
    assert calls == [
        ("acestep-v1.5-sft", False),
        ("acestep-v1.5-sft", True),
    ]


def test_mcp_server_no_mlx_dit_flag_sets_env(monkeypatch):
    monkeypatch.delenv("ANVIL_ACESTEP_USE_MLX_DIT", raising=False)

    args, remaining = mcp_server._parse_server_args(["--no-mlx-dit", "--extra"])
    mcp_server._apply_server_args(args)

    assert remaining == ["--extra"]
    assert args.no_mlx_dit is True
    assert mcp_server.os.environ["ANVIL_ACESTEP_USE_MLX_DIT"] == "0"


def test_mcp_server_use_mlx_dit_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("ANVIL_ACESTEP_USE_MLX_DIT", "0")

    args, remaining = mcp_server._parse_server_args(["--use-mlx-dit"])
    mcp_server._apply_server_args(args)

    assert remaining == []
    assert args.use_mlx_dit is True
    assert mcp_server.os.environ["ANVIL_ACESTEP_USE_MLX_DIT"] == "1"
