import json
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
    monkeypatch.setattr(mcp_server, "_get_pipeline", lambda _name: fake_pipeline)

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
    assert fake_pipeline.lora_calls[0][1:] == ("style", 0.75)
    metadata = result["metadata"]
    assert metadata["extra"]["lora"]["registry_id"] == "dark-blues"
    assert metadata["extra"]["lora"]["scale"] == 0.75
    assert metadata["extra"]["lyrics"] == "[Instrumental]"
