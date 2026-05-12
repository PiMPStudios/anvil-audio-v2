import json
from pathlib import Path

from anvil_audio.lora import (
    detect_adapter_format,
    import_local_adapter,
    list_adapters,
    resolve_adapter_reference,
)
from anvil_audio.lora_training import (
    LoRATrainConfig,
    build_train_command,
    write_acestep_dataset_json,
)


def _write_fake_peft_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"placeholder")


def test_import_local_peft_adapter_registers_loadable_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("ANVIL_AUDIO_LORA_DIR", str(tmp_path / "lora-cache"))
    source = tmp_path / "source_adapter"
    _write_fake_peft_adapter(source)

    entry = import_local_adapter(source, name="My Rock Style")

    assert entry.id == "my-rock-style"
    assert entry.format == "peft"
    assert entry.loadable is True
    assert Path(entry.path).is_dir()
    assert list_adapters()[0].id == entry.id

    resolved, resolved_entry = resolve_adapter_reference("my-rock-style")
    assert resolved == Path(entry.path)
    assert resolved_entry is not None
    assert resolved_entry.id == entry.id


def test_native_anvil_adapter_is_tracked_but_not_loadable(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "adapter.json").write_text("{}", encoding="utf-8")
    (native / "adapter.safetensors").write_bytes(b"placeholder")

    adapter_format, loadable, notes = detect_adapter_format(native)

    assert adapter_format == "anvil-native"
    assert loadable is False
    assert "MLX adapter" in notes[0]


def test_write_acestep_dataset_json_from_anvil_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    clips.mkdir(parents=True)
    (clips / "clip_0001.wav").write_bytes(b"RIFF")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"name": "test_style", "source_reference": "local"}),
        encoding="utf-8",
    )
    (dataset / "captions.json").write_text(
        json.dumps(
            [
                {
                    "file": "clips/clip_0001.wav",
                    "caption": "gritty rock, live drums",
                    "tags": ["rock", "live drums"],
                    "seconds_total": 35.0,
                    "analysis": {"tempo_bpm_estimate": 128},
                }
            ]
        ),
        encoding="utf-8",
    )

    output = write_acestep_dataset_json(dataset, custom_tag="my_style")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["metadata"]["custom_tag"] == "my_style"
    assert payload["samples"][0]["caption"] == "gritty rock, live drums"
    assert payload["samples"][0]["audio_path"].endswith("clip_0001.wav")
    assert payload["samples"][0]["lyrics"] == "[Instrumental]"


def test_build_train_command_targets_fixed_trainer(tmp_path):
    config = LoRATrainConfig(
        tensor_dir=tmp_path / "tensors",
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "checkpoints",
        epochs=3,
        rank=8,
        alpha=16,
    )

    command = build_train_command(config)

    assert command[:3] == [
        command[0],
        "-m",
        "acestep.training_v2.cli.train_fixed",
    ]
    assert "--yes" in command
    assert "--plain" in command
    assert command[command.index("--epochs") + 1] == "3"
    assert command[command.index("--rank") + 1] == "8"
