import json
from pathlib import Path

from anvil_audio.lora import (
    detect_adapter_format,
    import_local_adapter,
    list_adapters,
    resolve_adapter_reference,
    resolve_lora_stack,
)
from anvil_audio.lora_training import (
    LoRATrainConfig,
    adapter_output_is_finite,
    adapter_output_exists,
    build_train_command,
    validate_preprocessed_tensors,
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


def test_resolve_lora_stack_accepts_primary_and_text_specs(monkeypatch, tmp_path):
    monkeypatch.setenv("ANVIL_AUDIO_LORA_DIR", str(tmp_path / "lora-cache"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fake_peft_adapter(first)
    _write_fake_peft_adapter(second)
    import_local_adapter(first, name="Lead Style")
    import_local_adapter(second, name="Room Tone")

    stack = resolve_lora_stack(
        "lead-style",
        primary_scale=0.8,
        primary_adapter_name="style",
        stack="room-tone:0.25",
    )

    assert [item.registry_id for item in stack] == ["lead-style", "room-tone"]
    assert [item.adapter_name for item in stack] == ["style", "room-tone"]
    assert [item.scale for item in stack] == [0.8, 0.25]


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
    assert payload["samples"][0]["filename"] == "00001_clip_0001.wav"
    assert payload["samples"][0]["audio_path"].endswith(
        ".acestep_audio/00001_clip_0001.wav"
    )
    assert Path(payload["samples"][0]["audio_path"]).is_file()
    assert payload["samples"][0]["lyrics"] == "[Instrumental]"


def test_write_acestep_dataset_json_uses_unique_sample_names(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "stems/clip_0001").mkdir(parents=True)
    (dataset / "stems/clip_0002").mkdir(parents=True)
    (dataset / "stems/clip_0001/instrumental.wav").write_bytes(b"RIFF")
    (dataset / "stems/clip_0002/instrumental.wav").write_bytes(b"RIFF")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"name": "test_style"}),
        encoding="utf-8",
    )
    (dataset / "captions.json").write_text(
        json.dumps(
            [
                {"file": "stems/clip_0001/instrumental.wav", "caption": "first"},
                {"file": "stems/clip_0002/instrumental.wav", "caption": "second"},
            ]
        ),
        encoding="utf-8",
    )

    output = write_acestep_dataset_json(dataset)
    samples = json.loads(output.read_text(encoding="utf-8"))["samples"]

    assert [sample["filename"] for sample in samples] == [
        "00001_instrumental.wav",
        "00002_instrumental.wav",
    ]
    assert samples[0]["audio_path"].endswith(
        ".acestep_audio/00001_instrumental.wav"
    )
    assert samples[1]["audio_path"].endswith(
        ".acestep_audio/00002_instrumental.wav"
    )
    assert Path(samples[0]["audio_path"]).is_file()
    assert Path(samples[1]["audio_path"]).is_file()


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


def test_build_train_command_maps_xl_alias_for_acestep_cli(tmp_path):
    config = LoRATrainConfig(
        tensor_dir=tmp_path / "tensors",
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "checkpoints",
        model_variant="xl_sft",
    )

    command = build_train_command(config)

    assert command[command.index("--model-variant") + 1] == "acestep-v15-xl-sft"
    assert command[command.index("--base-model") + 1] == "xl_sft"


def test_build_train_command_can_force_basic_loop(tmp_path):
    config = LoRATrainConfig(
        tensor_dir=tmp_path / "tensors",
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "checkpoints",
        basic_loop=True,
    )

    command = build_train_command(config)

    assert command[1] == "-c"
    assert "trainer_fixed._FABRIC_AVAILABLE = False" in command[2]
    assert "--dataset-dir" in command


def test_adapter_output_exists_detects_peft_final(tmp_path):
    final = tmp_path / "out" / "final"
    _write_fake_peft_adapter(final)

    assert adapter_output_exists(tmp_path / "out") is True
    assert adapter_output_exists(tmp_path / "missing") is False


def test_adapter_output_is_finite_rejects_nan_weights(tmp_path):
    import torch
    from safetensors.torch import save_file

    final = tmp_path / "out" / "final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_file({"ok": torch.ones(1)}, final / "adapter_model.safetensors")

    assert adapter_output_is_finite(tmp_path / "out") is True

    save_file({"bad": torch.tensor([float("nan")])}, final / "adapter_model.safetensors")

    assert adapter_output_is_finite(tmp_path / "out") is False


def test_validate_preprocessed_tensors_rejects_nonfinite_values(tmp_path):
    import pytest
    import torch

    torch.save({"encoder_hidden_states": torch.ones(1)}, tmp_path / "ok.pt")

    validate_preprocessed_tensors(tmp_path)

    torch.save(
        {"encoder_hidden_states": torch.tensor([float("nan")])},
        tmp_path / "bad.pt",
    )

    with pytest.raises(RuntimeError, match="non-finite tensors"):
        validate_preprocessed_tensors(tmp_path)
