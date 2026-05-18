import json

from anvil_audio.dataset_bundle import (
    TrainingBundleConfig,
    export_training_bundle,
    parse_include,
)


def test_export_training_bundle_includes_full_mix_and_stems(tmp_path):
    dataset = _write_bundle_dataset(tmp_path)

    result = export_training_bundle(
        TrainingBundleConfig(
            dataset_dir=dataset,
            include=("full-mix", "instrumental", "vocals"),
            strict=True,
        )
    )

    assert result.bundle_path == dataset / "training_bundle.json"
    assert result.clip_count == 1
    assert result.asset_count == 3

    bundle = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    assert bundle["anvil_training_bundle_version"] == "1.0"
    assert bundle["profile"] == "acestep-lora"
    assert bundle["include"] == ["full-mix", "instrumental", "vocals"]
    assert bundle["clips"][0]["assets"] == {
        "full-mix": "clips/clip_0001.wav",
        "instrumental": "stems/clip_0001/instrumental.wav",
        "vocals": "stems/clip_0001/vocals.wav",
    }


def test_export_training_bundle_warns_for_missing_optional_assets(tmp_path):
    dataset = _write_bundle_dataset(tmp_path)

    result = export_training_bundle(
        TrainingBundleConfig(dataset_dir=dataset, include=("drums",))
    )

    assert result.asset_count == 0
    assert result.warnings == ["missing asset path for drums"]


def test_parse_include_normalizes_and_dedupes():
    assert parse_include("full-mix, vocals, vocals") == ("full-mix", "vocals")


def _write_bundle_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    stems = dataset / "stems/clip_0001"
    clips.mkdir(parents=True)
    stems.mkdir(parents=True)
    (clips / "clip_0001.wav").write_bytes(b"wav")
    (stems / "instrumental.wav").write_bytes(b"wav")
    (stems / "vocals.wav").write_bytes(b"wav")
    captions = [
        {
            "file": "clips/clip_0001.wav",
            "caption": "dark blues guitar vocal",
            "prompt": "dark blues guitar vocal",
            "tags": ["dark blues", "guitar"],
            "negative_tags": ["muddy mix"],
            "confidence": 0.8,
            "seconds_start": 0.0,
            "seconds_total": 10.0,
            "separation": {
                "stems": {
                    "instrumental": "stems/clip_0001/instrumental.wav",
                    "vocals": "stems/clip_0001/vocals.wav",
                }
            },
        }
    ]
    (dataset / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"name": "dark_blues"}), encoding="utf-8"
    )
    return dataset
