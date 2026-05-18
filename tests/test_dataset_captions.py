import json

from anvil_audio.dataset_captions import (
    CaptionAuditConfig,
    audit_or_repair_captions,
)


def test_caption_audit_reports_exact_duplicates_and_low_confidence(tmp_path):
    dataset = _write_caption_dataset(tmp_path)

    result = audit_or_repair_captions(CaptionAuditConfig(dataset_dir=dataset))

    assert result.exact_duplicate_groups == 1
    assert result.duplicate_record_count == 2
    assert result.low_confidence_count == 2
    assert result.repaired_count == 0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["summary"]["duplicate_record_count"] == 2


def test_caption_repair_dry_run_does_not_write(tmp_path):
    dataset = _write_caption_dataset(tmp_path)

    result = audit_or_repair_captions(
        CaptionAuditConfig(
            dataset_dir=dataset,
            mode="heuristic",
            style_hint="dark blues, slow guitar",
            write=False,
        )
    )

    assert result.repaired_count == 2
    assert result.warnings == ["Dry run only; pass --write to update captions.json."]
    captions = json.loads((dataset / "captions.json").read_text(encoding="utf-8"))
    assert captions[0]["caption"] == "same caption"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["duplicates"][0]["caption"] == "same caption"
    assert report["proposed_repairs"][0]["before"] == "same caption"
    assert report["proposed_repairs"][0]["after"] != "same caption"


def test_caption_repair_updates_captions_and_sidecars(tmp_path):
    dataset = _write_caption_dataset(tmp_path)

    result = audit_or_repair_captions(
        CaptionAuditConfig(
            dataset_dir=dataset,
            mode="heuristic",
            style_hint="dark blues, slow guitar",
            write=True,
        )
    )

    assert result.repaired_count == 2
    captions = json.loads((dataset / "captions.json").read_text(encoding="utf-8"))
    assert captions[0]["caption"] != "same caption"
    assert "slow guitar" in captions[0]["caption"]
    assert "playlist" not in captions[0]["caption"]
    assert captions[0]["confidence"] == 0.68
    assert captions[0]["caption_history"][0]["mode"] == "heuristic"

    sidecar = json.loads((dataset / "clips/clip_0001.json").read_text(encoding="utf-8"))
    assert sidecar["caption"] == captions[0]["caption"]

    manifest = json.loads(
        (dataset / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["caption_repair"]["repaired_count"] == 2


def _write_caption_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    clips.mkdir(parents=True)
    for index in (1, 2):
        (clips / f"clip_{index:04d}.wav").write_bytes(b"wav")
        (clips / f"clip_{index:04d}.json").write_text(
            json.dumps({"caption": "same caption"}),
            encoding="utf-8",
        )
    captions = [
        {
            "file": "clips/clip_0001.wav",
            "caption": "same caption",
            "prompt": "same caption",
            "tags": [],
            "negative_tags": ["muddy mix"],
            "confidence": 0.2,
            "analysis": {
                "energy": "medium",
                "brightness": "dark",
                "bass_character": "balanced low end",
                "density": "steady",
                "stereo_character": "wide stereo",
                "tempo_bpm_estimate": 72,
            },
            "source": {"title": "raw guitar midnight"},
            "separation": {
                "stems": {
                    "instrumental": "stems/clip_0001/instrumental.wav",
                    "vocals": "stems/clip_0001/vocals.wav",
                }
            },
        },
        {
            "file": "clips/clip_0002.wav",
            "caption": "same caption",
            "prompt": "same caption",
            "tags": [],
            "negative_tags": ["muddy mix"],
            "confidence": 0.2,
            "analysis": {
                "energy": "low",
                "brightness": "balanced",
                "bass_character": "bass-heavy",
                "density": "dense",
                "stereo_character": "centered stereo",
            },
            "source": {"title": "storm vocal blues"},
        },
    ]
    (dataset / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"name": "caption_test"}), encoding="utf-8"
    )
    return dataset
