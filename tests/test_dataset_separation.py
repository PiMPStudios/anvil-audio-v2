import json
import math
from pathlib import Path

import torch
import torchaudio

from anvil_audio.separation import (
    DatasetSeparationConfig,
    SeparationRequest,
    SeparationResult,
    separate_dataset,
)


class FakeSeparationBackend:
    name = "fake-separator"
    version = "1.0"

    def __init__(self):
        self.calls = 0

    def separate(self, request: SeparationRequest) -> SeparationResult:
        self.calls += 1
        request.output_dir.mkdir(parents=True, exist_ok=True)
        audio, sample_rate = torchaudio.load(str(request.input_path))
        stems = {}
        for stem in ("vocals", "instrumental"):
            output = request.output_dir / f"{stem}.wav"
            torchaudio.save(str(output), audio * 0.5, sample_rate)
            stems[stem] = output
        return SeparationResult(
            backend=self.name,
            backend_version=self.version,
            model=request.model,
            mode=request.mode,
            input_path=request.input_path,
            stems=stems,
            elapsed_seconds=0.25,
        )


def test_separate_dataset_writes_stems_and_metadata(tmp_path):
    dataset = _write_dataset(tmp_path)
    backend = FakeSeparationBackend()

    result = separate_dataset(
        DatasetSeparationConfig(dataset_dir=dataset, mode="instrumental"),
        backend=backend,
    )

    assert backend.calls == 2
    assert len(result.clips) == 2
    assert (dataset / "stems/clip_0001/vocals.wav").exists()
    assert (dataset / "stems/clip_0001/instrumental.wav").exists()

    separation = json.loads(
        (dataset / "stems/clip_0001/separation.json").read_text(encoding="utf-8")
    )
    assert separation["anvil_dataset_version"] == "1.0"
    assert separation["mode"] == "instrumental"
    assert sorted(separation["stems"]) == ["instrumental", "vocals"]
    assert separation["stems"]["vocals"]["analysis"]["duration_seconds"] == 1.0

    sidecar = json.loads(
        (dataset / "clips/clip_0001.json").read_text(encoding="utf-8")
    )
    assert sidecar["separation"]["stems"]["vocals"]["file"].endswith("vocals.wav")

    captions = json.loads((dataset / "captions.json").read_text(encoding="utf-8"))
    assert captions[0]["separation"]["stems"]["instrumental"].endswith(
        "instrumental.wav"
    )

    manifest = json.loads(
        (dataset / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["anvil_dataset_version"] == "1.0"
    assert manifest["separation"]["clip_count"] == 2
    assert manifest["separation"]["backend"] == "fake-separator"


def test_separate_dataset_reuses_cached_stems(tmp_path):
    dataset = _write_dataset(tmp_path)
    backend = FakeSeparationBackend()

    separate_dataset(
        DatasetSeparationConfig(dataset_dir=dataset, mode="instrumental"),
        backend=backend,
    )
    result = separate_dataset(
        DatasetSeparationConfig(dataset_dir=dataset, mode="instrumental"),
        backend=backend,
    )

    assert backend.calls == 2
    assert all(item.result.cached for item in result.clips)


def test_separate_dataset_limit_only_updates_selected_records(tmp_path):
    dataset = _write_dataset(tmp_path)
    backend = FakeSeparationBackend()

    result = separate_dataset(
        DatasetSeparationConfig(dataset_dir=dataset, mode="instrumental", limit=1),
        backend=backend,
    )

    assert len(result.clips) == 1
    captions = json.loads((dataset / "captions.json").read_text(encoding="utf-8"))
    assert "separation" in captions[0]
    assert "separation" not in captions[1]


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    clips.mkdir(parents=True)
    sample_rate = 8_000
    for index, freq in enumerate((220, 330), start=1):
        t = torch.linspace(0, 1.0, sample_rate)
        mono = 0.2 * torch.sin(2 * math.pi * freq * t)
        audio = torch.stack([mono, mono])
        clip = clips / f"clip_{index:04d}.wav"
        torchaudio.save(str(clip), audio, sample_rate)
        (clips / f"clip_{index:04d}.json").write_text(
            json.dumps(
                {
                    "file": f"clips/clip_{index:04d}.wav",
                    "caption": "dark blues guitar vocal",
                }
            ),
            encoding="utf-8",
        )

    captions = [
        {
            "file": "clips/clip_0001.wav",
            "caption": "dark blues guitar vocal",
            "tags": ["dark blues", "guitar", "vocal"],
            "negative_tags": ["muddy mix"],
            "confidence": 0.8,
        },
        {
            "file": "clips/clip_0002.wav",
            "caption": "slow blues guitar atmosphere",
            "tags": ["dark blues", "guitar"],
            "negative_tags": ["muddy mix"],
            "confidence": 0.75,
        },
    ]
    (dataset / "captions.json").write_text(
        json.dumps(captions), encoding="utf-8"
    )
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"name": "dataset", "clips": captions}), encoding="utf-8"
    )
    return dataset
