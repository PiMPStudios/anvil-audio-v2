import json
from pathlib import Path

import torch

from anvil_audio.dataset_qa import (
    DatasetQAConfig,
    build_dataset_qa_report,
    run_dataset_qa,
)


class FakeEmbedder:
    model_ref = "fake-qwen"

    def encode(self, texts):
        assert len(texts) == 5
        return torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.99, 0.1],
                [0.0, 0.0, 1.0],
            ]
        )


def test_run_dataset_qa_writes_reports(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "captions.json").write_text(
        json.dumps(_caption_records()),
        encoding="utf-8",
    )

    result = run_dataset_qa(
        DatasetQAConfig(
            dataset_dir=dataset,
            duplicate_threshold=0.93,
            cluster_threshold=0.78,
            outlier_threshold=0.35,
            nearest_neighbors=2,
        ),
        embedder=FakeEmbedder(),
    )

    assert result.json_path == dataset / "dataset_qa_report.json"
    assert result.markdown_path == dataset / "dataset_qa_report.md"
    assert result.json_path.exists()
    assert result.markdown_path and result.markdown_path.exists()

    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report["embedding_model"] == "fake-qwen"
    assert report["summary"]["duplicate_pair_count"] == 2
    assert report["summary"]["outlier_count"] == 1
    assert report["summary"]["low_confidence_count"] == 1
    assert report["duplicate_pairs"][0]["left_file"] == "clips/clip_0001.wav"
    assert report["outliers"][0]["file"] == "clips/clip_0005.wav"


def test_build_dataset_qa_report_groups_caption_clusters():
    records = _caption_records()
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.98, 0.2],
            [0.0, 0.0, 1.0],
        ]
    )

    report = build_dataset_qa_report(
        records,
        embeddings,
        DatasetQAConfig(
            dataset_dir=Path("."),
            duplicate_threshold=0.93,
            cluster_threshold=0.75,
            outlier_threshold=0.35,
            nearest_neighbors=2,
        ),
        model_ref="fake-qwen",
    )

    cluster_sizes = [cluster["size"] for cluster in report["clusters"]]
    assert cluster_sizes == [2, 2, 1]
    assert report["summary"]["top_tags"][0] == {"value": "dark blues", "count": 2}
    assert any("outliers" in item for item in report["recommendations"])

def _caption_records():
    return [
        {
            "file": "clips/clip_0001.wav",
            "caption": "dark blues, smoky male vocal, raw guitar",
            "tags": ["dark blues", "vocal", "guitar"],
            "negative_tags": ["muddy mix"],
            "confidence": 0.82,
        },
        {
            "file": "clips/clip_0002.wav",
            "caption": "dark blues, smoky vocal, expressive guitar bends",
            "tags": ["dark blues", "vocal", "guitar"],
            "negative_tags": ["muddy mix"],
            "confidence": 0.78,
        },
        {
            "file": "clips/clip_0003.wav",
            "caption": "bright synth pop, clean pads, steady beat",
            "tags": ["synth pop", "pads"],
            "negative_tags": ["harsh treble"],
            "confidence": 0.74,
        },
        {
            "file": "clips/clip_0004.wav",
            "caption": "clean synth pads, bright pop texture",
            "tags": ["synth pop", "pads"],
            "negative_tags": ["harsh treble"],
            "confidence": 0.7,
        },
        {
            "file": "clips/clip_0005.wav",
            "caption": "spoken interview, dry podcast room tone",
            "tags": ["spoken"],
            "negative_tags": ["unwanted voices"],
            "confidence": 0.32,
        },
    ]
