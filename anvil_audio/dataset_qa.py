"""Embedding-based QA reports for Anvil LoRA datasets."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F

DEFAULT_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_INSTRUCTION = ""


@dataclass(slots=True)
class DatasetQAConfig:
    """Configuration for caption-embedding dataset QA."""

    dataset_dir: Path
    output_json: Path | None = None
    output_markdown: Path | None = None
    write_markdown: bool = True
    embedding_model: str | None = None
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 512
    duplicate_threshold: float = 0.9
    cluster_threshold: float = 0.78
    outlier_threshold: float = 0.55
    nearest_neighbors: int = 5
    low_confidence_threshold: float = 0.45
    max_report_items: int = 20
    instruction: str = DEFAULT_EMBEDDING_INSTRUCTION


@dataclass(slots=True)
class DatasetQAResult:
    """Paths and report payload produced by dataset QA."""

    dataset_dir: Path
    json_path: Path
    markdown_path: Path | None
    report: dict[str, Any]


class CaptionEmbedder(Protocol):
    """Small protocol for real or test embedding backends."""

    model_ref: str

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return a 2D tensor of normalized or unnormalized embeddings."""


class QwenCaptionEmbedder:
    """Local Qwen3 caption embedder using HuggingFace Transformers."""

    def __init__(
        self,
        model_ref: str | None = None,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_ref = resolve_embedding_model_ref(model_ref)
        self.device = _resolve_torch_device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed caption texts with Qwen3 last-token pooling."""
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)

        tokenizer, model = self._load()
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            pooled = _last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            batches.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
        return torch.cat(batches, dim=0)

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for dataset QA embeddings. Run "
                "`bash install.sh` or `pip install transformers>=4.51.0`."
            ) from exc

        if _version_tuple(transformers.__version__) < (4, 51, 0):
            raise RuntimeError(
                "Qwen3 embeddings require transformers>=4.51.0. Update with "
                "`pip install -U transformers`."
            )

        model_kwargs: dict[str, Any] = {}
        if self.device.type in {"cuda", "mps"}:
            model_kwargs["torch_dtype"] = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_ref,
            padding_side="left",
        )
        model = AutoModel.from_pretrained(self.model_ref, **model_kwargs)
        model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model


def run_dataset_qa(
    config: DatasetQAConfig,
    *,
    embedder: CaptionEmbedder | None = None,
) -> DatasetQAResult:
    """Run embedding QA for a built Anvil dataset and write reports."""
    dataset_dir = config.dataset_dir.expanduser().resolve()
    records = _load_caption_records(dataset_dir)
    texts = [_embedding_text(record, config.instruction) for record in records]
    embedder = embedder or QwenCaptionEmbedder(
        config.embedding_model,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    embeddings = _normalize_embeddings(embedder.encode(texts))
    report = build_dataset_qa_report(
        records,
        embeddings,
        config,
        model_ref=embedder.model_ref,
    )

    json_path = (
        config.output_json.expanduser().resolve()
        if config.output_json
        else dataset_dir / "dataset_qa_report.json"
    )
    markdown_path = None
    if config.write_markdown:
        markdown_path = (
            config.output_markdown.expanduser().resolve()
            if config.output_markdown
            else dataset_dir / "dataset_qa_report.md"
        )

    _write_json(json_path, report)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_format_markdown_report(report), encoding="utf-8")

    return DatasetQAResult(
        dataset_dir=dataset_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )


def build_dataset_qa_report(
    records: list[dict[str, Any]],
    embeddings: torch.Tensor,
    config: DatasetQAConfig,
    *,
    model_ref: str,
) -> dict[str, Any]:
    """Build a JSON-serializable QA report from records and embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [clips, dimensions]")
    if embeddings.shape[0] != len(records):
        raise ValueError("embedding count must match caption record count")

    similarity = embeddings @ embeddings.T if records else torch.empty((0, 0))
    duplicate_pairs = _duplicate_pairs(records, similarity, config)
    clusters = _clusters(records, similarity, config)
    outliers = _outliers(records, similarity, config)
    low_confidence = _low_confidence_records(records, config)
    top_tags = _top_field_values(records, "tags", limit=12)
    top_negative_tags = _top_field_values(records, "negative_tags", limit=8)
    recommendations = _recommendations(
        records,
        clusters,
        duplicate_pairs,
        outliers,
        low_confidence,
        top_tags,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_dir": str(config.dataset_dir.expanduser().resolve()),
        "embedding_model": model_ref,
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.numel() else 0,
        "clip_count": len(records),
        "thresholds": {
            "duplicate_similarity": config.duplicate_threshold,
            "cluster_similarity": config.cluster_threshold,
            "outlier_neighbor_similarity": config.outlier_threshold,
            "low_confidence": config.low_confidence_threshold,
        },
        "summary": {
            "cluster_count": len(clusters),
            "duplicate_pair_count": len(duplicate_pairs),
            "outlier_count": len(outliers),
            "low_confidence_count": len(low_confidence),
            "average_caption_confidence": _average_confidence(records),
            "top_tags": top_tags,
            "top_negative_tags": top_negative_tags,
        },
        "clusters": clusters,
        "duplicate_pairs": duplicate_pairs[: config.max_report_items],
        "outliers": outliers[: config.max_report_items],
        "low_confidence_clips": low_confidence[: config.max_report_items],
        "recommendations": recommendations,
    }


def resolve_embedding_model_ref(model: str | None = None) -> str:
    """Return a local Qwen path when available, otherwise the HF model id."""
    explicit = (
        model
        or os.environ.get("ANVIL_EMBEDDING_MODEL")
        or os.environ.get("ANVIL_QWEN_EMBEDDING_MODEL")
    )
    if explicit:
        expanded = Path(os.path.expanduser(explicit))
        return str(expanded) if expanded.is_dir() else explicit

    local_qwen = (
        Path.home()
        / ".cache"
        / "anvil-audio"
        / "acestep"
        / "checkpoints"
        / "Qwen3-Embedding-0.6B"
    )
    if local_qwen.is_dir():
        return str(local_qwen)
    return DEFAULT_EMBEDDING_MODEL_ID


def _load_caption_records(dataset_dir: Path) -> list[dict[str, Any]]:
    captions_path = dataset_dir / "captions.json"
    if not captions_path.is_file():
        raise FileNotFoundError(f"Missing captions.json: {captions_path}")
    records = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"captions.json must contain a list: {captions_path}")
    normalized = [record for record in records if isinstance(record, dict)]
    if not normalized:
        raise RuntimeError(f"No caption records found in {captions_path}")
    return normalized


def _embedding_text(record: dict[str, Any], instruction: str) -> str:
    caption = str(record.get("caption") or record.get("prompt") or "").strip()
    tags = ", ".join(_as_string_list(record.get("tags")))
    negative = ", ".join(_as_string_list(record.get("negative_tags")))
    pieces = [f"Caption: {caption or 'audio training clip'}"]
    if tags:
        pieces.append(f"Tags: {tags}")
    if negative:
        pieces.append(f"Negative tags: {negative}")
    text = "\n".join(pieces)
    instruction = " ".join(instruction.split()).strip()
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.detach().cpu().float()
    if embeddings.ndim != 2:
        raise ValueError("embedder must return a 2D embedding tensor")
    return F.normalize(embeddings, p=2, dim=1)


def _duplicate_pairs(
    records: list[dict[str, Any]],
    similarity: torch.Tensor,
    config: DatasetQAConfig,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            score = float(similarity[left_index, right_index].item())
            if score < config.duplicate_threshold:
                continue
            pairs.append(
                {
                    "left_file": _record_file(records[left_index]),
                    "right_file": _record_file(records[right_index]),
                    "similarity": round(score, 4),
                    "left_caption": _record_caption(records[left_index]),
                    "right_caption": _record_caption(records[right_index]),
                }
            )
    return sorted(pairs, key=lambda item: item["similarity"], reverse=True)


def _clusters(
    records: list[dict[str, Any]],
    similarity: torch.Tensor,
    config: DatasetQAConfig,
) -> list[dict[str, Any]]:
    components = _connected_components(similarity, config.cluster_threshold)
    clusters: list[dict[str, Any]] = []
    for cluster_id, indices in enumerate(
        sorted(components, key=lambda group: (-len(group), group[0])),
        start=1,
    ):
        cluster_records = [records[index] for index in indices]
        representative_index = _representative_index(indices, similarity)
        clusters.append(
            {
                "id": cluster_id,
                "size": len(indices),
                "mean_similarity": _mean_pair_similarity(indices, similarity),
                "representative_file": _record_file(records[representative_index]),
                "representative_caption": _record_caption(records[representative_index]),
                "top_tags": _top_field_values(cluster_records, "tags", limit=8),
                "files": [_record_file(record) for record in cluster_records],
            }
        )
    return clusters


def _connected_components(
    similarity: torch.Tensor, threshold: float
) -> list[list[int]]:
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(similarity.shape[0]):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = torch.nonzero(similarity[current] >= threshold).flatten()
            for neighbor in neighbors.tolist():
                if neighbor == current or neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _representative_index(indices: list[int], similarity: torch.Tensor) -> int:
    if len(indices) == 1:
        return indices[0]
    submatrix = similarity[indices][:, indices]
    mean_scores = submatrix.mean(dim=1)
    best = int(torch.argmax(mean_scores).item())
    return indices[best]


def _mean_pair_similarity(indices: list[int], similarity: torch.Tensor) -> float:
    if len(indices) <= 1:
        return 1.0
    scores = [
        float(similarity[left, right].item())
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    ]
    return round(sum(scores) / len(scores), 4)


def _outliers(
    records: list[dict[str, Any]],
    similarity: torch.Tensor,
    config: DatasetQAConfig,
) -> list[dict[str, Any]]:
    if len(records) < 3:
        return []

    outliers: list[dict[str, Any]] = []
    k = max(1, min(config.nearest_neighbors, len(records) - 1))
    for index, record in enumerate(records):
        scores = similarity[index].clone()
        scores[index] = -1.0
        top_scores, top_indices = torch.topk(scores, k=k)
        nearest_mean = float(top_scores.mean().item())
        nearest_score = float(top_scores[0].item())
        if nearest_mean >= config.outlier_threshold:
            continue
        nearest_index = int(top_indices[0].item())
        outliers.append(
            {
                "file": _record_file(record),
                "caption": _record_caption(record),
                "nearest_file": _record_file(records[nearest_index]),
                "nearest_similarity": round(nearest_score, 4),
                "nearest_mean_similarity": round(nearest_mean, 4),
                "tags": _as_string_list(record.get("tags")),
            }
        )
    return sorted(outliers, key=lambda item: item["nearest_mean_similarity"])


def _low_confidence_records(
    records: list[dict[str, Any]],
    config: DatasetQAConfig,
) -> list[dict[str, Any]]:
    low: list[dict[str, Any]] = []
    for record in records:
        confidence = _record_confidence(record)
        if confidence >= config.low_confidence_threshold:
            continue
        low.append(
            {
                "file": _record_file(record),
                "caption": _record_caption(record),
                "confidence": confidence,
                "tags": _as_string_list(record.get("tags")),
            }
        )
    return sorted(low, key=lambda item: item["confidence"])


def _recommendations(
    records: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
    low_confidence: list[dict[str, Any]],
    top_tags: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    if len(records) < 8:
        recommendations.append(
            "Dataset is small; use this as a smoke test before trusting a style LoRA."
        )
    if duplicates:
        recommendations.append(
            f"Review {len(duplicates)} near-duplicate caption pairs before training."
        )
    if outliers:
        recommendations.append(
            f"Review {len(outliers)} semantic outliers that may pull the LoRA off-style."
        )
    if low_confidence:
        recommendations.append(
            f"Improve or remove {len(low_confidence)} low-confidence captions."
        )
    if clusters and len(records) >= 8:
        largest_ratio = clusters[0]["size"] / len(records)
        if largest_ratio >= 0.75:
            recommendations.append(
                "Dataset is heavily concentrated in one cluster; add variety if you "
                "want a broader adapter, or keep it focused for a narrow style LoRA."
            )
    if top_tags:
        coverage = ", ".join(item["value"] for item in top_tags[:5])
        recommendations.append(f"Dominant caption coverage: {coverage}.")
    if not recommendations:
        recommendations.append("No obvious caption embedding QA issues were found.")
    return recommendations


def _format_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dataset QA Report",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Clips: {report['clip_count']}",
        f"- Embedding model: `{report['embedding_model']}`",
        f"- Clusters: {summary['cluster_count']}",
        f"- Duplicate pairs: {summary['duplicate_pair_count']}",
        f"- Outliers: {summary['outlier_count']}",
        f"- Low-confidence captions: {summary['low_confidence_count']}",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.extend(["", "## Clusters", ""])
    for cluster in report["clusters"]:
        lines.append(
            f"- Cluster {cluster['id']}: {cluster['size']} clips, "
            f"mean similarity {cluster['mean_similarity']}, "
            f"representative `{cluster['representative_file']}`"
        )
        lines.append(f"  - {cluster['representative_caption']}")
    lines.extend(["", "## Potential Duplicates", ""])
    if report["duplicate_pairs"]:
        for pair in report["duplicate_pairs"]:
            lines.append(
                f"- `{pair['left_file']}` <> `{pair['right_file']}` "
                f"({pair['similarity']})"
            )
    else:
        lines.append("- None above threshold.")
    lines.extend(["", "## Potential Outliers", ""])
    if report["outliers"]:
        for outlier in report["outliers"]:
            lines.append(
                f"- `{outlier['file']}` nearest `{outlier['nearest_file']}` "
                f"mean {outlier['nearest_mean_similarity']}: {outlier['caption']}"
            )
    else:
        lines.append("- None below threshold.")
    return "\n".join(lines) + "\n"


def _top_field_values(
    records: list[dict[str, Any]], field: str, *, limit: int
) -> list[dict[str, Any]]:
    counts = Counter(
        value
        for record in records
        for value in _as_string_list(record.get(field))
    )
    return [
        {"value": value, "count": count}
        for value, count in counts.most_common(limit)
    ]


def _average_confidence(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return round(sum(_record_confidence(record) for record in records) / len(records), 3)


def _record_confidence(record: dict[str, Any]) -> float:
    try:
        return round(float(record.get("confidence", 0.0)), 3)
    except (TypeError, ValueError):
        return 0.0


def _record_file(record: dict[str, Any]) -> str:
    return str(record.get("file") or record.get("path") or "unknown")


def _record_caption(record: dict[str, Any]) -> str:
    return str(record.get("caption") or record.get("prompt") or "").strip()


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_clean_text(value)] if value.strip() else []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if str(item).strip()]
    return []


def _clean_text(value: Any) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _resolve_torch_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _version_tuple(version: str) -> tuple[int, int, int]:
    pieces = re.split(r"[.+-]", version)
    numbers = []
    for piece in pieces[:3]:
        try:
            numbers.append(int(piece))
        except ValueError:
            numbers.append(0)
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])  # type: ignore[return-value]
