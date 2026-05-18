"""Caption audit and repair helpers for Anvil datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CaptionRepairMode = Literal["audit", "heuristic"]


@dataclass(slots=True)
class CaptionAuditConfig:
    """Configuration for caption audit/repair."""

    dataset_dir: Path
    mode: CaptionRepairMode = "audit"
    output: Path | None = None
    write: bool = False
    style_hint: str = ""
    min_confidence: float = 0.62


@dataclass(slots=True)
class CaptionAuditResult:
    """Caption audit/repair result."""

    dataset_dir: Path
    report_path: Path
    exact_duplicate_groups: int
    duplicate_record_count: int
    low_confidence_count: int
    repaired_count: int = 0
    warnings: list[str] = field(default_factory=list)


def audit_or_repair_captions(config: CaptionAuditConfig) -> CaptionAuditResult:
    """Audit captions and optionally rewrite low-information duplicate captions."""
    dataset_dir = config.dataset_dir.expanduser().resolve()
    captions_path = dataset_dir / "captions.json"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not captions_path.is_file():
        raise FileNotFoundError(f"Missing captions.json: {captions_path}")
    records = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No caption records found in {captions_path}")

    normalized = [record for record in records if isinstance(record, dict)]
    duplicate_groups = _duplicate_groups(normalized)
    low_confidence = [
        record
        for record in normalized
        if _as_float(record.get("confidence"), default=0.0) < config.min_confidence
    ]
    repaired_count = 0
    warnings: list[str] = []

    if config.mode == "heuristic":
        for index, record in enumerate(normalized, start=1):
            if not _should_repair(record, duplicate_groups, config):
                continue
            caption_payload = _repaired_caption(record, config.style_hint)
            record["caption"] = caption_payload["caption"]
            record["prompt"] = caption_payload["caption"]
            record["tags"] = caption_payload["tags"]
            record["negative_tags"] = caption_payload["negative_tags"]
            record["confidence"] = caption_payload["confidence"]
            record.setdefault("caption_history", []).append(
                {
                    "updated_at": datetime.now(UTC).isoformat(),
                    "mode": "heuristic",
                    "reason": "duplicate_or_low_confidence",
                    "index": index,
                }
            )
            _update_clip_sidecar(dataset_dir, record)
            repaired_count += 1
        if config.write:
            _write_json(captions_path, records)
            manifest = _load_json_object(manifest_path)
            manifest["caption_repair"] = {
                "updated_at": datetime.now(UTC).isoformat(),
                "mode": config.mode,
                "repaired_count": repaired_count,
                "style_hint": config.style_hint,
            }
            _write_json(manifest_path, manifest)
        elif repaired_count:
            warnings.append("Dry run only; pass --write to update captions.json.")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_dir": str(dataset_dir),
        "mode": config.mode,
        "write": config.write,
        "clip_count": len(normalized),
        "summary": {
            "exact_duplicate_groups": len(duplicate_groups),
            "duplicate_record_count": sum(len(group) for group in duplicate_groups),
            "low_confidence_count": len(low_confidence),
            "repaired_count": repaired_count,
        },
        "duplicates": [
            {
                "caption": group[0]["caption"],
                "count": len(group),
                "files": [str(item.get("file") or "") for item in group[:20]],
            }
            for group in duplicate_groups[:20]
        ],
        "low_confidence": [
            {
                "file": str(record.get("file") or ""),
                "caption": str(record.get("caption") or ""),
                "confidence": _as_float(record.get("confidence"), default=0.0),
            }
            for record in low_confidence[:40]
        ],
        "warnings": warnings,
    }
    report_path = (
        config.output.expanduser().resolve()
        if config.output
        else dataset_dir / "caption_audit_report.json"
    )
    _write_json(report_path, report)
    return CaptionAuditResult(
        dataset_dir=dataset_dir,
        report_path=report_path,
        exact_duplicate_groups=len(duplicate_groups),
        duplicate_record_count=sum(len(group) for group in duplicate_groups),
        low_confidence_count=len(low_confidence),
        repaired_count=repaired_count,
        warnings=warnings,
    )


def _duplicate_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = _normalize_caption(str(record.get("caption") or record.get("prompt") or ""))
        if not key:
            continue
        buckets.setdefault(key, []).append(record)
    return [group for group in buckets.values() if len(group) > 1]


def _should_repair(
    record: dict[str, Any],
    duplicate_groups: list[list[dict[str, Any]]],
    config: CaptionAuditConfig,
) -> bool:
    if _as_float(record.get("confidence"), default=0.0) < config.min_confidence:
        return True
    duplicate_ids = {id(item) for group in duplicate_groups for item in group}
    return id(record) in duplicate_ids


def _repaired_caption(record: dict[str, Any], style_hint: str) -> dict[str, Any]:
    analysis = _analysis(record)
    separation = record.get("separation") if isinstance(record.get("separation"), dict) else {}
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    title = str(source.get("title") or Path(str(record.get("file") or "")).stem)
    tags = []
    tags.extend(_tags_from_text(style_hint))
    tags.extend(_tags_from_text(title))
    tags.extend(_audio_tags(analysis))
    tags.extend(_stem_tags(separation))
    tags = _dedupe(tags)[:12]
    caption = ", ".join(tags or ["audio training clip"])
    if title and title not in caption:
        caption = f"{caption}, source {title[:60]}"
    return {
        "caption": _clean_caption(caption),
        "tags": tags,
        "negative_tags": _negative_tags(record, analysis),
        "confidence": 0.68 if tags else 0.52,
    }


def _analysis(record: dict[str, Any]) -> dict[str, Any]:
    analysis = record.get("analysis")
    return analysis if isinstance(analysis, dict) else {}


def _audio_tags(analysis: dict[str, Any]) -> list[str]:
    tags = []
    for field_name in (
        "energy",
        "brightness",
        "bass_character",
        "density",
        "stereo_character",
    ):
        value = str(analysis.get(field_name) or "").strip()
        if value:
            tags.append(value if field_name != "brightness" else f"{value} tone")
    tempo = analysis.get("tempo_bpm_estimate")
    if tempo:
        tags.append(f"around {tempo} bpm")
    return tags


def _stem_tags(separation: Any) -> list[str]:
    if not isinstance(separation, dict):
        return []
    stems = separation.get("stems")
    if not isinstance(stems, dict):
        return []
    tags = []
    if "instrumental" in stems:
        tags.append("instrumental stem available")
    if "vocals" in stems:
        tags.append("vocal stem available")
    if {"drums", "bass", "other"}.intersection(stems):
        tags.append("multi-stem separated")
    return tags


def _negative_tags(record: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    negative = _as_string_list(record.get("negative_tags"))
    if analysis.get("peak_db", -99) >= -0.2:
        negative.append("possible clipping")
    return _dedupe(negative or ["muddy mix", "noisy artifacts"])


def _update_clip_sidecar(dataset_dir: Path, record: dict[str, Any]) -> None:
    file_value = str(record.get("file") or "")
    if not file_value:
        return
    sidecar_path = (dataset_dir / file_value).with_suffix(".json")
    if not sidecar_path.is_file():
        return
    sidecar = _load_json_object(sidecar_path)
    sidecar.update(
        {
            "caption": record.get("caption", ""),
            "prompt": record.get("prompt", record.get("caption", "")),
            "tags": record.get("tags", []),
            "negative_tags": record.get("negative_tags", []),
            "confidence": record.get("confidence", 0.0),
        }
    )
    _write_json(sidecar_path, sidecar)


def _tags_from_text(text: str) -> list[str]:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    pieces = []
    for token in lowered.replace("|", ",").replace("/", ",").split(","):
        cleaned = " ".join(token.split()).strip()
        if 3 <= len(cleaned) <= 36 and any(ch.isalpha() for ch in cleaned):
            pieces.append(cleaned)
    keywords = [
        "ambient",
        "blues",
        "cinematic",
        "dark",
        "guitar",
        "instrumental",
        "rock",
        "slow",
        "vocal",
    ]
    pieces.extend(keyword for keyword in keywords if keyword in lowered)
    return _dedupe(pieces)


def _normalize_caption(text: str) -> str:
    return " ".join(text.lower().split())


def _clean_caption(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip(" ,.;")[:600]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
