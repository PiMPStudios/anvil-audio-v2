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
    proposed_repairs: list[dict[str, Any]] = []
    warnings: list[str] = []

    if config.mode == "heuristic":
        for index, record in enumerate(normalized, start=1):
            if not _should_repair(record, duplicate_groups, config):
                continue
            original_caption = str(record.get("caption") or record.get("prompt") or "")
            caption_payload = _repaired_caption(record, config.style_hint)
            proposed_repairs.append(
                {
                    "file": str(record.get("file") or ""),
                    "before": original_caption,
                    "after": caption_payload["caption"],
                    "confidence": caption_payload["confidence"],
                    "tags": caption_payload["tags"],
                }
            )
            repaired_count += 1
            if config.write:
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
        "proposed_repairs": proposed_repairs[:80],
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
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    title = _useful_title(str(source.get("title") or ""))
    tags = []
    tags.extend(_style_tags(style_hint))
    tags.extend(_tags_from_text(title))
    tags.extend(_audio_tags(analysis))
    tags = _dedupe_meaningful(tags)[:12]
    caption = ", ".join(tags or ["audio training clip"])
    if title:
        title_tags = _dedupe_meaningful(_tags_from_text(title))[:3]
        if title_tags:
            caption = f"{caption}, {', '.join(title_tags)}"
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
    energy = str(analysis.get("energy") or "").strip()
    if energy:
        tags.append(f"{energy} energy")
    brightness = str(analysis.get("brightness") or "").strip()
    if brightness:
        tags.append(f"{brightness} tone")
    bass = str(analysis.get("bass_character") or "").strip()
    if bass:
        tags.append(bass)
    density = str(analysis.get("density") or "").strip()
    if density:
        tags.append(f"{density} rhythm")
    stereo = str(analysis.get("stereo_character") or "").strip()
    if stereo:
        tags.append(stereo)
    tempo = analysis.get("tempo_bpm_estimate")
    if tempo:
        tags.append(f"around {tempo} bpm")
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
    lowered = _clean_source_text(text)
    pieces = []
    for token in lowered.replace("|", ",").replace("/", ",").split(","):
        cleaned = " ".join(token.split()).strip()
        if (
            3 <= len(cleaned) <= 36
            and any(ch.isalpha() for ch in cleaned)
            and not _looks_like_id_or_playlist_junk(cleaned)
        ):
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


def _style_tags(style_hint: str) -> list[str]:
    tags = []
    for item in _tags_from_text(style_hint):
        lowered = item.lower()
        if lowered in {"dark", "blues", "cinematic", "instrumental", "rock", "vocal"}:
            continue
        tags.append(item)
    return tags


def _useful_title(title: str) -> str:
    cleaned = _clean_source_text(title)
    if not cleaned or _looks_like_id_or_playlist_junk(cleaned):
        return ""
    banned = (
        "playlist for",
        "best of",
        "move in silence",
        "youtube",
        "official",
    )
    if any(piece in cleaned for piece in banned):
        return ""
    return cleaned


def _clean_source_text(text: str) -> str:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    return " ".join(lowered.split())


def _looks_like_id_or_playlist_junk(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    if len(words) <= 3 and any(any(ch.isdigit() for ch in word) for word in words):
        return True
    compact = "".join(ch for ch in text if ch.isalnum())
    if len(compact) >= 8:
        upperish = sum(1 for ch in compact if ch.isupper())
        digitish = sum(1 for ch in compact if ch.isdigit())
        if digitish >= 3 or upperish >= 3:
            return True
    return False


def _normalize_caption(text: str) -> str:
    return " ".join(text.lower().split())


def _clean_caption(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip(" ,.;")[:600]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _dedupe_meaningful(values: list[str]) -> list[str]:
    result: list[str] = []
    seen_words: set[str] = set()
    for value in values:
        cleaned = _clean_caption(value.lower())
        if not cleaned:
            continue
        words = set(cleaned.replace("&", " ").split())
        if cleaned in result:
            continue
        if words and words.issubset(seen_words):
            continue
        result.append(cleaned)
        seen_words.update(words)
    return result


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
