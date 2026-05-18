"""Training-bundle export for Anvil datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BUNDLE_VERSION = "1.0"
DEFAULT_INCLUDE = ("full-mix",)


@dataclass(slots=True)
class TrainingBundleConfig:
    """Configuration for exporting a portable dataset training bundle."""

    dataset_dir: Path
    profile: str = "acestep-lora"
    include: tuple[str, ...] = DEFAULT_INCLUDE
    output: Path | None = None
    strict: bool = False


@dataclass(slots=True)
class TrainingBundleResult:
    """Training-bundle export result."""

    dataset_dir: Path
    bundle_path: Path
    clip_count: int
    asset_count: int
    warnings: list[str] = field(default_factory=list)


def export_training_bundle(config: TrainingBundleConfig) -> TrainingBundleResult:
    """Write `training_bundle.json` for a dataset and its selected assets."""
    dataset_dir = config.dataset_dir.expanduser().resolve()
    captions_path = dataset_dir / "captions.json"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not captions_path.is_file():
        raise FileNotFoundError(f"Missing captions.json: {captions_path}")

    records = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No caption records found in {captions_path}")

    manifest = _load_json_object(manifest_path)
    include = _normalize_include(config.include)
    warnings: list[str] = []
    clips: list[dict[str, Any]] = []
    asset_count = 0
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        clip_payload = _bundle_clip(
            dataset_dir=dataset_dir,
            index=index,
            record=record,
            include=include,
            warnings=warnings,
            strict=config.strict,
        )
        asset_count += len(clip_payload["assets"])
        clips.append(clip_payload)

    bundle = {
        "anvil_training_bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": config.profile,
        "dataset_name": manifest.get("name") or dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "include": list(include),
        "clip_count": len(clips),
        "asset_count": asset_count,
        "warnings": warnings,
        "clips": clips,
    }
    output = (
        config.output.expanduser().resolve()
        if config.output
        else dataset_dir / "training_bundle.json"
    )
    _write_json(output, bundle)
    return TrainingBundleResult(
        dataset_dir=dataset_dir,
        bundle_path=output,
        clip_count=len(clips),
        asset_count=asset_count,
        warnings=warnings,
    )


def parse_include(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated include list."""
    if not value:
        return DEFAULT_INCLUDE
    return _normalize_include(tuple(piece.strip() for piece in value.split(",")))


def _bundle_clip(
    *,
    dataset_dir: Path,
    index: int,
    record: dict[str, Any],
    include: tuple[str, ...],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    file_value = str(record.get("file") or "")
    assets: dict[str, str] = {}
    if "full-mix" in include:
        _add_asset(
            assets,
            name="full-mix",
            file_value=file_value,
            dataset_dir=dataset_dir,
            warnings=warnings,
            strict=strict,
        )

    separation = record.get("separation")
    stems = separation.get("stems") if isinstance(separation, dict) else {}
    if not isinstance(stems, dict):
        stems = {}
    for name in include:
        if name == "full-mix":
            continue
        _add_asset(
            assets,
            name=name,
            file_value=_stem_file(stems.get(name)),
            dataset_dir=dataset_dir,
            warnings=warnings,
            strict=strict,
        )

    return {
        "id": f"clip_{index:04d}",
        "file": file_value,
        "caption": str(record.get("caption") or record.get("prompt") or ""),
        "prompt": str(record.get("prompt") or record.get("caption") or ""),
        "tags": _as_string_list(record.get("tags")),
        "negative_tags": _as_string_list(record.get("negative_tags")),
        "confidence": _as_float(record.get("confidence"), default=0.0),
        "seconds_start": _as_float(record.get("seconds_start"), default=0.0),
        "seconds_total": _as_float(record.get("seconds_total"), default=0.0),
        "assets": assets,
    }


def _add_asset(
    assets: dict[str, str],
    *,
    name: str,
    file_value: str,
    dataset_dir: Path,
    warnings: list[str],
    strict: bool,
) -> None:
    if not file_value:
        message = f"missing asset path for {name}"
        if strict:
            raise FileNotFoundError(message)
        warnings.append(message)
        return
    path = dataset_dir / file_value
    if not path.is_file():
        message = f"missing asset file for {name}: {file_value}"
        if strict:
            raise FileNotFoundError(message)
        warnings.append(message)
        return
    assets[name] = file_value


def _normalize_include(include: tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    allowed = {"full-mix", "vocals", "instrumental", "drums", "bass", "other"}
    for item in include:
        value = item.strip().lower()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(
                f"Unsupported training bundle include asset: {item}. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )
        normalized.append(value)
    return tuple(dict.fromkeys(normalized)) or DEFAULT_INCLUDE


def _stem_file(stem_payload: Any) -> str:
    if isinstance(stem_payload, str):
        return stem_payload
    if isinstance(stem_payload, dict):
        return str(stem_payload.get("file") or "")
    return ""


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
