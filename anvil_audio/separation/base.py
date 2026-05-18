"""Dataset-level source separation workflow."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import torchaudio

from anvil_audio.dataset_builder import analyze_audio_tensor

SeparationMode = Literal["instrumental", "four-stem", "vocals"]

DATASET_VERSION = "1.0"
EXPECTED_STEMS: dict[SeparationMode, tuple[str, ...]] = {
    "instrumental": ("vocals", "instrumental"),
    "four-stem": ("vocals", "drums", "bass", "other"),
    "vocals": ("vocals",),
}


@dataclass(slots=True)
class StemInfo:
    """Metadata for a separated stem."""

    name: str
    file: str
    analysis: dict[str, Any]


@dataclass(slots=True)
class SeparationRequest:
    """One clip separation request."""

    input_path: Path
    output_dir: Path
    mode: SeparationMode = "instrumental"
    model: str = "auto"
    output_format: str = "wav"
    force: bool = False
    model_file_dir: Path | None = None


@dataclass(slots=True)
class SeparationResult:
    """Result from a backend for one clip."""

    backend: str
    backend_version: str
    model: str
    mode: SeparationMode
    input_path: Path
    stems: dict[str, Path]
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    cached: bool = False


@dataclass(slots=True)
class ClipSeparationResult:
    """Dataset-level separation result for one clip."""

    clip_file: str
    sidecar_path: Path
    separation_path: Path
    result: SeparationResult
    stem_info: dict[str, StemInfo]


@dataclass(slots=True)
class DatasetSeparationConfig:
    """Configuration for separating an existing Anvil dataset."""

    dataset_dir: Path
    backend: str = "audio-separator"
    mode: SeparationMode = "instrumental"
    model: str = "auto"
    output_format: str = "wav"
    force: bool = False
    limit: int | None = None
    model_file_dir: Path | None = None


@dataclass(slots=True)
class DatasetSeparationResult:
    """Result from separating a dataset."""

    dataset_dir: Path
    stems_dir: Path
    manifest_path: Path
    clips: list[ClipSeparationResult]


class SeparationBackend(Protocol):
    """Backend protocol for source separation engines."""

    name: str
    version: str

    def separate(self, request: SeparationRequest) -> SeparationResult:
        """Separate one clip into stems."""


def separate_dataset(
    config: DatasetSeparationConfig,
    *,
    backend: SeparationBackend | None = None,
) -> DatasetSeparationResult:
    """Separate clips in an existing Anvil dataset and write stem metadata."""
    dataset_dir = config.dataset_dir.expanduser().resolve()
    captions_path = dataset_dir / "captions.json"
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not captions_path.is_file():
        raise FileNotFoundError(f"Missing captions.json: {captions_path}")

    records = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No caption records found in {captions_path}")

    backend = backend or _load_backend(config.backend)
    stems_dir = dataset_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    updated_records: list[dict[str, Any]] = []
    clip_results: list[ClipSeparationResult] = []
    selected_records = records[: config.limit] if config.limit else records
    selected_files = {
        str(record.get("file") or "")
        for record in selected_records
        if isinstance(record, dict)
    }

    for record in records:
        if not isinstance(record, dict):
            continue
        clip_file = str(record.get("file") or "").strip()
        if not clip_file:
            updated_records.append(record)
            continue
        if clip_file not in selected_files:
            updated_records.append(record)
            continue

        clip_path = (dataset_dir / clip_file).resolve()
        if not clip_path.is_file():
            raise FileNotFoundError(f"Missing clip audio: {clip_path}")

        clip_stem = clip_path.stem
        output_dir = stems_dir / clip_stem
        separator_input, normalization_warning = _separator_input_path(
            clip_path,
            output_dir,
            force=config.force,
        )
        request = SeparationRequest(
            input_path=separator_input,
            output_dir=output_dir,
            mode=config.mode,
            model=config.model,
            output_format=config.output_format,
            force=config.force,
            model_file_dir=config.model_file_dir,
        )
        result = _separate_clip(request, backend)
        if separator_input != clip_path:
            result.input_path = clip_path
        if normalization_warning:
            result.warnings.append(normalization_warning)
        stem_info = _build_stem_info(dataset_dir, result)
        separation_path = output_dir / "separation.json"
        payload = _separation_payload(dataset_dir, result, stem_info)
        _write_json(separation_path, payload)

        sidecar_path = clip_path.with_suffix(".json")
        sidecar = _load_json_object(sidecar_path)
        sidecar["separation"] = payload
        _write_json(sidecar_path, sidecar)

        record["separation"] = {
            "mode": result.mode,
            "backend": result.backend,
            "model": result.model,
            "stems": {
                name: info.file for name, info in sorted(stem_info.items())
            },
            "separation_file": str(separation_path.relative_to(dataset_dir)),
            "cached": result.cached,
            "warnings": result.warnings,
        }
        updated_records.append(record)
        clip_results.append(
            ClipSeparationResult(
                clip_file=clip_file,
                sidecar_path=sidecar_path,
                separation_path=separation_path,
                result=result,
                stem_info=stem_info,
            )
        )

    _write_json(captions_path, updated_records)
    _update_dataset_manifest(
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        config=config,
        backend=backend,
        clip_results=clip_results,
    )
    return DatasetSeparationResult(
        dataset_dir=dataset_dir,
        stems_dir=stems_dir,
        manifest_path=manifest_path,
        clips=clip_results,
    )


def _separate_clip(
    request: SeparationRequest,
    backend: SeparationBackend,
) -> SeparationResult:
    expected = EXPECTED_STEMS[request.mode]
    cached = _cached_result(request, backend, expected)
    if cached is not None:
        return cached
    if request.force and request.output_dir.exists():
        shutil.rmtree(request.output_dir)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    result = backend.separate(request)
    missing = [name for name in expected if name not in result.stems]
    if missing:
        raise RuntimeError(
            "separation backend did not produce expected stems "
            f"for {request.input_path.name}: {', '.join(missing)}"
        )
    return result


def _cached_result(
    request: SeparationRequest,
    backend: SeparationBackend,
    expected_stems: tuple[str, ...],
) -> SeparationResult | None:
    separation_path = request.output_dir / "separation.json"
    if request.force or not separation_path.is_file():
        return None
    payload = _load_json_object(separation_path)
    if payload.get("mode") != request.mode:
        return None
    if payload.get("model") != _display_model(request.model, request.mode):
        return None
    stems_payload = payload.get("stems")
    if not isinstance(stems_payload, dict):
        return None
    stems: dict[str, Path] = {}
    for stem in expected_stems:
        stem_payload = stems_payload.get(stem)
        if not isinstance(stem_payload, dict):
            return None
        file_value = stem_payload.get("file")
        if not isinstance(file_value, str):
            return None
        stem_path = request.output_dir.parent.parent / file_value
        if not stem_path.is_file():
            return None
        stems[stem] = stem_path
    return SeparationResult(
        backend=str(payload.get("backend") or backend.name),
        backend_version=str(payload.get("backend_version") or backend.version),
        model=str(payload.get("model") or request.model),
        mode=request.mode,
        input_path=request.input_path,
        stems=stems,
        elapsed_seconds=0.0,
        warnings=list(payload.get("warnings") or []),
        cached=True,
    )


def _build_stem_info(
    dataset_dir: Path,
    result: SeparationResult,
) -> dict[str, StemInfo]:
    stem_info: dict[str, StemInfo] = {}
    for name, path in sorted(result.stems.items()):
        audio, sample_rate = torchaudio.load(str(path))
        stem_info[name] = StemInfo(
            name=name,
            file=str(path.resolve().relative_to(dataset_dir)),
            analysis=analyze_audio_tensor(audio, sample_rate),
        )
    return stem_info


def _separator_input_path(
    clip_path: Path,
    output_dir: Path,
    *,
    force: bool,
) -> tuple[Path, str]:
    if clip_path.suffix.lower() == ".wav":
        return clip_path, ""
    normalized_path = output_dir / "source.wav"
    if normalized_path.is_file() and not force:
        return (
            normalized_path,
            f"normalized non-WAV source for separation: {clip_path.name}",
        )
    executable = shutil.which("ffmpeg")
    if executable is None:
        return (
            clip_path,
            "ffmpeg not found; using original non-WAV source for separation",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-ar",
        "48000",
        "-ac",
        "2",
        str(normalized_path),
    ]
    subprocess.run(command, check=True)
    return (
        normalized_path,
        f"normalized non-WAV source for separation: {clip_path.name}",
    )


def _separation_payload(
    dataset_dir: Path,
    result: SeparationResult,
    stem_info: dict[str, StemInfo],
) -> dict[str, Any]:
    return {
        "anvil_dataset_version": DATASET_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": result.backend,
        "backend_version": result.backend_version,
        "model": result.model,
        "mode": result.mode,
        "source_clip": str(result.input_path.resolve().relative_to(dataset_dir)),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "cached": result.cached,
        "warnings": result.warnings,
        "stems": {
            name: {
                "file": info.file,
                "analysis": info.analysis,
            }
            for name, info in sorted(stem_info.items())
        },
    }


def _update_dataset_manifest(
    *,
    manifest_path: Path,
    dataset_dir: Path,
    config: DatasetSeparationConfig,
    backend: SeparationBackend,
    clip_results: list[ClipSeparationResult],
) -> None:
    manifest = _load_json_object(manifest_path) if manifest_path.exists() else {}
    manifest.setdefault("anvil_dataset_version", DATASET_VERSION)
    manifest["separation"] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "backend": backend.name,
        "backend_version": backend.version,
        "mode": config.mode,
        "model": _display_model(config.model, config.mode),
        "clip_count": len(clip_results),
        "stems_dir": "stems/",
        "clips": [
            {
                "file": item.clip_file,
                "separation_file": str(item.separation_path.relative_to(dataset_dir)),
                "stems": {
                    name: info.file for name, info in sorted(item.stem_info.items())
                },
                "cached": item.result.cached,
                "warnings": item.result.warnings,
            }
            for item in clip_results
        ],
    }
    _write_json(manifest_path, manifest)


def _load_backend(name: str) -> SeparationBackend:
    if name != "audio-separator":
        raise ValueError(f"Unsupported separation backend: {name}")
    from anvil_audio.separation.audio_separator_backend import AudioSeparatorBackend

    return AudioSeparatorBackend()


def _display_model(model: str, mode: SeparationMode) -> str:
    if model != "auto":
        return model
    if mode == "four-stem":
        return "htdemucs_ft.yaml"
    return "auto"


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


def elapsed_since(start: float) -> float:
    """Return elapsed wall-clock seconds from a perf_counter value."""
    return perf_counter() - start
