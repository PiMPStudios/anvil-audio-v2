"""Automated dataset preparation for LoRA training workflows."""

from __future__ import annotations

import json
import importlib.util
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T

CaptionMode = Literal["heuristic", "llm", "off"]
TranscriptionBackend = Literal["auto", "lightning-whisper-mlx", "whisper"]

AUDIO_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}

STYLE_KEYWORDS = (
    "acoustic",
    "alternative",
    "ambient",
    "anthemic",
    "bass",
    "cinematic",
    "country",
    "dance",
    "drum",
    "edm",
    "electronic",
    "folk",
    "funk",
    "guitar",
    "hip hop",
    "house",
    "industrial",
    "jazz",
    "metal",
    "orchestral",
    "piano",
    "pop",
    "punk",
    "r&b",
    "rap",
    "rock",
    "singer songwriter",
    "soul",
    "synth",
    "trap",
    "vocal",
)


@dataclass(slots=True)
class DatasetBuildConfig:
    """Configuration for local or YouTube dataset construction."""

    output_dir: Path
    name: str = "anvil_dataset"
    max_clips: int = 40
    clip_length_seconds: float = 35.0
    stride_seconds: float | None = None
    sample_rate: int = 48_000
    audio_channels: int = 2
    min_clip_seconds: float = 8.0
    style_hint: str = ""
    caption_mode: CaptionMode = "heuristic"
    llm_model: str | None = None
    max_sources: int | None = None
    keep_downloads: bool = True
    quiet_ytdlp: bool = False
    transcribe_vocals: bool = False
    transcribe_all: bool = False
    transcription_backend: TranscriptionBackend = "auto"
    transcription_model: str | None = None
    transcription_language: str | None = None
    transcription_batch_size: int = 12
    transcription_max_chars: int = 180


@dataclass(slots=True)
class SourceTrack:
    """A discovered or downloaded source audio file."""

    path: Path
    source_type: str
    title: str
    source_url: str = ""
    index: int = 0


@dataclass(slots=True)
class ClipRecord:
    """Metadata written for each generated training clip."""

    file: str
    prompt: str
    caption: str
    tags: list[str]
    negative_tags: list[str]
    confidence: float
    analysis: dict[str, Any]
    source: dict[str, Any]
    seconds_start: float
    seconds_total: float
    transcript: str = ""
    transcription: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetBuildResult:
    """Paths and records produced by a dataset build."""

    dataset_dir: Path
    clips_dir: Path
    manifest_path: Path
    captions_path: Path
    character_sheet_path: Path
    dataset_config_path: Path
    records: list[ClipRecord]


def build_local_dataset(
    source_dir: Path, config: DatasetBuildConfig
) -> DatasetBuildResult:
    """Build a training dataset from a local folder of audio files."""
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Local source directory does not exist: {source_dir}")

    files = _find_audio_files(source_dir)
    if not files:
        raise RuntimeError(f"No audio files found under {source_dir}")

    sources = [
        SourceTrack(
            path=file,
            source_type="local",
            title=_title_from_path(file),
            index=index,
        )
        for index, file in enumerate(files, start=1)
    ]
    if config.max_sources:
        sources = sources[: config.max_sources]
    return _build_dataset_from_sources(
        sources, config, source_reference=str(source_dir)
    )


def build_youtube_dataset(
    source_url: str, config: DatasetBuildConfig
) -> DatasetBuildResult:
    """Download authorized YouTube audio with yt-dlp and build a dataset."""
    dataset_dir = config.output_dir.expanduser().resolve()
    downloads_dir = dataset_dir / "sources"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    _download_youtube_audio(source_url, downloads_dir, config)

    files = _find_audio_files(downloads_dir)
    if not files:
        raise RuntimeError(
            "yt-dlp completed but no audio files were found. Check that ffmpeg is "
            "available and the URL contains downloadable audio."
        )
    sources = [
        SourceTrack(
            path=file,
            source_type="youtube",
            title=_title_from_path(file),
            source_url=source_url,
            index=index,
        )
        for index, file in enumerate(files, start=1)
    ]
    return _build_dataset_from_sources(sources, config, source_reference=source_url)


def analyze_audio_tensor(audio: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    """Return deterministic, lightweight audio features for captioning."""
    if audio.ndim != 2:
        raise ValueError("audio must have shape [channels, samples]")
    if audio.shape[-1] == 0:
        raise ValueError("audio has no samples")

    mono = audio.mean(dim=0).float()
    duration = mono.numel() / sample_rate
    rms = torch.sqrt(torch.mean(mono.square())).item()
    peak = torch.max(torch.abs(audio)).item()
    rms_db = _linear_to_db(rms)
    peak_db = _linear_to_db(peak)
    spectral = _spectral_features(mono, sample_rate)
    onset_density, tempo_bpm = _rhythm_features(mono, sample_rate)
    width = _stereo_width(audio)

    energy = _label_energy(rms_db)
    brightness = _label_brightness(spectral["spectral_centroid_hz"])
    density = "dense" if onset_density >= 0.16 else "steady"
    bass = (
        "bass-heavy" if spectral["low_frequency_ratio"] >= 0.22 else "balanced low end"
    )
    stereo = "wide stereo" if width >= 0.35 else "centered stereo"

    return {
        "duration_seconds": round(duration, 3),
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
        "energy": energy,
        "spectral_centroid_hz": spectral["spectral_centroid_hz"],
        "brightness": brightness,
        "high_frequency_ratio": spectral["high_frequency_ratio"],
        "low_frequency_ratio": spectral["low_frequency_ratio"],
        "bass_character": bass,
        "onset_density": round(onset_density, 3),
        "density": density,
        "tempo_bpm_estimate": tempo_bpm,
        "stereo_width": round(width, 3),
        "stereo_character": stereo,
    }


def make_heuristic_caption(
    *,
    title: str,
    analysis: dict[str, Any],
    style_hint: str = "",
    caption_mode: CaptionMode = "heuristic",
    transcription: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact training caption from analysis and source metadata."""
    style_tags = _tags_from_text(style_hint)
    title_tags = _tags_from_text(title)
    transcript = _transcript_text(transcription)
    vocal_hint = _vocal_hint(title, style_hint, transcript)

    audio_tags = [
        f"{analysis['energy']} energy",
        f"{analysis['brightness']} tone",
        analysis["bass_character"],
        analysis["density"],
        analysis["stereo_character"],
    ]
    if analysis.get("tempo_bpm_estimate"):
        audio_tags.append(f"around {analysis['tempo_bpm_estimate']} bpm")
    if vocal_hint != "unknown vocals":
        audio_tags.append(vocal_hint)
    if transcript:
        audio_tags.append("transcribed vocals")

    tags = _dedupe(style_tags + title_tags + audio_tags)
    if caption_mode == "off":
        caption = style_hint.strip() or title.strip() or "audio training clip"
        confidence = 0.25
    else:
        leading_tags = tags[:9] or ["audio training clip"]
        caption = ", ".join(leading_tags)
        lyric_hint = _lyric_hint(transcript)
        if lyric_hint:
            caption = f'{caption}, lyric hint "{lyric_hint}"'
        confidence = 0.54
        if style_tags:
            confidence += 0.14
        if title_tags:
            confidence += 0.08
        if analysis.get("tempo_bpm_estimate"):
            confidence += 0.04
        if transcript:
            confidence += 0.08

    negative_tags = _negative_tags_for_analysis(analysis, vocal_hint)
    return {
        "caption": _clean_caption(caption),
        "tags": tags,
        "negative_tags": negative_tags,
        "confidence": round(min(confidence, 0.82), 2),
    }


def write_dataset_files(
    *,
    dataset_dir: Path,
    clips_dir: Path,
    records: list[ClipRecord],
    config: DatasetBuildConfig,
    source_reference: str,
) -> DatasetBuildResult:
    """Write manifest, captions, character sheet, and training config."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    captions_path = dataset_dir / "captions.json"
    character_sheet_path = dataset_dir / "character_sheet.json"
    dataset_config_path = dataset_dir / "dataset_config.json"

    created_at = datetime.now(UTC).isoformat()
    captions = [_record_to_json(record) for record in records]
    character_sheet = build_character_sheet(
        records=records,
        dataset_name=config.name,
        style_hint=config.style_hint,
        source_reference=source_reference,
        caption_mode=config.caption_mode,
        llm_model=config.llm_model,
    )
    manifest = {
        "name": config.name,
        "created_at": created_at,
        "source_reference": source_reference,
        "sample_rate": config.sample_rate,
        "audio_channels": config.audio_channels,
        "clip_length_seconds": config.clip_length_seconds,
        "clip_count": len(records),
        "total_clip_seconds": round(sum(r.seconds_total for r in records), 3),
        "caption_mode": config.caption_mode,
        "style_hint": config.style_hint,
        "transcription": {
            "enabled": config.transcribe_vocals,
            "transcribe_all": config.transcribe_all,
            "backend": config.transcription_backend,
            "model": config.transcription_model or "",
            "language": config.transcription_language or "",
        },
        "files": {
            "clips": "clips/",
            "captions": "captions.json",
            "character_sheet": "character_sheet.json",
            "dataset_config": "dataset_config.json",
        },
        "clips": captions,
    }
    dataset_config = {
        "dataset_type": "audio_dir",
        "datasets": [{"id": config.name, "path": str(clips_dir)}],
        "random_crop": True,
    }

    _write_json(manifest_path, manifest)
    _write_json(captions_path, captions)
    _write_json(character_sheet_path, character_sheet)
    _write_json(dataset_config_path, dataset_config)
    return DatasetBuildResult(
        dataset_dir=dataset_dir,
        clips_dir=clips_dir,
        manifest_path=manifest_path,
        captions_path=captions_path,
        character_sheet_path=character_sheet_path,
        dataset_config_path=dataset_config_path,
        records=records,
    )


def build_character_sheet(
    *,
    records: list[ClipRecord],
    dataset_name: str,
    style_hint: str,
    source_reference: str,
    caption_mode: CaptionMode,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Summarize clip captions into a reusable style profile."""
    tag_counts = Counter(tag for record in records for tag in record.tags)
    top_tags = [tag for tag, _ in tag_counts.most_common(12)]
    negative_counts = Counter(tag for record in records for tag in record.negative_tags)
    negative_traits = [tag for tag, _ in negative_counts.most_common(8)]
    average_confidence = (
        sum(record.confidence for record in records) / len(records) if records else 0.0
    )
    summary = _character_summary(dataset_name, style_hint, top_tags)
    sheet = {
        "name": dataset_name,
        "source_reference": source_reference,
        "summary": summary,
        "core_traits": top_tags[:8],
        "prompt_guidance": _prompt_guidance(style_hint, top_tags),
        "negative_traits": negative_traits,
        "clip_count": len(records),
        "average_caption_confidence": round(average_confidence, 2),
        "caption_mode": caption_mode,
        "confidence_notes": [
            "Audio features are deterministic estimates.",
            _transcription_confidence_note(records),
        ],
    }
    if caption_mode == "llm" and records:
        llm_sheet = _character_sheet_with_llm(sheet, records, llm_model)
        if llm_sheet:
            sheet.update(llm_sheet)
    return sheet


def _build_dataset_from_sources(
    sources: list[SourceTrack],
    config: DatasetBuildConfig,
    *,
    source_reference: str,
) -> DatasetBuildResult:
    dataset_dir = config.output_dir.expanduser().resolve()
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    records: list[ClipRecord] = []
    llm = _load_caption_llm(config.caption_mode, config.llm_model)
    transcriber = _load_transcriber(config)
    for source in sources:
        if len(records) >= config.max_clips:
            break
        audio = _load_audio(source.path, config.sample_rate, config.audio_channels)
        for clip, start_seconds in _iter_clips(audio, config):
            if len(records) >= config.max_clips:
                break
            index = len(records) + 1
            clip_name = f"clip_{index:04d}.wav"
            clip_path = clips_dir / clip_name
            torchaudio.save(str(clip_path), clip.cpu(), config.sample_rate)

            analysis = analyze_audio_tensor(clip, config.sample_rate)
            transcription = _transcribe_clip(
                transcriber=transcriber,
                clip_path=clip_path,
                title=source.title,
                style_hint=config.style_hint,
                transcribe_all=config.transcribe_all,
            )
            caption_payload = _caption_payload(
                title=source.title,
                analysis=analysis,
                style_hint=config.style_hint,
                caption_mode=config.caption_mode,
                llm=llm,
                transcription=transcription,
            )
            record = ClipRecord(
                file=f"clips/{clip_name}",
                prompt=caption_payload["caption"],
                caption=caption_payload["caption"],
                tags=caption_payload["tags"],
                negative_tags=caption_payload["negative_tags"],
                confidence=float(caption_payload["confidence"]),
                analysis=analysis,
                source={
                    "type": source.source_type,
                    "title": source.title,
                    "url": source.source_url,
                    "path": str(source.path),
                    "index": source.index,
                },
                seconds_start=round(start_seconds, 3),
                seconds_total=round(clip.shape[-1] / config.sample_rate, 3),
                transcript=_transcript_text(transcription),
                transcription=transcription,
            )
            records.append(record)
            _write_json(clip_path.with_suffix(".json"), _clip_sidecar(record))

    if not records:
        raise RuntimeError("No clips were produced from the selected sources.")
    if not config.keep_downloads:
        sources_dir = dataset_dir / "sources"
        if sources_dir.exists():
            shutil.rmtree(sources_dir)
    return write_dataset_files(
        dataset_dir=dataset_dir,
        clips_dir=clips_dir,
        records=records,
        config=config,
        source_reference=source_reference,
    )


def _download_youtube_audio(
    source_url: str, downloads_dir: Path, config: DatasetBuildConfig
) -> None:
    executable = shutil.which("yt-dlp")
    if executable is None:
        raise RuntimeError(
            "yt-dlp is required for YouTube dataset builds. Install it with "
            "`brew install yt-dlp` or `pip install yt-dlp`."
        )
    command = [
        executable,
        "--ignore-errors",
        "--no-overwrites",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--write-info-json",
        "--paths",
        str(downloads_dir),
        "--output",
        "%(autonumber)03d_%(id)s_%(title).80B.%(ext)s",
    ]
    if config.quiet_ytdlp:
        command.append("--quiet")
    if config.max_sources is not None and config.max_sources > 0:
        command.extend(["--playlist-end", str(config.max_sources)])
    command.append(source_url)
    subprocess.run(command, check=True)


def _find_audio_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files)


def _load_audio(path: Path, sample_rate: int, channels: int) -> torch.Tensor:
    audio, in_sample_rate = torchaudio.load(str(path))
    audio = audio.float()
    if audio.ndim != 2 or audio.shape[-1] == 0:
        raise RuntimeError(f"Could not load audio from {path}")
    if in_sample_rate != sample_rate:
        audio = T.Resample(in_sample_rate, sample_rate)(audio)
    if channels == 1:
        audio = audio.mean(dim=0, keepdim=True)
    elif channels == 2:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
    else:
        raise ValueError("audio_channels must be 1 or 2")
    return audio.clamp(-1.0, 1.0)


def _iter_clips(
    audio: torch.Tensor, config: DatasetBuildConfig
) -> list[tuple[torch.Tensor, float]]:
    clip_samples = max(1, int(round(config.clip_length_seconds * config.sample_rate)))
    min_samples = max(1, int(round(config.min_clip_seconds * config.sample_rate)))
    stride_seconds = config.stride_seconds or config.clip_length_seconds
    stride_samples = max(1, int(round(stride_seconds * config.sample_rate)))
    total_samples = audio.shape[-1]
    if total_samples < min_samples:
        return []
    if total_samples <= clip_samples:
        padded = F.pad(audio, (0, clip_samples - total_samples))
        return [(padded, 0.0)]

    clips = []
    for start in range(0, total_samples - clip_samples + 1, stride_samples):
        end = start + clip_samples
        clips.append((audio[:, start:end], start / config.sample_rate))
    return clips


def _caption_payload(
    *,
    title: str,
    analysis: dict[str, Any],
    style_hint: str,
    caption_mode: CaptionMode,
    llm: Any | None,
    transcription: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback = make_heuristic_caption(
        title=title,
        analysis=analysis,
        style_hint=style_hint,
        caption_mode=caption_mode,
        transcription=transcription,
    )
    if caption_mode != "llm" or llm is None:
        return fallback

    system_prompt = (
        "You write compact captions for AI music LoRA training clips.\n"
        "Return strict JSON with these fields: caption, tags, negative_tags, confidence.\n"
        "The caption must be one concise comma-separated phrase grounded in the "
        "provided facts. Do not invent specific artists or song titles. Do not "
        "overweight the transcription; use it only as a short vocal or lyric hint. Do not "
        "include markdown."
    )
    user_prompt = json.dumps(
        {
            "source_title": title,
            "style_hint": style_hint,
            "audio_analysis": analysis,
            "clip_transcription": transcription or {},
            "draft_caption": fallback["caption"],
            "draft_tags": fallback["tags"],
        },
        indent=2,
        sort_keys=True,
    )
    raw = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=260,
        temperature=0.35,
    )
    parsed = _parse_json_object(raw)
    caption = _clean_caption(str(parsed.get("caption", "")).strip())
    if not caption:
        return fallback
    tags = parsed.get("tags", fallback["tags"])
    negative_tags = parsed.get("negative_tags", fallback["negative_tags"])
    confidence = parsed.get("confidence", fallback["confidence"])
    return {
        "caption": caption,
        "tags": _coerce_string_list(tags) or fallback["tags"],
        "negative_tags": _coerce_string_list(negative_tags)
        or fallback["negative_tags"],
        "confidence": _coerce_confidence(confidence, fallback["confidence"]),
    }


class _LocalWhisperTranscriber:
    """Optional local transcription wrapper for dataset vocal hints."""

    def __init__(
        self,
        *,
        backend: TranscriptionBackend,
        model: str | None,
        language: str | None,
        batch_size: int,
        max_hint_chars: int,
    ) -> None:
        self.backend = _resolve_transcription_backend(backend)
        self.model = model or _default_transcription_model(self.backend)
        self.language = language or None
        self.batch_size = max(1, int(batch_size))
        self.max_hint_chars = max(24, int(max_hint_chars))
        self._engine: Any | None = None

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        if self.backend == "lightning-whisper-mlx":
            result = self._transcribe_lightning(audio_path)
        else:
            result = self._transcribe_openai_whisper(audio_path)
        return _normalize_transcription_result(
            result,
            backend=self.backend,
            model=self.model,
            max_hint_chars=self.max_hint_chars,
        )

    def _transcribe_lightning(self, audio_path: Path) -> dict[str, Any]:
        if self._engine is None:
            from lightning_whisper_mlx import LightningWhisperMLX

            self._engine = LightningWhisperMLX(
                model=self.model,
                batch_size=self.batch_size,
                quant=None,
            )
        kwargs: dict[str, Any] = {"audio_path": str(audio_path)}
        if self.language:
            kwargs["language"] = self.language
        return self._engine.transcribe(**kwargs)

    def _transcribe_openai_whisper(self, audio_path: Path) -> dict[str, Any]:
        if self._engine is None:
            import whisper

            self._engine = whisper.load_model(self.model)
        kwargs: dict[str, Any] = {"fp16": torch.cuda.is_available()}
        if self.language:
            kwargs["language"] = self.language
        return self._engine.transcribe(str(audio_path), **kwargs)


def _load_transcriber(config: DatasetBuildConfig) -> _LocalWhisperTranscriber | None:
    if not config.transcribe_vocals and not config.transcribe_all:
        return None
    return _LocalWhisperTranscriber(
        backend=config.transcription_backend,
        model=config.transcription_model,
        language=config.transcription_language,
        batch_size=config.transcription_batch_size,
        max_hint_chars=config.transcription_max_chars,
    )


def _resolve_transcription_backend(
    requested: TranscriptionBackend,
) -> Literal["lightning-whisper-mlx", "whisper"]:
    if requested != "auto":
        if not _has_transcription_backend(requested):
            raise RuntimeError(_missing_transcription_backend_message(requested))
        return requested
    if _has_transcription_backend("lightning-whisper-mlx"):
        return "lightning-whisper-mlx"
    if _has_transcription_backend("whisper"):
        return "whisper"
    raise RuntimeError(_missing_transcription_backend_message("auto"))


def _has_transcription_backend(backend: str) -> bool:
    if backend == "lightning-whisper-mlx":
        return importlib.util.find_spec("lightning_whisper_mlx") is not None
    if backend == "whisper":
        return importlib.util.find_spec("whisper") is not None
    return False


def _missing_transcription_backend_message(backend: str) -> str:
    backend_note = f" for backend '{backend}'" if backend != "auto" else ""
    return (
        f"Local vocal transcription was requested{backend_note}, but no supported "
        "local Whisper runtime is installed. Install one of: "
        "`pip install lightning-whisper-mlx` on Apple Silicon, or "
        "`pip install openai-whisper` for the local PyTorch Whisper runtime."
    )


def _default_transcription_model(backend: str) -> str:
    if backend == "lightning-whisper-mlx":
        return "distil-medium.en"
    return "small"


def _transcribe_clip(
    *,
    transcriber: _LocalWhisperTranscriber | None,
    clip_path: Path,
    title: str,
    style_hint: str,
    transcribe_all: bool,
) -> dict[str, Any]:
    if transcriber is None:
        return {}
    if not transcribe_all and _vocal_hint(title, style_hint, "") != "vocal-forward":
        return {}
    try:
        return transcriber.transcribe(clip_path)
    except Exception as exc:
        raise RuntimeError(f"Could not transcribe {clip_path}: {exc}") from exc


def _normalize_transcription_result(
    result: dict[str, Any],
    *,
    backend: str,
    model: str,
    max_hint_chars: int,
) -> dict[str, Any]:
    text = _clean_transcript(str(result.get("text", "")))
    if not text:
        return {}
    return {
        "text": text,
        "hint": _lyric_hint(text, max_chars=max_hint_chars),
        "language": str(result.get("language") or "").strip(),
        "backend": backend,
        "model": model,
        "segments": _transcription_segments(result.get("segments")),
    }


def _transcription_segments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        text = _clean_transcript(str(item.get("text", "")))
        if not text:
            continue
        segments.append(
            {
                "start": _round_optional_float(item.get("start")),
                "end": _round_optional_float(item.get("end")),
                "text": text,
            }
        )
    return segments


def _round_optional_float(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _character_sheet_with_llm(
    base_sheet: dict[str, Any], records: list[ClipRecord], llm_model: str | None
) -> dict[str, Any]:
    llm = _load_caption_llm("llm", llm_model)
    if llm is None:
        return {}
    system_prompt = (
        "You summarize an AI music LoRA dataset into a character sheet.\n"
        "Return strict JSON with fields: summary, core_traits, prompt_guidance, "
        "negative_traits, confidence_notes. Keep it factual and compact."
    )
    user_prompt = json.dumps(
        {
            "base_sheet": base_sheet,
            "sample_captions": [record.caption for record in records[:20]],
        },
        indent=2,
        sort_keys=True,
    )
    raw = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=420,
        temperature=0.45,
    )
    parsed = _parse_json_object(raw)
    return {
        key: value
        for key, value in parsed.items()
        if key
        in {
            "summary",
            "core_traits",
            "prompt_guidance",
            "negative_traits",
            "confidence_notes",
        }
    }


def _load_caption_llm(caption_mode: CaptionMode, model: str | None) -> Any | None:
    if caption_mode != "llm":
        return None
    from anvil_audio.intelligence import LocalLLM

    return LocalLLM(model)


def _spectral_features(mono: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    if mono.numel() < 512:
        return {
            "spectral_centroid_hz": 0,
            "high_frequency_ratio": 0.0,
            "low_frequency_ratio": 0.0,
        }
    n_fft = min(2048, 2 ** int(math.floor(math.log2(mono.numel()))))
    n_fft = max(n_fft, 512)
    hop_length = max(128, n_fft // 4)
    window = torch.hann_window(n_fft, device=mono.device)
    spectrum = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    mean_magnitude = spectrum.mean(dim=1)
    total = mean_magnitude.sum().clamp_min(1e-8)
    freqs = torch.linspace(0, sample_rate / 2, mean_magnitude.numel())
    centroid = ((freqs * mean_magnitude).sum() / total).item()
    high_ratio = (mean_magnitude[freqs >= 5_000].sum() / total).item()
    low_ratio = (mean_magnitude[freqs <= 160].sum() / total).item()
    return {
        "spectral_centroid_hz": int(round(centroid)),
        "high_frequency_ratio": round(high_ratio, 3),
        "low_frequency_ratio": round(low_ratio, 3),
    }


def _rhythm_features(mono: torch.Tensor, sample_rate: int) -> tuple[float, int | None]:
    frame_size = min(2048, mono.numel())
    hop = max(256, frame_size // 2)
    if mono.numel() < frame_size * 3:
        return 0.0, None
    frames = mono.unfold(0, frame_size, hop)
    envelope = torch.sqrt(torch.mean(frames.square(), dim=1))
    onset = F.relu(envelope[1:] - envelope[:-1])
    if onset.numel() < 8 or onset.max().item() <= 1e-6:
        return 0.0, None
    threshold = onset.mean() + onset.std()
    onset_density = (onset > threshold).float().mean().item()
    tempo = _estimate_tempo(onset, sample_rate / hop)
    return onset_density, tempo


def _estimate_tempo(onset: torch.Tensor, frames_per_second: float) -> int | None:
    centered = onset - onset.mean()
    if centered.abs().sum().item() <= 1e-6:
        return None
    best_bpm = None
    best_score = 0.0
    for bpm in range(60, 181):
        lag = max(1, int(round(frames_per_second * 60 / bpm)))
        if lag >= centered.numel():
            continue
        score = torch.dot(centered[:-lag], centered[lag:]).item()
        if score > best_score:
            best_score = score
            best_bpm = bpm
    if best_bpm is None or best_score <= 1e-7:
        return None
    return int(best_bpm)


def _stereo_width(audio: torch.Tensor) -> float:
    if audio.shape[0] < 2:
        return 0.0
    left = audio[0]
    right = audio[1]
    side = torch.mean(torch.abs(left - right)).item()
    mid = torch.mean(torch.abs(left + right)).item()
    return side / max(side + mid, 1e-8)


def _record_to_json(record: ClipRecord) -> dict[str, Any]:
    return {
        "file": record.file,
        "caption": record.caption,
        "prompt": record.prompt,
        "tags": record.tags,
        "negative_tags": record.negative_tags,
        "confidence": record.confidence,
        "analysis": record.analysis,
        "source": record.source,
        "seconds_start": record.seconds_start,
        "seconds_total": record.seconds_total,
        "transcript": record.transcript,
        "transcription": record.transcription,
    }


def _clip_sidecar(record: ClipRecord) -> dict[str, Any]:
    data = _record_to_json(record)
    data["prompt"] = record.prompt
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _title_from_path(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^\d+[_ -]+", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    return " ".join(stem.split())


def _tags_from_text(text: str) -> list[str]:
    lowered = text.lower()
    tags = [keyword for keyword in STYLE_KEYWORDS if keyword in lowered]
    for piece in re.split(r"[,;/|]+", lowered):
        cleaned = " ".join(piece.split()).strip()
        if 3 <= len(cleaned) <= 32 and any(ch.isalpha() for ch in cleaned):
            tags.append(cleaned)
    return _dedupe(tags)


def _vocal_hint(title: str, style_hint: str, transcript: str = "") -> str:
    text = f"{title} {style_hint}".lower()
    if any(word in text for word in ("instrumental", "karaoke", "no vocal")):
        return "instrumental"
    if transcript or any(
        word in text for word in ("vocal", "singer", "lyrics", "feat", "ft.")
    ):
        return "vocal-forward"
    return "unknown vocals"


def _negative_tags_for_analysis(analysis: dict[str, Any], vocal_hint: str) -> list[str]:
    tags = ["clipping", "muddy mix", "noisy artifacts"]
    if analysis["brightness"] == "bright":
        tags.append("harsh treble")
    if analysis["bass_character"] == "bass-heavy":
        tags.append("boomy low end")
    if vocal_hint == "vocal-forward":
        tags.append("off-key vocals")
    return _dedupe(tags)


def _character_summary(dataset_name: str, style_hint: str, top_tags: list[str]) -> str:
    if style_hint:
        return _clean_caption(style_hint)
    if top_tags:
        return f"{dataset_name} style focused on {', '.join(top_tags[:6])}"
    return f"{dataset_name} audio style profile"


def _prompt_guidance(style_hint: str, top_tags: list[str]) -> str:
    pieces = _dedupe(_tags_from_text(style_hint) + top_tags[:8])
    if not pieces:
        return "Use the generated clip captions as the training prompt style."
    return ", ".join(pieces)


def _clean_caption(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split())
    text = text.strip(" ,.;")
    return text[:600]


def _clean_transcript(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = " ".join(text.replace("\n", " ").split()).strip()
    return text.strip(" ,.;")


def _transcript_text(transcription: dict[str, Any] | None) -> str:
    if not transcription:
        return ""
    return _clean_transcript(str(transcription.get("text") or ""))


def _lyric_hint(text: str, max_chars: int = 180) -> str:
    text = _clean_transcript(text)
    if not text:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    hint = first_sentence if len(first_sentence) <= max_chars else text[:max_chars]
    return hint.strip(" ,.;")


def _transcription_confidence_note(records: list[ClipRecord]) -> str:
    transcribed_count = sum(1 for record in records if record.transcript)
    if transcribed_count:
        return (
            f"{transcribed_count} clips include local Whisper transcription hints; "
            "review transcript text before training vocal-specific LoRAs."
        )
    return (
        "Vocal and instrument tags are inferred from source text unless optional "
        "local transcription is enabled."
    )


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_clean_caption(value)] if value.strip() else []
    if isinstance(value, list):
        return _dedupe(_clean_caption(str(item)) for item in value if str(item).strip())
    return []


def _coerce_confidence(value: Any, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return fallback
    return round(min(max(confidence, 0.0), 1.0), 2)


def _dedupe(items: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = _clean_caption(str(item)).lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _label_energy(rms_db: float) -> str:
    if rms_db >= -18:
        return "high"
    if rms_db >= -30:
        return "medium"
    return "low"


def _label_brightness(centroid_hz: int) -> str:
    if centroid_hz >= 3_500:
        return "bright"
    if centroid_hz >= 1_800:
        return "balanced"
    return "dark"


def _linear_to_db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-8))
