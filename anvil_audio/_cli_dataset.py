"""Dataset preparation CLI."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from anvil_audio.dataset_builder import (
    CaptionMode,
    DatasetBuildConfig,
    TranscriptionBackend,
    build_local_dataset,
    build_youtube_dataset,
)
from anvil_audio.dataset_bundle import (
    TrainingBundleConfig,
    export_training_bundle,
    parse_include,
)
from anvil_audio.dataset_captions import (
    CaptionAuditConfig,
    audit_or_repair_captions,
)
from anvil_audio.dataset_qa import (
    DEFAULT_EMBEDDING_INSTRUCTION,
    DatasetQAConfig,
    run_dataset_qa,
)
from anvil_audio.separation import (
    DatasetSeparationConfig,
    separate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil dataset",
        description="Build reviewable audio datasets for LoRA training.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser(
        "build-local",
        help="Build a dataset from a local folder of audio files.",
    )
    local.add_argument("source_dir", help="Folder containing source audio.")
    _add_common_build_args(local)

    youtube = subparsers.add_parser(
        "build-youtube",
        help="Download authorized YouTube audio with yt-dlp and build a dataset.",
    )
    youtube.add_argument("url", help="YouTube video, playlist, or channel URL.")
    youtube.add_argument(
        "--tracks",
        type=int,
        default=None,
        help="Maximum source videos/tracks to download before clipping.",
    )
    youtube.add_argument(
        "--delete-downloads",
        action="store_true",
        help="Delete raw downloaded source files after clips are written.",
    )
    youtube.add_argument(
        "--quiet-ytdlp",
        action="store_true",
        help="Pass --quiet to yt-dlp.",
    )
    _add_common_build_args(youtube)

    qa = subparsers.add_parser(
        "qa",
        help="Run Qwen embedding QA against a built dataset's captions.",
    )
    qa.add_argument("dataset_dir", help="Directory produced by `anvil dataset`.")
    qa.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Embedding model path or HuggingFace repo. Default: local "
            "Qwen3-Embedding-0.6B cache when present, otherwise "
            "Qwen/Qwen3-Embedding-0.6B."
        ),
    )
    qa.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cuda, mps, or cpu. Default: auto.",
    )
    qa.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size. Default: 8.",
    )
    qa.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max tokenizer length for captions. Default: 512.",
    )
    qa.add_argument(
        "--output",
        default=None,
        help="Output JSON report path. Default: DATASET_DIR/dataset_qa_report.json.",
    )
    qa.add_argument(
        "--markdown-output",
        default=None,
        help="Output Markdown report path. Default: DATASET_DIR/dataset_qa_report.md.",
    )
    qa.add_argument(
        "--no-markdown",
        action="store_true",
        help="Only write the JSON report.",
    )
    qa.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for duplicate pairs. Default: 0.9.",
    )
    qa.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold used to group clusters. Default: 0.78.",
    )
    qa.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.55,
        help="Mean neighbor similarity below which a clip is an outlier. Default: 0.55.",
    )
    qa.add_argument(
        "--nearest-neighbors",
        type=int,
        default=5,
        help="Neighbor count used for outlier scoring. Default: 5.",
    )
    qa.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.45,
        help="Caption confidence below which clips are flagged. Default: 0.45.",
    )
    qa.add_argument(
        "--instruction",
        default=DEFAULT_EMBEDDING_INSTRUCTION,
        help="Optional instruction prefix used for Qwen caption embeddings.",
    )
    qa.add_argument(
        "--include-stems",
        action="store_true",
        help="Include source-separation stem health checks when metadata exists.",
    )

    separate = subparsers.add_parser(
        "separate",
        help="Separate clips in an existing dataset into vocals/instrumental/stems.",
    )
    separate.add_argument("dataset_dir", help="Directory produced by `anvil dataset`.")
    separate.add_argument(
        "--backend",
        choices=("audio-separator",),
        default="audio-separator",
        help="Source separation backend. Default: audio-separator.",
    )
    separate.add_argument(
        "--mode",
        choices=("instrumental", "four-stem", "vocals"),
        default="instrumental",
        help=(
            "Stem output mode. Default: instrumental "
            "(vocals + instrumental)."
        ),
    )
    separate.add_argument(
        "--model",
        default="auto",
        help=(
            "audio-separator model filename. Default: auto "
            "(htdemucs_ft.yaml for four-stem, backend default otherwise)."
        ),
    )
    separate.add_argument(
        "--output-format",
        default="wav",
        choices=("wav", "flac"),
        help="Stem output format. Default: wav.",
    )
    separate.add_argument(
        "--model-file-dir",
        default=None,
        help="Optional audio-separator model cache directory.",
    )
    separate.add_argument(
        "--force",
        action="store_true",
        help="Recompute stems even when cached separation metadata exists.",
    )
    separate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only separate the first N caption records. Useful for smoke tests.",
    )

    bundle = subparsers.add_parser(
        "export-training-bundle",
        help="Write a portable training_bundle.json from a built dataset.",
    )
    bundle.add_argument("dataset_dir", help="Directory produced by `anvil dataset`.")
    bundle.add_argument(
        "--profile",
        default="acestep-lora",
        help="Training profile name written into the bundle. Default: acestep-lora.",
    )
    bundle.add_argument(
        "--include",
        default="full-mix",
        help=(
            "Comma-separated assets to include: full-mix, vocals, instrumental, "
            "drums, bass, other. Default: full-mix."
        ),
    )
    bundle.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: DATASET_DIR/training_bundle.json.",
    )
    bundle.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested asset is missing.",
    )

    captions = subparsers.add_parser(
        "captions",
        help="Audit or repair duplicate/low-confidence dataset captions.",
    )
    captions.add_argument("dataset_dir", help="Directory produced by `anvil dataset`.")
    captions.add_argument(
        "--repair",
        action="store_true",
        help="Generate deterministic replacement captions for weak duplicates.",
    )
    captions.add_argument(
        "--write",
        action="store_true",
        help="Write repaired captions. Without this, repair mode is a dry run.",
    )
    captions.add_argument(
        "--style-hint",
        default="",
        help="Optional style hint to blend into repaired captions.",
    )
    captions.add_argument(
        "--min-confidence",
        type=float,
        default=0.62,
        help="Captions below this confidence are repair candidates. Default: 0.62.",
    )
    captions.add_argument(
        "--output",
        default=None,
        help="Output audit JSON path. Default: DATASET_DIR/caption_audit_report.json.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "qa":
        try:
            result = run_dataset_qa(
                DatasetQAConfig(
                    dataset_dir=Path(args.dataset_dir),
                    output_json=Path(args.output) if args.output else None,
                    output_markdown=(
                        Path(args.markdown_output) if args.markdown_output else None
                    ),
                    write_markdown=not args.no_markdown,
                    embedding_model=args.embedding_model,
                    device=args.device,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    duplicate_threshold=args.duplicate_threshold,
                    cluster_threshold=args.cluster_threshold,
                    outlier_threshold=args.outlier_threshold,
                    nearest_neighbors=args.nearest_neighbors,
                    low_confidence_threshold=args.low_confidence_threshold,
                    instruction=args.instruction,
                    include_stems=args.include_stems,
                )
            )
        except Exception as exc:
            print(f"dataset QA failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_qa_result(result.report, result.json_path, result.markdown_path)
        return

    if args.command == "separate":
        try:
            result = separate_dataset(
                DatasetSeparationConfig(
                    dataset_dir=Path(args.dataset_dir),
                    backend=args.backend,
                    mode=args.mode,
                    model=args.model,
                    output_format=args.output_format,
                    force=args.force,
                    limit=args.limit,
                    model_file_dir=(
                        Path(args.model_file_dir) if args.model_file_dir else None
                    ),
                )
            )
        except Exception as exc:
            print(f"dataset separation failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_separation_result(result)
        return

    if args.command == "export-training-bundle":
        try:
            result = export_training_bundle(
                TrainingBundleConfig(
                    dataset_dir=Path(args.dataset_dir),
                    profile=args.profile,
                    include=parse_include(args.include),
                    output=Path(args.output) if args.output else None,
                    strict=args.strict,
                )
            )
        except Exception as exc:
            print(f"training bundle export failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_bundle_result(result)
        return

    if args.command == "captions":
        mode = "heuristic" if args.repair else "audit"
        try:
            result = audit_or_repair_captions(
                CaptionAuditConfig(
                    dataset_dir=Path(args.dataset_dir),
                    mode=mode,
                    write=args.write,
                    style_hint=args.style_hint,
                    min_confidence=args.min_confidence,
                    output=Path(args.output) if args.output else None,
                )
            )
        except Exception as exc:
            print(f"caption audit failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_caption_audit_result(result)
        return

    output_dir = (
        Path(args.output_dir) if args.output_dir else _default_output_dir(args.name)
    )
    config = DatasetBuildConfig(
        output_dir=output_dir,
        name=args.name,
        max_clips=args.clips,
        clip_length_seconds=args.clip_length,
        stride_seconds=args.stride,
        sample_rate=args.sample_rate,
        audio_channels=args.channels,
        min_clip_seconds=args.min_clip_length,
        style_hint=args.style_hint,
        caption_mode=args.caption_mode,
        llm_model=args.llm_model,
        max_sources=getattr(args, "tracks", None),
        keep_downloads=not getattr(args, "delete_downloads", False),
        quiet_ytdlp=getattr(args, "quiet_ytdlp", False),
        transcribe_vocals=args.transcribe_vocals or args.transcribe_all,
        transcribe_all=args.transcribe_all,
        transcription_backend=args.transcription_backend,
        transcription_model=args.transcription_model,
        transcription_language=args.transcription_language,
        transcription_batch_size=args.transcription_batch_size,
        transcription_max_chars=args.transcription_max_chars,
    )
    try:
        if args.command == "build-local":
            result = build_local_dataset(Path(args.source_dir), config)
        elif args.command == "build-youtube":
            _print_youtube_notice()
            result = build_youtube_dataset(args.url, config)
        else:
            parser.error(f"unknown dataset command: {args.command}")
    except Exception as exc:
        print(f"dataset build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Dataset: {result.dataset_dir}")
    print(f"Clips: {len(result.records)} -> {result.clips_dir}")
    print(f"Captions: {result.captions_path}")
    print(f"Character sheet: {result.character_sheet_path}")
    print(f"Training dataset config: {result.dataset_config_path}")


def _print_qa_result(
    report: dict, json_path: Path, markdown_path: Path | None
) -> None:
    summary = report["summary"]
    print(f"Dataset QA report: {json_path}")
    if markdown_path is not None:
        print(f"Markdown report: {markdown_path}")
    print(
        "Summary: "
        f"{summary['cluster_count']} clusters, "
        f"{summary['duplicate_pair_count']} duplicate pairs, "
        f"{summary['outlier_count']} outliers, "
        f"{summary['low_confidence_count']} low-confidence captions"
    )
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")


def _print_separation_result(result) -> None:
    print(f"Dataset: {result.dataset_dir}")
    print(f"Stems: {result.stems_dir}")
    print(f"Separated clips: {len(result.clips)}")
    for item in result.clips[:10]:
        stems = ", ".join(sorted(item.stem_info))
        cached = " cached" if item.result.cached else ""
        print(f"- {item.clip_file}: {stems}{cached}")
    if len(result.clips) > 10:
        print(f"- ... {len(result.clips) - 10} more")


def _print_bundle_result(result) -> None:
    print(f"Training bundle: {result.bundle_path}")
    print(f"Clips: {result.clip_count}")
    print(f"Assets: {result.asset_count}")
    if result.warnings:
        print(f"Warnings: {len(result.warnings)}")
        for warning in result.warnings[:10]:
            print(f"- {warning}")
        if len(result.warnings) > 10:
            print(f"- ... {len(result.warnings) - 10} more")


def _print_caption_audit_result(result) -> None:
    print(f"Caption audit: {result.report_path}")
    print(f"Exact duplicate groups: {result.exact_duplicate_groups}")
    print(f"Duplicate records: {result.duplicate_record_count}")
    print(f"Low-confidence captions: {result.low_confidence_count}")
    print(f"Repaired captions: {result.repaired_count}")
    for warning in result.warnings:
        print(f"- {warning}")


def _add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--name",
        default="anvil_dataset",
        help="Dataset name written into manifests. Default: anvil_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dataset directory. Default: ./datasets/<name>_<timestamp>.",
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=40,
        help="Maximum number of clips to write. Default: 40.",
    )
    parser.add_argument(
        "--clip-length",
        type=float,
        default=35.0,
        help="Clip length in seconds. Default: 35.",
    )
    parser.add_argument(
        "--min-clip-length",
        type=float,
        default=8.0,
        help="Skip source files shorter than this many seconds. Default: 8.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=None,
        help="Seconds between clip starts. Default: same as --clip-length.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Output sample rate. Default: 48000.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=2,
        help="Output channel count. Default: 2.",
    )
    parser.add_argument(
        "--style-hint",
        default="",
        help="Optional style hint applied to generated captions.",
    )
    parser.add_argument(
        "--caption-mode",
        choices=("heuristic", "llm", "off"),
        default="heuristic",
        type=_caption_mode,
        help="Caption generation mode. Default: heuristic.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional local path or HuggingFace repo for LLM caption cleanup.",
    )
    parser.add_argument(
        "--transcribe-vocals",
        action="store_true",
        help=(
            "Use an optional local Whisper runtime to add short transcription hints "
            "for clips likely to contain vocals."
        ),
    )
    parser.add_argument(
        "--transcribe-all",
        action="store_true",
        help=(
            "Run local transcription on every clip. Implies --transcribe-vocals."
        ),
    )
    parser.add_argument(
        "--transcription-backend",
        choices=("auto", "lightning-whisper-mlx", "whisper"),
        default="auto",
        type=_transcription_backend,
        help=(
            "Optional transcription backend. Default: auto "
            "(lightning-whisper-mlx, then local whisper)."
        ),
    )
    parser.add_argument(
        "--transcription-model",
        default=None,
        help=(
            "Optional local transcription model. Defaults to distil-medium.en for "
            "lightning-whisper-mlx or small for openai-whisper."
        ),
    )
    parser.add_argument(
        "--transcription-language",
        default=None,
        help="Optional source language code for transcription, e.g. en.",
    )
    parser.add_argument(
        "--transcription-batch-size",
        type=int,
        default=12,
        help="Batch size for lightning-whisper-mlx. Default: 12.",
    )
    parser.add_argument(
        "--transcription-max-chars",
        type=int,
        default=180,
        help="Maximum characters from a transcript to include as a lyric hint.",
    )


def _caption_mode(value: str) -> CaptionMode:
    if value not in {"heuristic", "llm", "off"}:
        raise argparse.ArgumentTypeError("must be heuristic, llm, or off")
    return value  # type: ignore[return-value]


def _transcription_backend(value: str) -> TranscriptionBackend:
    if value not in {"auto", "lightning-whisper-mlx", "whisper"}:
        raise argparse.ArgumentTypeError(
            "must be auto, lightning-whisper-mlx, or whisper"
        )
    return value  # type: ignore[return-value]


def _default_output_dir(name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("datasets") / f"{safe_name}_{stamp}"


def _print_youtube_notice() -> None:
    print(
        "YouTube dataset builds should only be used with material you own or are "
        "authorized to train on."
    )


if __name__ == "__main__":
    main()
