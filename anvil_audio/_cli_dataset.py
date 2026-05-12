"""Dataset preparation CLI."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from anvil_audio.dataset_builder import (
    CaptionMode,
    DatasetBuildConfig,
    build_local_dataset,
    build_youtube_dataset,
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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


def _caption_mode(value: str) -> CaptionMode:
    if value not in {"heuristic", "llm", "off"}:
        raise argparse.ArgumentTypeError("must be heuristic, llm, or off")
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
