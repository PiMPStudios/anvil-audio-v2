"""Prompt enhancement and lyric-writing CLI."""

from __future__ import annotations

import argparse
import json

from anvil_audio.intelligence import prepare_song_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance an audio prompt and optionally write duration-aware lyrics.",
    )
    parser.add_argument("--prompt", required=True, help="Base prompt to enhance.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Existing negative prompt to preserve or improve.",
    )
    parser.add_argument(
        "--duration",
        "--seconds-total",
        dest="duration_seconds",
        type=float,
        default=60.0,
        help="Target track duration in seconds. Default: 60.",
    )
    parser.add_argument(
        "--mode",
        default="music",
        help="Generation mode/context for enhancement. Default: music.",
    )
    parser.add_argument(
        "--style",
        default="",
        help="Optional lyric style hint.",
    )
    parser.add_argument(
        "--no-lyrics",
        action="store_true",
        help="Only enhance prompt/negative prompt; do not write lyrics.",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Only write lyrics from the prompt; do not rewrite the prompt.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional local path or HuggingFace repo for mlx-lm.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    package = prepare_song_prompt(
        args.prompt,
        mode=args.mode,
        duration_seconds=args.duration_seconds,
        negative_prompt=args.negative_prompt,
        style=args.style,
        write_vocals=not args.no_lyrics,
        enhance=not args.no_enhance,
        model=args.model,
    )
    print(json.dumps(package.to_dict(), indent=2))


if __name__ == "__main__":
    main()
