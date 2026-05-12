"""
Anvil Audio — unified CLI entry point.

Usage::

    anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"
    anvil generate --model sfx-v1 --cond-yaml-path batch.yaml --output-dir ./out
    anvil --list-models
    anvil generate --list-models
    anvil generate --model-config path/to/config.json --ckpt-path path/to/ckpt.pt \\
        --prompt "rain on tin roof" --output-dir ./out

All subcommand logic lives in the dedicated modules; this file only dispatches.
"""

from __future__ import annotations

import sys


def main() -> None:
    # Apply warning filters before any heavy imports happen inside subcommands.
    # --verbose anywhere in argv disables them; we also strip the flag so
    # subcommand parsers don't complain about an unrecognised argument.
    _verbose = "--verbose" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--verbose"]
    if not _verbose:
        from anvil_audio._warning_filters import apply_filters

        apply_filters()

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    # Top-level shorthand: anvil --list-models → anvil generate --list-models
    if sys.argv[1] == "--list-models":
        sys.argv = ["anvil generate", "--list-models"]
        from anvil_audio._cli_generate import main as gen_main

        gen_main()
        return

    sub = sys.argv[1]
    # Remove the subcommand token so the delegated parser sees a clean argv.
    sys.argv = [f"anvil {sub}"] + sys.argv[2:]

    if sub == "generate":
        from anvil_audio._cli_generate import main as gen_main

        gen_main()
    elif sub == "enhance-prompt":
        from anvil_audio._cli_intelligence import main as intelligence_main

        intelligence_main()
    elif sub == "setup":
        _cmd_setup(_parse_setup_args())
    else:
        print(f"anvil: unknown subcommand '{sub}'")
        _print_help()
        sys.exit(1)


def _parse_setup_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog="anvil setup",
        description="Check environment and optionally pre-download model weights.",
    )
    parser.add_argument(
        "--download",
        metavar="MODEL",
        nargs="*",
        help=(
            "Model names to pre-download weights for (e.g. stable-audio-open-1.0). "
            "Omit to just check the environment."
        ),
    )
    return parser.parse_args()


def _cmd_setup(args) -> None:
    from anvil_audio._setup import print_environment_report
    from anvil_audio.core.registry import registry

    print("=== Anvil Audio Environment ===\n")
    print_environment_report()

    if args.download is not None:
        models = args.download or [e.name for e in registry.list_models()]
        print(f"\nPre-downloading weights for: {', '.join(models)}")
        for name in models:
            entry = registry.get_model(name)
            if entry is None:
                print(f"  [{name}] not found in registry — skipping")
                continue
            try:
                print(f"  [{name}] loading pipeline to trigger weight download...")
                registry.load_pipeline(name)
                print(f"  [{name}] OK")
            except Exception as exc:
                print(f"  [{name}] FAILED: {exc}")


def _print_help() -> None:
    print(
        "Anvil Audio — pluggable AI audio generation\n"
        "\n"
        "Usage:  anvil [--verbose] <subcommand> [options]\n"
        "\n"
        "Global flags:\n"
        "  --verbose      Show all upstream deprecation warnings (default: suppressed)\n"
        "  --list-models  List all available models and exit\n"
        "\n"
        "Subcommands:\n"
        "  generate    Generate audio from a model and prompt\n"
        "  enhance-prompt  Enhance a prompt and optionally write lyrics\n"
        "  setup       Check environment and optionally pre-download model weights\n"
        "\n"
        "Run 'anvil <subcommand> --help' for subcommand options.\n"
    )


if __name__ == "__main__":
    main()
