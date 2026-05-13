"""Platform detection and first-run environment validation for anvil-audio."""

from __future__ import annotations

import platform
import sys


def detect_platform() -> str:
    """Return a canonical platform string.

    Returns one of: ``"macos-arm"``, ``"linux-x86"``, ``"linux-arm"``,
    ``"windows-x64"``, or ``"unknown"``.
    """
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine == "arm64":
        return "macos-arm"
    if sys.platform == "linux":
        if machine == "x86_64":
            return "linux-x86"
        if machine == "aarch64":
            return "linux-arm"
    if sys.platform == "win32":
        return "windows-x64"
    return "unknown"


def check_acestep_installed() -> bool:
    """Return True if the ``acestep`` package is importable."""
    try:
        __import__("acestep")
        return True
    except ImportError:
        return False


def check_mlx_installed() -> bool:
    """Return True if ``mlx_audiogen`` is importable."""
    try:
        __import__("mlx_audiogen")
        return True
    except ImportError:
        return False


def check_mlx_lm_installed() -> bool:
    """Return True if ``mlx_lm`` is importable."""
    try:
        __import__("mlx_lm")
        return True
    except ImportError:
        return False


def check_mlx_transcription_installed() -> bool:
    """Return True if ``lightning_whisper_mlx`` is importable."""
    try:
        __import__("lightning_whisper_mlx")
        return True
    except ImportError:
        return False


def print_environment_report() -> None:
    """Print a human-readable summary of the current environment."""
    plat = detect_platform()
    acestep_ok = check_acestep_installed()
    mlx_ok = check_mlx_installed()
    mlx_lm_ok = check_mlx_lm_installed()
    mlx_transcription_ok = check_mlx_transcription_installed()

    print(f"Platform:           {plat}")
    print(f"ACE-Step:           {'installed' if acestep_ok else 'NOT installed'}")
    print(f"MLX (Stable Audio): {'installed' if mlx_ok else 'NOT installed'}")
    print(f"MLX-LM intelligence: {'installed' if mlx_lm_ok else 'NOT installed'}")
    print(
        "MLX vocal transcription: "
        f"{'installed' if mlx_transcription_ok else 'NOT installed'}"
    )

    if not acestep_ok:
        print("\nTo install ACE-Step:")
        print("  bash install.sh   (macOS/Linux)")
        print("  .\\install.ps1    (Windows)")
    if not mlx_ok and plat == "macos-arm":
        print("\nTo enable MLX acceleration:")
        print("  pip install mlx-audiogen")
    if not mlx_lm_ok and plat == "macos-arm":
        print("\nTo enable local prompt enhancement and lyric writing:")
        print("  pip install mlx-lm")
    if not mlx_transcription_ok and plat == "macos-arm":
        print("\nTo enable vocal transcription for dataset builds:")
        print("  pip install lightning-whisper-mlx")
