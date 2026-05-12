"""Tests for platform detection in anvil_audio._setup."""

import builtins
import importlib
import platform
import sys
from unittest.mock import patch


def _reload_setup():
    """Force a fresh import of anvil_audio._setup so patches take effect."""
    import anvil_audio._setup as mod

    importlib.reload(mod)
    return mod


def test_detect_platform_macos_arm():
    with (
        patch.object(sys, "platform", "darwin"),
        patch.object(platform, "machine", return_value="arm64"),
    ):
        mod = _reload_setup()
        result = mod.detect_platform()
    assert result == "macos-arm"


def test_detect_platform_linux_x86():
    with (
        patch.object(sys, "platform", "linux"),
        patch.object(platform, "machine", return_value="x86_64"),
    ):
        mod = _reload_setup()
        result = mod.detect_platform()
    assert result == "linux-x86"


def test_detect_platform_windows():
    with (
        patch.object(sys, "platform", "win32"),
        patch.object(platform, "machine", return_value="AMD64"),
    ):
        mod = _reload_setup()
        result = mod.detect_platform()
    assert result == "windows-x64"


def test_check_acestep_installed_false_when_missing(monkeypatch):
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "acestep":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio._setup import check_acestep_installed

    assert check_acestep_installed() is False


def test_check_mlx_audiogen_installed_false_when_missing(monkeypatch):
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlx_audiogen":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio._setup import check_mlx_installed

    assert check_mlx_installed() is False


def test_check_mlx_lm_installed_false_when_missing(monkeypatch):
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlx_lm":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio._setup import check_mlx_lm_installed

    assert check_mlx_lm_installed() is False
