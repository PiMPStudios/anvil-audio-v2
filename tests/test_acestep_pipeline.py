"""Tests for ACEStepPipeline without project_root."""

import sys

import pytest


def test_import_error_without_project_root_gives_clear_message(monkeypatch):
    """When acestep isn't installed and project_root=None, error is actionable."""
    for key in list(sys.modules.keys()):
        if key.startswith("acestep"):
            monkeypatch.delitem(sys.modules, key)

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "acestep.handler":
            raise ImportError("No module named 'acestep'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    with pytest.raises(ImportError, match="pip install anvil-audio\\[acestep\\]"):
        ACEStepPipeline(project_root=None)


def test_project_root_none_skips_sys_path_injection(monkeypatch):
    """project_root=None does not mutate sys.path."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "acestep.handler":
            raise ImportError("No module named 'acestep'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    path_before = set(sys.path)
    try:
        ACEStepPipeline(project_root=None)
    except ImportError:
        pass

    new_entries = [p for p in sys.path if p not in path_before]
    assert not new_entries, f"sys.path was mutated with project_root=None: {new_entries}"
