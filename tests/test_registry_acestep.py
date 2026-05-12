"""Tests for ACE-Step registry entry with optional project_root."""
from pathlib import Path

from anvil_audio.core.registry import RegistryEntry


def test_acestep_entry_registers_without_project_root(monkeypatch):
    """RegistryEntry for acestep defaults to Anvil's cache root."""
    monkeypatch.delenv("ACESTEP_PROJECT_ROOT", raising=False)
    entry = RegistryEntry(
        name="test-acestep",
        pipeline_type="acestep",
        acestep_project_root=None,
    )
    assert entry.name == "test-acestep"
    assert entry.acestep_project_root == str(
        Path.home() / ".cache" / "anvil-audio" / "acestep"
    )


def test_acestep_entry_ignores_stale_env_var_for_builtin_cache(monkeypatch, tmp_path):
    """ACESTEP_PROJECT_ROOT no longer redirects built-in entries to old checkouts."""
    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", str(tmp_path))
    entry = RegistryEntry(
        name="test-acestep",
        pipeline_type="acestep",
        acestep_project_root=None,
    )
    assert entry.acestep_project_root == str(
        Path.home() / ".cache" / "anvil-audio" / "acestep"
    )


def test_acestep_entry_explicit_root_takes_priority(monkeypatch, tmp_path):
    """Explicit project_root overrides env var."""
    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", "/should/be/ignored")
    entry = RegistryEntry(
        name="test-acestep",
        pipeline_type="acestep",
        acestep_project_root=str(tmp_path),
    )
    assert entry.acestep_project_root == str(tmp_path)


def test_builtin_sft_defaults_match_anvilapp_direct_dit_path():
    """SFT defaults should match the known-good AnvilApp generation path."""
    from anvil_audio.core.registry import registry

    params = registry.get_model("acestep-v1.5-sft").resolved_params()
    assert params["steps"] == 50
    assert params["cfg_scale"] == 7.5
    assert params["shift"] == 3.0
    assert params["use_adg"] is False
    assert params["lm_cfg_scale"] == 2.0
    assert params["thinking"] is False
    assert params["use_cot_metas"] is False
    assert params["use_cot_caption"] is False
    assert params["use_cot_language"] is False
    assert params["dcw_enabled"] is False
    assert params["velocity_norm_threshold"] == 0.0
    assert params["velocity_ema_factor"] == 0.0
