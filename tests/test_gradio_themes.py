"""Tests for Gradio runtime theme helpers."""

from anvil_audio.interface import gradio as gradio_ui


def test_theme_dropdown_choices_include_bundled_presets():
    choices = gradio_ui._theme_dropdown_choices()

    assert ("Anvil Default", "anvil-default") in choices
    assert ("Ocean", "ocean") in choices
    assert ("Citrus", "citrus") in choices
    assert ("Glass", "glass") in choices
    assert ("Monochrome", "monochrome") in choices
    assert ("Terminal", "terminal") in choices
    assert ("Shiki", "shiki") in choices
    assert ("Minecraft", "minecraft") in choices
    assert ("Sketch", "sketch") in choices


def test_custom_theme_css_scopes_gradio_theme_variables(monkeypatch, tmp_path):
    monkeypatch.setattr(gradio_ui, "_theme_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        gradio_ui,
        "_load_hub_theme_css",
        lambda _repo_name: ":root { --block-background-fill: #000; }",
    )

    css = gradio_ui._build_custom_theme_css()

    assert 'html[data-anvil-theme="ocean"]' in css
    assert 'html[data-anvil-theme="citrus"]' in css
    assert 'html[data-anvil-theme="terminal"]' in css
    assert 'html[data-anvil-theme="shiki"]' in css
    assert 'html[data-anvil-theme="minecraft"]' in css
    assert 'html[data-anvil-theme="sketch"]' in css
    assert "--button-primary-background-fill" in css
    assert "--block-background-fill" in css


def test_hub_theme_css_uses_local_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "terminal.css"
    cache_path.write_text(":root { --cached-theme: yes; }", encoding="utf-8")
    monkeypatch.setattr(gradio_ui, "_theme_cache_root", lambda: tmp_path)
    monkeypatch.setattr(
        gradio_ui,
        "_load_hub_theme_css",
        lambda _repo_name: (_ for _ in ()).throw(AssertionError("unexpected fetch")),
    )

    assert (
        gradio_ui._get_hub_theme_css("terminal", "hmb/terminal")
        == ":root { --cached-theme: yes; }"
    )


def test_theme_javascript_persists_and_restores_selection():
    assert "localStorage.setItem" in gradio_ui._THEME_APPLY_JS
    assert "localStorage.getItem" in gradio_ui._THEME_LOAD_JS
    assert "data-anvil-theme" in gradio_ui._THEME_APPLY_JS
    assert "data-anvil-theme" in gradio_ui._THEME_LOAD_JS
    assert "hmb/terminal" in gradio_ui._THEME_APPLY_JS
    assert "YTheme/Minecraft" in gradio_ui._THEME_LOAD_JS
