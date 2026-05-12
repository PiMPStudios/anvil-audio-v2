"""Tests for local prompt and lyric intelligence helpers."""

from anvil_audio import intelligence
from anvil_audio.intelligence import (
    LyricWritingPlan,
    enhance_prompt,
    prepare_song_prompt,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_enhance_prompt_parses_json_response():
    llm = FakeLLM(
        [
            '{"prompt":"anthemic alt rock, live drums","negative_prompt":"muddy mix, clipping"}'
        ]
    )

    package = enhance_prompt("alt rock", duration_seconds=60, llm=llm)

    assert package.prompt == "anthemic alt rock, live drums"
    assert package.negative_prompt == "muddy mix, clipping"
    assert "strict JSON" in llm.calls[0]["system_prompt"]


def test_prepare_song_uses_enhanced_prompt_for_lyrics():
    llm = FakeLLM(
        [
            '{"prompt":"polished synth pop, clear vocal","negative_prompt":"muddy low end"}',
            "[Verse]\nNeon hearts awake\n[Chorus]\nWe shine through the rain",
        ]
    )

    package = prepare_song_prompt("synth pop", duration_seconds=45, llm=llm)

    assert package.prompt == "polished synth pop, clear vocal"
    assert package.negative_prompt == "muddy low end"
    assert "[Chorus]" in package.lyrics
    assert "polished synth pop" in llm.calls[1]["user_prompt"]


def test_lyric_plan_scales_with_duration():
    short = LyricWritingPlan.make(30)
    long = LyricWritingPlan.make(180)

    assert short.max_tokens == 96
    assert "4 to 6" in short.line_budget
    assert long.max_tokens == 420
    assert "20 to 32" in long.line_budget


def test_default_model_alias_prefers_existing_anvilapp_download(monkeypatch, tmp_path):
    model_dir = (
        tmp_path
        / "Library"
        / "Application Support"
        / "Anvil"
        / "LLM"
        / intelligence.DEFAULT_LLM_MODEL_ID
    )
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.safetensors").write_text("", encoding="utf-8")
    monkeypatch.setattr(intelligence.Path, "home", lambda: tmp_path)

    resolved = intelligence.resolve_llm_model_ref(intelligence.DEFAULT_LLM_MODEL_ID)

    assert resolved == str(model_dir)


def test_default_model_uses_anvil_audio_cache_when_present(monkeypatch, tmp_path):
    model_dir = (
        tmp_path / ".cache" / "anvil-audio" / "llm" / intelligence.DEFAULT_LLM_MODEL_ID
    )
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.safetensors").write_text("", encoding="utf-8")
    monkeypatch.setattr(intelligence.Path, "home", lambda: tmp_path)

    resolved = intelligence.resolve_llm_model_ref()

    assert resolved == str(model_dir)
