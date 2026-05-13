"""Tests for local prompt and lyric intelligence helpers."""

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from types import ModuleType

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


def test_enhance_prompt_falls_back_when_llm_echoes_original():
    llm = FakeLLM(
        [
            '{"prompt":"dark blues vocals slow tempo minor key raw guitar atmospheric","negative_prompt":""}'
        ]
    )

    package = enhance_prompt(
        "dark blues vocals slow tempo minor key raw guitar atmospheric",
        duration_seconds=60,
        llm=llm,
    )

    assert package.prompt != "dark blues vocals slow tempo minor key raw guitar atmospheric"
    assert "raw blues guitar" in package.prompt
    assert "expressive lead vocal" in package.prompt
    assert "60 seconds" not in package.prompt
    assert package.negative_prompt == (
        "muddy mix, harsh clipping, distorted vocals, off-key vocals, weak drums, "
        "thin bass, noisy artifacts"
    )


def test_enhance_prompt_removes_duration_from_llm_prompt():
    llm = FakeLLM(
        [
            '{"prompt":"dark blues, raw guitar, smoky vocal, analog tape, 60 seconds","negative_prompt":"harsh treble"}'
        ]
    )

    package = enhance_prompt("dark blues raw guitar", duration_seconds=60, llm=llm)

    assert package.prompt == "dark blues, raw guitar, smoky vocal, analog tape"
    assert package.negative_prompt == "harsh treble"


def test_enhance_prompt_extracts_truncated_json_prompt():
    llm = FakeLLM(
        [
            '{"prompt":"dark blues, raw guitar, smoky vocal, raw guitar, slow tempo, slow tempo'
        ]
    )

    package = enhance_prompt("dark blues raw guitar", duration_seconds=60, llm=llm)

    assert package.prompt == "dark blues, raw guitar, smoky vocal, slow tempo"
    assert not package.prompt.startswith("{")


def test_enhance_prompt_strips_raw_json_prefix():
    llm = FakeLLM(
        [
            '{"prompt":"dark blues, raw guitar, smoky vocal","negative_prompt":"muddy mix"}'
        ]
    )

    package = enhance_prompt("dark blues raw guitar", duration_seconds=60, llm=llm)

    assert package.prompt == "dark blues, raw guitar, smoky vocal"
    assert package.negative_prompt == "muddy mix"


def test_enhance_prompt_converts_instruction_sentence_to_tags():
    llm = FakeLLM(
        [
            '{"prompt":"Create a 60-second dark blues music piece with raw guitar, atmospheric textures, minor key mood, slow tempo, and expressive vocals, and avoid muddy mix, harsh treble","negative_prompt":"Avoid muddy mix, harsh treble, weak drums"}'
        ]
    )

    package = enhance_prompt(
        "dark blues, vocals, slow tempo, minor key, raw guitar, atmospheric",
        duration_seconds=60,
        llm=llm,
    )

    assert package.prompt.startswith("dark blues, raw guitar")
    assert "Create" not in package.prompt
    assert "avoid" not in package.prompt.lower()
    assert "60-second" not in package.prompt
    assert package.negative_prompt == "muddy mix, harsh treble, weak drums"


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


def test_write_lyrics_removes_parenthetical_notes_and_caps_lines():
    llm = FakeLLM(
        [
            "\n".join(
                [
                    "[Intro]",
                    "(soft atmospheric pads)",
                    "Line one",
                    "[Verse]",
                    "(guitar enters)",
                    "Line two",
                    "Line three",
                    "Line four",
                    "Line five",
                    "Line six",
                    "Line seven",
                    "Line eight",
                    "Line nine",
                    "Line ten",
                    "Line eleven",
                    "[Bridge]",
                    "Line twelve",
                ]
            )
        ]
    )

    lyrics = intelligence.write_lyrics("dark blues", duration_seconds=60, llm=llm)

    assert "(" not in lyrics
    assert "Line eleven" not in lyrics
    assert "Line ten" in lyrics


def test_lyric_plan_scales_with_duration():
    short = LyricWritingPlan.make(30)
    long = LyricWritingPlan.make(180)

    assert short.max_tokens == 96
    assert "4 to 6" in short.line_budget
    assert long.max_tokens == 420
    assert "20 to 32" in long.line_budget
    assert "[Verse 1], [Chorus], [Verse 2], [Chorus], [Bridge]" in long.section_plan


def test_write_lyrics_prompt_discourages_generic_cliches():
    llm = FakeLLM(["[Verse 1]\nNeon sign above the laundromat\n[Chorus]\nYour coat still smells like smoke"])

    intelligence.write_lyrics("dark blues", duration_seconds=200, llm=llm)

    system_prompt = llm.calls[0]["system_prompt"]
    assert "Avoid default lyric cliches" in system_prompt
    assert "Do not collapse the whole song into one [Verse]" in system_prompt
    assert "Do not put more than 6 lyric lines" in system_prompt


def test_write_lyrics_repairs_oversized_single_verse_for_long_songs():
    llm = FakeLLM(
        [
            "\n".join(
                ["[Verse]"]
                + [
                    f"Concrete lyric line {index}"
                    for index in range(1, 19)
                ]
            )
        ]
    )

    lyrics = intelligence.write_lyrics(
        "slow dark blues, smoky barroom vocal",
        duration_seconds=200,
        llm=llm,
    )

    assert lyrics.startswith("[Verse 1]\nConcrete lyric line 1")
    assert "[Chorus]" in lyrics
    assert "[Verse 2]" in lyrics
    assert "[Bridge]" in lyrics
    assert "\n[Verse]\n" not in lyrics


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


def test_local_llm_generation_uses_one_mlx_worker_thread(monkeypatch):
    calls = []

    def fake_generate_mlx_text(*args, **kwargs):
        calls.append(threading.get_ident())
        return "ok"

    monkeypatch.setattr(intelligence, "_generate_mlx_text", fake_generate_mlx_text)
    monkeypatch.setattr(intelligence, "_MLX_EXECUTOR", None)
    monkeypatch.setattr(intelligence, "_MLX_WORKER_THREAD_ID", None)

    llm = intelligence.LocalLLM(model="fake-model")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: llm.generate(
                    system_prompt="system",
                    user_prompt="user",
                    max_tokens=4,
                ),
                range(2),
            )
        )

    try:
        assert results == ["ok", "ok"]
        assert len(set(calls)) == 1
    finally:
        mlx_executor = intelligence._MLX_EXECUTOR
        if mlx_executor is not None:
            mlx_executor.shutdown(wait=True)
        intelligence._MLX_EXECUTOR = None
        intelligence._MLX_WORKER_THREAD_ID = None


def test_generate_mlx_text_rebinds_mlx_lm_stream_before_generate(monkeypatch):
    events = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"{messages[0]['content']} {messages[1]['content']}"

    def fake_load_model(model_ref):
        return "model", FakeTokenizer()

    def fake_bind_stream(generate_module):
        events.append("bind")
        generate_module.generation_stream = "worker-stream"

    generate_module = ModuleType("mlx_lm.generate")
    generate_module.generation_stream = "main-thread-stream"

    def fake_generate(model, tokenizer, **kwargs):
        events.append("generate")
        assert generate_module.generation_stream == "worker-stream"
        assert kwargs["prompt"] == "system user"
        return "done"

    generate_module.generate = fake_generate

    sample_utils_module = ModuleType("mlx_lm.sample_utils")

    def fake_make_sampler(**kwargs):
        events.append("sampler")
        return "sampler"

    sample_utils_module.make_sampler = fake_make_sampler

    monkeypatch.setattr(intelligence, "_load_mlx_model", fake_load_model)
    monkeypatch.setattr(
        intelligence,
        "_bind_mlx_lm_generation_stream",
        fake_bind_stream,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils_module)

    text = intelligence._generate_mlx_text(
        "fake-model",
        system_prompt="system",
        user_prompt="user",
        max_tokens=4,
        temperature=0.7,
    )

    assert text == "done"
    assert events == ["bind", "sampler", "generate"]
