import json
import math

import torch
import torchaudio

from anvil_audio.dataset_builder import (
    DatasetBuildConfig,
    analyze_audio_tensor,
    build_local_dataset,
)


def test_analyze_audio_tensor_returns_caption_features():
    sample_rate = 8_000
    t = torch.linspace(0, 2.0, sample_rate * 2)
    mono = 0.25 * torch.sin(2 * math.pi * 440 * t)
    audio = torch.stack([mono, mono])

    analysis = analyze_audio_tensor(audio, sample_rate)

    assert analysis["duration_seconds"] == 2.0
    assert analysis["energy"] in {"medium", "high"}
    assert analysis["brightness"] in {"dark", "balanced", "bright"}
    assert "stereo_width" in analysis


def test_build_local_dataset_writes_clips_and_metadata(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sample_rate = 8_000
    t = torch.linspace(0, 5.0, sample_rate * 5)
    mono = 0.2 * torch.sin(2 * math.pi * 220 * t)
    audio = torch.stack([mono, mono])
    torchaudio.save(str(source_dir / "anthemic_rock_guitar.wav"), audio, sample_rate)

    result = build_local_dataset(
        source_dir,
        DatasetBuildConfig(
            output_dir=tmp_path / "dataset",
            name="test_rock",
            max_clips=2,
            clip_length_seconds=2.0,
            min_clip_seconds=1.0,
            sample_rate=sample_rate,
            style_hint="alternative rock, live drums",
            caption_mode="heuristic",
        ),
    )

    assert len(result.records) == 2
    assert result.manifest_path.exists()
    assert result.captions_path.exists()
    assert result.character_sheet_path.exists()
    assert result.dataset_config_path.exists()

    captions = json.loads(result.captions_path.read_text(encoding="utf-8"))
    assert captions[0]["file"] == "clips/clip_0001.wav"
    assert "alternative rock" in captions[0]["caption"]
    assert (result.clips_dir / "clip_0001.json").exists()

    dataset_config = json.loads(result.dataset_config_path.read_text(encoding="utf-8"))
    assert dataset_config["dataset_type"] == "audio_dir"
    assert dataset_config["datasets"][0]["path"] == str(result.clips_dir)


def test_build_local_dataset_can_use_fake_llm_captioner(monkeypatch, tmp_path):
    class FakeLLM:
        def generate(self, **kwargs):
            if "character sheet" in kwargs["system_prompt"]:
                return '{"summary":"focused rock dataset","core_traits":["rock"]}'
            return (
                '{"caption":"polished rock, live drums, gritty guitar",'
                '"tags":["rock","live drums","guitar"],'
                '"negative_tags":["muddy mix"],"confidence":0.91}'
            )

    monkeypatch.setattr(
        "anvil_audio.dataset_builder._load_caption_llm",
        lambda caption_mode, model: FakeLLM(),
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sample_rate = 8_000
    t = torch.linspace(0, 2.0, sample_rate * 2)
    mono = 0.2 * torch.sin(2 * math.pi * 330 * t)
    audio = torch.stack([mono, mono])
    torchaudio.save(str(source_dir / "rock_vocal.wav"), audio, sample_rate)

    result = build_local_dataset(
        source_dir,
        DatasetBuildConfig(
            output_dir=tmp_path / "dataset",
            name="llm_test",
            max_clips=1,
            clip_length_seconds=2.0,
            min_clip_seconds=1.0,
            sample_rate=sample_rate,
            caption_mode="llm",
        ),
    )

    assert result.records[0].caption == "polished rock, live drums, gritty guitar"
    sheet = json.loads(result.character_sheet_path.read_text(encoding="utf-8"))
    assert sheet["summary"] == "focused rock dataset"


def test_build_local_dataset_adds_optional_vocal_transcription(monkeypatch, tmp_path):
    class FakeTranscriber:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio_path):
            self.calls.append(audio_path)
            return {
                "text": "I keep the light low on the backroad",
                "hint": "I keep the light low",
                "language": "en",
                "backend": "fake",
                "model": "fake-whisper",
                "segments": [],
            }

    fake_transcriber = FakeTranscriber()
    monkeypatch.setattr(
        "anvil_audio.dataset_builder._load_transcriber",
        lambda config: fake_transcriber if config.transcribe_vocals else None,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sample_rate = 8_000
    t = torch.linspace(0, 2.0, sample_rate * 2)
    mono = 0.2 * torch.sin(2 * math.pi * 330 * t)
    audio = torch.stack([mono, mono])
    torchaudio.save(str(source_dir / "smoky_vocal_blues.wav"), audio, sample_rate)

    result = build_local_dataset(
        source_dir,
        DatasetBuildConfig(
            output_dir=tmp_path / "dataset",
            name="vocal_test",
            max_clips=1,
            clip_length_seconds=2.0,
            min_clip_seconds=1.0,
            sample_rate=sample_rate,
            style_hint="dark blues, vocal-forward",
            caption_mode="heuristic",
            transcribe_vocals=True,
        ),
    )

    assert len(fake_transcriber.calls) == 1
    assert result.records[0].transcript == "I keep the light low on the backroad"
    assert "lyric hint" in result.records[0].caption
    assert "transcribed vocals" in result.records[0].tags

    captions = json.loads(result.captions_path.read_text(encoding="utf-8"))
    assert captions[0]["transcription"]["backend"] == "fake"
    assert captions[0]["transcript"] == "I keep the light low on the backroad"


def test_vocal_transcription_skips_non_vocal_clips(monkeypatch, tmp_path):
    class FakeTranscriber:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio_path):
            self.calls.append(audio_path)
            return {"text": "unexpected words"}

    fake_transcriber = FakeTranscriber()
    monkeypatch.setattr(
        "anvil_audio.dataset_builder._load_transcriber",
        lambda config: fake_transcriber if config.transcribe_vocals else None,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sample_rate = 8_000
    t = torch.linspace(0, 2.0, sample_rate * 2)
    mono = 0.2 * torch.sin(2 * math.pi * 220 * t)
    audio = torch.stack([mono, mono])
    torchaudio.save(str(source_dir / "ambient_guitar.wav"), audio, sample_rate)

    result = build_local_dataset(
        source_dir,
        DatasetBuildConfig(
            output_dir=tmp_path / "dataset",
            name="instrumental_test",
            max_clips=1,
            clip_length_seconds=2.0,
            min_clip_seconds=1.0,
            sample_rate=sample_rate,
            style_hint="ambient instrumental guitar",
            caption_mode="heuristic",
            transcribe_vocals=True,
        ),
    )

    assert fake_transcriber.calls == []
    assert result.records[0].transcript == ""
