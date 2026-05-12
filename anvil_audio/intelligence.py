"""Local prompt and lyric intelligence helpers.

This module mirrors AnvilApp's local-first intelligence path: a small MLX Llama
model expands short audio prompts and writes duration-aware lyrics without
calling an external API.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LLM_MODEL_ID = "llama-3.2-3b-instruct-4bit"
DEFAULT_LLM_REPO_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


@dataclass(slots=True)
class PromptPackage:
    """Prompt enhancement result returned by the local intelligence layer."""

    prompt: str
    negative_prompt: str = ""
    lyrics: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "lyrics": self.lyrics,
            "raw": self.raw,
        }


@dataclass(slots=True)
class LyricWritingPlan:
    """Duration-aware lyric budget matching AnvilApp's local lyric writer."""

    duration_seconds: int
    line_budget: str
    section_plan: str
    max_tokens: int

    @classmethod
    def make(cls, duration_seconds: float) -> "LyricWritingPlan":
        seconds = max(int(round(duration_seconds)), 15)
        if seconds <= 35:
            return cls(
                duration_seconds=seconds,
                line_budget="4 to 6 total lyric lines",
                section_plan=(
                    "Use [Verse] with 2 to 4 lines and optional [Hook] with "
                    "1 to 2 lines. Do not use [Verse 2], [Bridge], or [Outro]."
                ),
                max_tokens=96,
            )
        if seconds <= 60:
            return cls(
                duration_seconds=seconds,
                line_budget="6 to 10 total lyric lines",
                section_plan=(
                    "Use [Verse] and [Chorus]. Keep each section short. "
                    "Do not use [Bridge]."
                ),
                max_tokens=140,
            )
        if seconds <= 90:
            return cls(
                duration_seconds=seconds,
                line_budget="10 to 14 total lyric lines",
                section_plan=(
                    "Use [Verse], [Chorus], and optional [Verse 2]. Do not use "
                    "[Bridge] unless it replaces [Verse 2]."
                ),
                max_tokens=200,
            )
        if seconds <= 150:
            return cls(
                duration_seconds=seconds,
                line_budget="14 to 20 total lyric lines",
                section_plan=(
                    "Use [Verse 1], [Chorus], [Verse 2], [Chorus], and optional "
                    "[Bridge]. Keep the bridge to 2 lines."
                ),
                max_tokens=300,
            )
        return cls(
            duration_seconds=seconds,
            line_budget="20 to 32 total lyric lines",
            section_plan=(
                "Use a full song structure, but keep sections compact and avoid "
                "repeated filler."
            ),
            max_tokens=420,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are a concise songwriter writing lyrics for an AI music generator.\n"
            "Match the user's mood, genre, and energy, but fit the lyric density "
            "to the exact clip duration.\n"
            f"Target duration: {self.duration_seconds} seconds.\n"
            f"Write {self.line_budget}, total, across the whole song.\n"
            f"{self.section_plan}\n"
            "If the prompt sounds instrumental, riff-based, ambient, or like a "
            "backing track, write only a sparse chant or hook instead of a full "
            "vocal song.\n"
            "Prefer short concrete phrases over long narrative lines.\n"
            "End on a complete line. Never trail off mid-sentence.\n"
            "Output only lyrics with section markers. No commentary."
        )

    def user_prompt(self, prompt: str, style: str = "") -> str:
        style = style.strip()
        if not style:
            return f"Song description:\n{prompt}"
        return f"Song description:\n{prompt}\n\nStyle:\n{style}"


class LocalLLM:
    """Small wrapper around ``mlx-lm`` for local prompt/lyric generation."""

    def __init__(self, model: str | None = None) -> None:
        self.model_ref = resolve_llm_model_ref(model)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
    ) -> str:
        model, tokenizer = _load_mlx_model(self.model_ref)
        prompt = _format_chat_prompt(tokenizer, system_prompt, user_prompt)

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=0.9)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return str(text).strip()


def resolve_llm_model_ref(model: str | None = None) -> str:
    """Return a local model directory when available, otherwise an HF repo id."""
    explicit = (
        model
        or os.environ.get("ANVIL_LLM_MODEL")
        or os.environ.get("ANVIL_LLM_MODEL_PATH")
    )
    if explicit:
        explicit = os.path.expanduser(explicit.strip())
        if explicit in {DEFAULT_LLM_MODEL_ID, DEFAULT_LLM_REPO_ID}:
            return _resolve_default_llm_model_ref()
        explicit_path = Path(explicit)
        if explicit_path.is_dir():
            return str(explicit_path)
        return explicit

    return _resolve_default_llm_model_ref()


def _resolve_default_llm_model_ref() -> str:
    candidates = [
        Path.home()
        / "Library"
        / "Application Support"
        / "Anvil"
        / "LLM"
        / DEFAULT_LLM_MODEL_ID,
        _default_llm_cache_dir(),
    ]
    for path in candidates:
        if _has_llm_files(path):
            return str(path)
    return DEFAULT_LLM_REPO_ID


def enhance_prompt(
    prompt: str,
    *,
    mode: str = "music",
    duration_seconds: float | None = None,
    negative_prompt: str = "",
    llm: LocalLLM | None = None,
    model: str | None = None,
) -> PromptPackage:
    """Expand a short audio prompt and suggest a matching negative prompt."""
    prompt = prompt.strip()
    if not prompt:
        return PromptPackage(prompt="", negative_prompt=negative_prompt.strip())

    llm = llm or LocalLLM(model)
    duration_line = (
        f"Target duration: {duration_seconds:.0f} seconds."
        if duration_seconds is not None and duration_seconds > 0
        else "Target duration: unknown."
    )
    system_prompt = (
        "You are an audio prompt enhancer for Anvil Audio.\n"
        "Return strict JSON with exactly two string fields: prompt and negative_prompt.\n"
        "The prompt must be a concise comma-separated audio generation prompt, "
        "rich in genre, instruments, vocal character, tempo, mood, arrangement, "
        "texture, and production style when relevant.\n"
        "The negative_prompt must list unwanted artifacts or qualities such as "
        "muddy mix, clipping, harsh treble, weak drums, noisy artifacts, or "
        "off-key vocals. Avoid banning the requested genre, instruments, or vocals.\n"
        "Do not include lyrics. Do not include markdown."
    )
    user_prompt = (
        f"Generation mode: {mode}\n"
        f"{duration_line}\n"
        f"Current negative prompt: {negative_prompt.strip() or '(blank)'}\n\n"
        f"User prompt:\n{prompt}"
    )
    raw = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=360,
        temperature=0.65,
    )
    parsed = _parse_prompt_json(raw)
    enhanced = parsed.get("prompt") or _strip_wrapping(raw) or prompt
    suggested_negative = (
        parsed.get("negative_prompt")
        or negative_prompt
        or _default_negative_prompt(mode)
    )
    return PromptPackage(
        prompt=_clean_single_line(enhanced),
        negative_prompt=_clean_single_line(suggested_negative),
        raw=raw,
    )


def write_lyrics(
    prompt: str,
    *,
    style: str = "",
    duration_seconds: float = 60.0,
    llm: LocalLLM | None = None,
    model: str | None = None,
) -> str:
    """Write duration-aware lyrics from a prompt using the local LLM."""
    prompt = prompt.strip()
    if not prompt:
        return ""
    llm = llm or LocalLLM(model)
    plan = LyricWritingPlan.make(duration_seconds)
    text = llm.generate(
        system_prompt=plan.system_prompt,
        user_prompt=plan.user_prompt(prompt, style),
        max_tokens=plan.max_tokens,
        temperature=0.8,
    )
    return _clean_lyrics(text)


def prepare_song_prompt(
    prompt: str,
    *,
    mode: str = "music",
    duration_seconds: float = 60.0,
    negative_prompt: str = "",
    style: str = "",
    write_vocals: bool = True,
    enhance: bool = True,
    model: str | None = None,
    llm: LocalLLM | None = None,
) -> PromptPackage:
    """Enhance a prompt, suggest negatives, then optionally write lyrics."""
    llm = llm or LocalLLM(model)
    package = (
        enhance_prompt(
            prompt,
            mode=mode,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            llm=llm,
        )
        if enhance
        else PromptPackage(prompt=prompt, negative_prompt=negative_prompt)
    )
    if write_vocals:
        package.lyrics = write_lyrics(
            package.prompt,
            style=style,
            duration_seconds=duration_seconds,
            llm=llm,
        )
    return package


def _load_mlx_model(model_ref: str) -> tuple[Any, Any]:
    if sys.platform != "darwin":
        raise RuntimeError(
            "Local MLX intelligence is currently supported on macOS only."
        )
    if model_ref == DEFAULT_LLM_REPO_ID:
        model_ref = _download_default_llm_model()
    if model_ref in _MODEL_CACHE:
        return _MODEL_CACHE[model_ref]
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is required for local prompt enhancement. Install it with "
            "`pip install mlx-lm` or run `bash install.sh` on Apple Silicon."
        ) from exc

    model, tokenizer = load(model_ref)
    _MODEL_CACHE[model_ref] = (model, tokenizer)
    return model, tokenizer


def _default_llm_cache_dir() -> Path:
    return Path.home() / ".cache" / "anvil-audio" / "llm" / DEFAULT_LLM_MODEL_ID


def _download_default_llm_model() -> str:
    cache_dir = _default_llm_cache_dir()
    if _has_llm_files(cache_dir):
        return str(cache_dir)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the default local LLM."
        ) from exc

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=DEFAULT_LLM_REPO_ID, local_dir=str(cache_dir))
    return str(cache_dir)


def _format_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def _has_llm_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_core = (path / "config.json").exists() and (path / "tokenizer.json").exists()
    has_weights = any(path.glob("*.safetensors"))
    return has_core and has_weights


def _parse_prompt_json(text: str) -> dict[str, str]:
    text = _strip_wrapping(text)
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return {
                "prompt": str(obj.get("prompt", "")).strip(),
                "negative_prompt": str(obj.get("negative_prompt", "")).strip(),
            }
    return {}


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_single_line(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _clean_lyrics(text: str) -> str:
    text = _strip_wrapping(text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _default_negative_prompt(mode: str) -> str:
    if mode.lower() in {"music", "song", "acestep"}:
        return (
            "muddy mix, harsh clipping, distorted vocals, off-key vocals, "
            "weak drums, thin bass, noisy artifacts"
        )
    return (
        "noise, clipping, distortion, excessive reverb, unwanted music, unwanted voices"
    )
