"""Local prompt and lyric intelligence helpers.

This module mirrors AnvilApp's local-first intelligence path: a small MLX Llama
model expands short audio prompts and writes duration-aware lyrics without
calling an external API.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import importlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LLM_MODEL_ID = "llama-3.2-3b-instruct-4bit"
DEFAULT_LLM_REPO_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MLX_EXECUTOR: ThreadPoolExecutor | None = None
_MLX_EXECUTOR_LOCK = threading.Lock()
_MLX_WORKER_THREAD_ID: int | None = None


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
    max_lyric_lines: int

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
                max_lyric_lines=6,
            )
        if seconds <= 60:
            return cls(
                duration_seconds=seconds,
                line_budget="6 to 10 total lyric lines",
                section_plan=(
                    "Use exactly these section markers in this order: "
                    "[Verse 1], [Chorus]. Keep each section short. Do not use "
                    "[Bridge]."
                ),
                max_tokens=140,
                max_lyric_lines=10,
            )
        if seconds <= 90:
            return cls(
                duration_seconds=seconds,
                line_budget="10 to 14 total lyric lines",
                section_plan=(
                    "Use exactly these section markers in this order: "
                    "[Verse 1], [Chorus], optional [Verse 2]. Do not use [Bridge] "
                    "unless it replaces [Verse 2]."
                ),
                max_tokens=200,
                max_lyric_lines=14,
            )
        if seconds <= 150:
            return cls(
                duration_seconds=seconds,
                line_budget="14 to 20 total lyric lines",
                section_plan=(
                    "Use exactly these section markers in this order: "
                    "[Verse 1], [Chorus], [Verse 2], [Chorus], optional [Bridge]. "
                    "Keep the bridge to 2 lines."
                ),
                max_tokens=300,
                max_lyric_lines=20,
            )
        return cls(
            duration_seconds=seconds,
            line_budget="20 to 32 total lyric lines",
            section_plan=(
                "Use exactly these section markers in this order: [Verse 1], "
                "[Chorus], [Verse 2], [Chorus], [Bridge], optional [Outro]. "
                "Keep sections compact and avoid repeated filler."
            ),
            max_tokens=420,
            max_lyric_lines=32,
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
            "Use specific, genre-aware images that feel grounded in the song "
            "description. Avoid default lyric cliches like shadows, darkness, "
            "light, fire, rain, broken hearts, and finding my way unless the "
            "user specifically asks for them.\n"
            "End on a complete line. Never trail off mid-sentence.\n"
            "The first non-empty character of your answer must be '['.\n"
            "Every requested section marker must appear on its own line. Do not "
            "collapse the whole song into one [Verse].\n"
            "Do not put more than 6 lyric lines under a single section marker.\n"
            "Output only section markers and singable lyric lines. No commentary, "
            "no chord notes, and no parenthetical performance directions."
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
        return _generate_mlx_text_on_worker(
            self.model_ref,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


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
        "Rewrite the user's idea into 18 to 28 concise comma-separated audio "
        "tags, not a sentence and not an instruction. Do not start with words "
        "like Create, Generate, Make, Write, or Avoid. Do not echo the original "
        "wording unchanged. Include genre or subgenre, instruments, vocal "
        "character, tempo, key or mood, arrangement, texture, room/space, and "
        "production style when relevant. Do not mention the target duration in "
        "the final prompt.\n"
        "The negative_prompt must list unwanted artifacts or qualities such as "
        "muddy mix, clipping, harsh treble, weak drums, noisy artifacts, or "
        "off-key vocals as comma-separated tags. Do not start it with Avoid. "
        "Avoid banning the requested genre, instruments, or vocals.\n"
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
    enhanced = _compact_prompt_tags(
        _clean_prompt_line(parsed.get("prompt") or _strip_wrapping(raw) or prompt)
    )
    if _is_weak_enhancement(prompt, enhanced):
        enhanced = _fallback_enhanced_prompt(prompt, mode)
    suggested_negative = (
        parsed.get("negative_prompt")
        or negative_prompt
        or _default_negative_prompt(mode)
    )
    return PromptPackage(
        prompt=enhanced,
        negative_prompt=_clean_negative_prompt(suggested_negative),
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
    return _clean_lyrics(
        text,
        max_lyric_lines=plan.max_lyric_lines,
        target_duration_seconds=plan.duration_seconds,
    )


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


def _generate_mlx_text_on_worker(
    model_ref: str,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Run MLX-LM generation on one stable thread.

    MLX streams are thread-local. Gradio can run button callbacks on different
    worker threads, so sharing a cached MLX model directly across callbacks can
    fail with "There is no Stream(gpu, N) in current thread." Keeping all local
    LLM loads and generations on one executor thread keeps the model and stream
    ownership aligned.
    """
    if threading.get_ident() == _MLX_WORKER_THREAD_ID:
        return _generate_mlx_text(
            model_ref,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    executor = _get_mlx_executor()
    future = executor.submit(
        _generate_mlx_text,
        model_ref,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return future.result()


def _get_mlx_executor() -> ThreadPoolExecutor:
    global _MLX_EXECUTOR
    with _MLX_EXECUTOR_LOCK:
        if _MLX_EXECUTOR is None:
            _MLX_EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="anvil-mlx-llm",
                initializer=_mark_mlx_worker_thread,
            )
        return _MLX_EXECUTOR


def _mark_mlx_worker_thread() -> None:
    global _MLX_WORKER_THREAD_ID
    _MLX_WORKER_THREAD_ID = threading.get_ident()


def _generate_mlx_text(
    model_ref: str,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    model, tokenizer = _load_mlx_model(model_ref)
    prompt = _format_chat_prompt(tokenizer, system_prompt, user_prompt)

    generate_module = importlib.import_module("mlx_lm.generate")
    _bind_mlx_lm_generation_stream(generate_module)
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=0.9)
    text = generate_module.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    return str(text).strip()


def _bind_mlx_lm_generation_stream(generate_module: Any) -> None:
    """Bind mlx-lm's module-level generation stream to the current thread."""
    import mlx.core as mx

    generate_module.generation_stream = mx.new_stream(mx.default_device())


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
    return _extract_prompt_json_fields(text)


def _extract_prompt_json_fields(text: str) -> dict[str, str]:
    """Best-effort extraction for truncated local LLM JSON."""
    result: dict[str, str] = {}
    for key in ("prompt", "negative_prompt"):
        match = re.search(
            rf'"{key}"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)',
            text,
            flags=re.DOTALL,
        )
        if not match:
            continue
        value = match.group("value")
        value = re.sub(r'"\s*,\s*"(?:prompt|negative_prompt)"\s*:\s*.*$', "", value)
        try:
            result[key] = json.loads(f'"{value}"').strip()
        except json.JSONDecodeError:
            result[key] = value.replace('\\"', '"').strip()
    return result


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_single_line(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _clean_prompt_line(text: str) -> str:
    text = _clean_single_line(text)
    text = re.sub(r'^\s*\{?\s*"prompt"\s*:\s*"?', "", text, flags=re.IGNORECASE)
    text = re.sub(r'"\s*,\s*"negative_prompt"\s*:\s*".*$', "", text)
    text = re.sub(r'"\s*\}?\s*$', "", text)
    text = re.split(r"\b(?:and\s+)?avoid\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"\b\d+\s*-?\s*(?:seconds?|secs?)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:create|generate|make|write)\s+(?:a|an)?\s*(?P<desc>.+?)\s+"
        r"(?:music\s+)?(?:piece|track|song)\s+(?:with|featuring)\s+",
        lambda match: f"{match.group('desc')}, ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:create|generate|make|write)\s+(?:a|an)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:featuring|using|with a focus on|with an emphasis on|with)\b",
        ", ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:and|plus)\b", ", ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def _compact_prompt_tags(text: str, max_tags: int = 28) -> str:
    tags = [tag.strip() for tag in re.split(r",|;", text) if tag.strip()]
    if len(tags) <= 1:
        return text

    clean_tags: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        tag = _clean_prompt_line(tag)
        tag = re.sub(
            r"^(?:a|an|the|and|with|featuring|using)\s+",
            "",
            tag,
            flags=re.IGNORECASE,
        )
        tag = re.sub(r"^mix of\s+", "", tag, flags=re.IGNORECASE)
        if not tag:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", tag.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        clean_tags.append(tag)
        if len(clean_tags) >= max_tags:
            break
    return ", ".join(clean_tags)


def _clean_negative_prompt(text: str) -> str:
    text = _clean_single_line(text)
    text = re.sub(r"^(?:avoid|no|without)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:with a focus on|while also|and avoiding)\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    return text.strip(" .,")


def _is_weak_enhancement(original: str, enhanced: str) -> bool:
    original_clean = _clean_single_line(original).lower()
    enhanced_clean = _clean_prompt_line(enhanced).lower()
    if not enhanced_clean:
        return True
    if enhanced_clean == original_clean:
        return True

    original_words = set(re.findall(r"[a-z0-9']+", original_clean))
    enhanced_words = set(re.findall(r"[a-z0-9']+", enhanced_clean))
    added_words = enhanced_words - original_words
    similarity = SequenceMatcher(None, original_clean, enhanced_clean).ratio()
    return similarity > 0.9 and len(added_words) <= 3


def _fallback_enhanced_prompt(prompt: str, mode: str) -> str:
    text = _clean_single_line(prompt)
    lowered = text.lower()
    tags = [text]

    keyword_tags = [
        (("blues",), ["dark blues", "raw blues guitar", "smoky club ambience"]),
        (("rock",), ["guitar-driven rock arrangement", "live drums", "warm bass"]),
        (("vocal", "vocals", "singer"), ["expressive lead vocal", "natural vocal phrasing"]),
        (("male",), ["gritty male vocal"]),
        (("female",), ["emotive female vocal"]),
        (("slow",), ["slow tempo", "laid-back groove"]),
        (("fast", "upbeat"), ["driving tempo", "forward momentum"]),
        (("minor", "dark"), ["minor-key harmony", "brooding mood", "melancholic tension"]),
        (("guitar",), ["raw guitar tone", "expressive bends"]),
        (("atmospheric", "ambient"), ["wide atmospheric reverb", "spacious texture"]),
        (("cinematic",), ["cinematic build", "wide dynamic range"]),
        (("electronic", "synth"), ["layered synth texture", "clean electronic production"]),
        (("drum", "drums"), ["punchy drum kit", "tight transient detail"]),
        (("bass",), ["warm low end", "defined bass movement"]),
    ]
    for keywords, additions in keyword_tags:
        if any(keyword in lowered for keyword in keywords):
            tags.extend(additions)

    if len(tags) == 1:
        if mode.lower() in {"music", "song", "acestep"}:
            tags.extend(
                [
                    "clear musical arrangement",
                    "defined instrumentation",
                    "natural dynamics",
                    "balanced modern mix",
                ]
            )
        else:
            tags.extend(
                [
                    "clean sound design",
                    "detailed texture",
                    "focused transient detail",
                    "balanced mix",
                ]
            )

    tags.extend(["organic performance", "intimate room tone", "polished but natural mix"])
    return _dedupe_tags(tags)


def _dedupe_tags(tags: list[str]) -> str:
    seen: set[str] = set()
    clean_tags: list[str] = []
    for tag in tags:
        clean = _clean_prompt_line(tag)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            clean_tags.append(clean)
    return ", ".join(clean_tags)


def _clean_lyrics(
    text: str,
    max_lyric_lines: int | None = None,
    target_duration_seconds: int | None = None,
) -> str:
    text = _strip_wrapping(text)
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if not re.fullmatch(r"\s*\([^)]*\)\s*", line)
    ]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if max_lyric_lines is not None:
        lines = _limit_lyric_lines(lines, max_lyric_lines)
    lines = _repair_oversized_single_verse(lines, target_duration_seconds)
    return "\n".join(lines).strip()


def _limit_lyric_lines(lines: list[str], max_lyric_lines: int) -> list[str]:
    limited: list[str] = []
    lyric_count = 0
    for line in lines:
        stripped = line.strip()
        is_section = bool(re.fullmatch(r"\[[^\]]+\]", stripped))
        if not stripped:
            if limited and limited[-1].strip():
                limited.append(line)
            continue
        if is_section:
            if lyric_count < max_lyric_lines:
                limited.append(line)
            continue
        if lyric_count >= max_lyric_lines:
            continue
        limited.append(line)
        lyric_count += 1

    while limited and (not limited[-1].strip() or re.fullmatch(r"\[[^\]]+\]", limited[-1].strip())):
        limited.pop()
    return limited


def _repair_oversized_single_verse(
    lines: list[str], target_duration_seconds: int | None
) -> list[str]:
    non_blank_lines = [line for line in lines if line.strip()]
    section_markers = [line.strip() for line in non_blank_lines if _is_section_marker(line)]
    if len(section_markers) != 1 or not _is_generic_verse_marker(section_markers[0]):
        return lines

    lyric_lines = [line for line in non_blank_lines if not _is_section_marker(line)]
    threshold = _oversized_verse_threshold(target_duration_seconds)
    if len(lyric_lines) < threshold:
        return lines

    markers = _repair_section_markers(target_duration_seconds, len(lyric_lines))
    if len(markers) <= 1:
        return lines

    repaired: list[str] = []
    lyric_index = 0
    base_count = len(lyric_lines) // len(markers)
    extra_count = len(lyric_lines) % len(markers)
    for marker_index, marker in enumerate(markers):
        if repaired and repaired[-1] != "":
            repaired.append("")
        repaired.append(marker)
        section_line_count = base_count + (1 if marker_index < extra_count else 0)
        for _ in range(section_line_count):
            if lyric_index >= len(lyric_lines):
                break
            repaired.append(lyric_lines[lyric_index])
            lyric_index += 1

    return _collapse_blank_lines(repaired)


def _is_section_marker(line: str) -> bool:
    return bool(re.fullmatch(r"\[[^\]]+\]", line.strip()))


def _is_generic_verse_marker(line: str) -> bool:
    normalized = line.strip().lower()
    return normalized in {"[verse]", "[verse 1]"}


def _oversized_verse_threshold(target_duration_seconds: int | None) -> int:
    if target_duration_seconds is None:
        return 16
    if target_duration_seconds < 60:
        return 10
    if target_duration_seconds < 90:
        return 12
    if target_duration_seconds < 150:
        return 14
    return 12


def _repair_section_markers(
    target_duration_seconds: int | None, lyric_line_count: int
) -> list[str]:
    if (target_duration_seconds or 0) >= 150 or lyric_line_count >= 16:
        if lyric_line_count >= 28:
            return [
                "[Verse 1]",
                "[Chorus]",
                "[Verse 2]",
                "[Chorus]",
                "[Bridge]",
                "[Chorus]",
                "[Outro]",
            ]
        if lyric_line_count >= 22:
            return [
                "[Verse 1]",
                "[Chorus]",
                "[Verse 2]",
                "[Chorus]",
                "[Bridge]",
                "[Chorus]",
            ]
        return ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]"]

    if (target_duration_seconds or 0) >= 90 or lyric_line_count >= 10:
        return ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]"]

    return ["[Verse 1]", "[Chorus]"]


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    for line in lines:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)
    return collapsed


def _default_negative_prompt(mode: str) -> str:
    if mode.lower() in {"music", "song", "acestep"}:
        return (
            "muddy mix, harsh clipping, distorted vocals, off-key vocals, "
            "weak drums, thin bass, noisy artifacts"
        )
    return (
        "noise, clipping, distortion, excessive reverb, unwanted music, unwanted voices"
    )
