"""python-audio-separator backend."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

from anvil_audio.separation.base import (
    EXPECTED_STEMS,
    SeparationRequest,
    SeparationResult,
)

STEM_ALIASES = {
    "vocals": ("vocals", "vocal", "voice"),
    "instrumental": ("instrumental", "inst", "no_vocals", "no-vocals"),
    "drums": ("drums", "drum"),
    "bass": ("bass",),
    "other": ("other", "accompaniment"),
}


class AudioSeparatorBackend:
    """Backend using the optional `audio-separator` package."""

    name = "audio-separator"

    @property
    def version(self) -> str:
        executable = _audio_separator_executable()
        if executable is None:
            return "not-installed"
        try:
            completed = subprocess.run(
                [str(executable), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            return "unknown"
        return completed.stdout.strip() or "unknown"

    def separate(self, request: SeparationRequest) -> SeparationResult:
        """Separate one clip with python-audio-separator."""
        start = perf_counter()
        tmp_dir = request.output_dir / ".audio_separator_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        model = _model_for_request(request)
        raw_outputs = self._separate_with_cli(request, tmp_dir, model)

        stems = _canonicalize_outputs(
            raw_outputs,
            output_dir=request.output_dir,
            expected=EXPECTED_STEMS[request.mode],
            output_format=request.output_format,
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return SeparationResult(
            backend=self.name,
            backend_version=self.version,
            model=model,
            mode=request.mode,
            input_path=request.input_path,
            stems=stems,
            elapsed_seconds=perf_counter() - start,
        )

    def _separate_with_cli(
        self,
        request: SeparationRequest,
        tmp_dir: Path,
        model: str,
    ) -> list[Path]:
        executable = _audio_separator_executable()
        if executable is None:
            raise RuntimeError(_missing_dependency_message())

        before = set(tmp_dir.glob("*"))
        command = [
            str(executable),
            str(request.input_path),
            "--output_dir",
            str(tmp_dir),
            "--output_format",
            request.output_format.upper(),
        ]
        if model != "auto":
            command.extend(["--model_filename", model])
        if request.model_file_dir is not None:
            command.extend(["--model_file_dir", str(request.model_file_dir)])
        subprocess.run(command, check=True)
        after = set(tmp_dir.glob("*"))
        return [path for path in sorted(after - before) if path.is_file()]


def _model_for_request(request: SeparationRequest) -> str:
    if request.model != "auto":
        return request.model
    if request.mode == "four-stem":
        return "htdemucs_ft.yaml"
    return "auto"


def _canonicalize_outputs(
    output_files: list[Path],
    *,
    output_dir: Path,
    expected: tuple[str, ...],
    output_format: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stems: dict[str, Path] = {}
    for stem in expected:
        source = _find_stem_file(output_files, stem)
        if source is None:
            continue
        target = output_dir / f"{stem}.{output_format.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        stems[stem] = target
    return stems


def _find_stem_file(paths: list[Path], stem: str) -> Path | None:
    aliases = STEM_ALIASES[stem]
    candidates = [path for path in paths if path.is_file()]
    for path in candidates:
        lowered = path.stem.lower()
        if any(alias in lowered for alias in aliases):
            return path
    return None


def _audio_separator_executable() -> Path | None:
    configured = os.environ.get("ANVIL_AUDIO_SEPARATOR_BIN")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    discovered = shutil.which("audio-separator")
    return Path(discovered) if discovered else None


def _missing_dependency_message() -> str:
    return (
        "audio-separator CLI is required for dataset source separation. Install "
        "it in an isolated tool environment, then set ANVIL_AUDIO_SEPARATOR_BIN "
        "to that environment's audio-separator executable. On Python 3.13, also "
        "install onnxruntime in that tool environment."
    )
