"""ACE-Step LoRA preprocessing and training wrappers.

Anvil's dataset builder creates reviewable WAV clips plus JSON captions.
ACE-Step's corrected LoRA trainer expects a labeled dataset JSON first, then
preprocessed tensor files. This module bridges those formats and delegates the
heavy lifting to ACE-Step's own training_v2 code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LoRATrainConfig:
    """Options for launching ACE-Step training_v2 fixed LoRA training."""

    tensor_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    model_variant: str = "sft"
    base_model: str | None = None
    device: str = "auto"
    precision: str = "auto"
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation: int = 4
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    attention_type: str = "both"
    cfg_ratio: float = 0.15
    save_every: int = 10
    seed: int = 42
    num_workers: int | None = None
    yes: bool = True
    plain: bool = True
    basic_loop: bool = False


def default_checkpoint_dir() -> Path:
    """Return the ACE-Step checkpoint root used by Anvil installs."""
    return Path.home() / ".cache" / "anvil-audio" / "acestep" / "checkpoints"


def write_acestep_dataset_json(
    dataset_dir: Path,
    *,
    output_path: Path | None = None,
    custom_tag: str = "",
    lyrics: str = "[Instrumental]",
    genre: str = "",
) -> Path:
    """Write an ACE-Step training_v2 dataset JSON for an Anvil dataset.

    Args:
        dataset_dir: Directory produced by ``anvil dataset``.
        output_path: Optional JSON path. Defaults to
            ``<dataset_dir>/acestep_dataset.json``.
        custom_tag: Optional trigger tag inserted into ACE-Step prompts.
        lyrics: Default lyrics for each clip. Use ``"[Instrumental]"`` for
            instrumental style LoRAs.
        genre: Optional genre fallback for every clip.

    Returns:
        Path to the written dataset JSON.
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    captions_path = dataset_dir / "captions.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset_manifest.json: {manifest_path}")
    if not captions_path.is_file():
        raise FileNotFoundError(f"Missing captions.json: {captions_path}")

    manifest = _read_json(manifest_path)
    captions = _read_json(captions_path)
    if not isinstance(captions, list):
        raise ValueError(f"captions.json must contain a list: {captions_path}")

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(captions, start=1):
        if not isinstance(record, dict):
            continue
        rel_file = str(record.get("file") or "")
        if not rel_file:
            continue
        audio_path = (dataset_dir / rel_file).resolve()
        caption = str(record.get("caption") or record.get("prompt") or audio_path.stem)
        analysis = record.get("analysis") if isinstance(record.get("analysis"), dict) else {}
        duration = float(
            record.get("seconds_total")
            or analysis.get("duration_seconds")
            or 0.0
        )
        sample_genre = genre or ", ".join(record.get("tags", [])[:5])
        sample_filename = _sample_filename(index, rel_file)
        staged_audio_path = _stage_sample_audio(
            dataset_dir,
            source_audio=audio_path,
            sample_filename=sample_filename,
        )
        samples.append(
            {
                "filename": sample_filename,
                "audio_path": str(staged_audio_path),
                "caption": caption,
                "lyrics": lyrics,
                "genre": sample_genre,
                "duration": duration,
                "bpm": analysis.get("tempo_bpm_estimate"),
                "keyscale": "",
                "timesignature": "",
                "is_instrumental": _is_instrumental(caption, lyrics),
                "custom_tag": custom_tag,
                "prompt_override": "caption",
                "source_index": index,
            }
        )

    if not samples:
        raise RuntimeError(f"No valid clip records found in {captions_path}")

    output_path = output_path or dataset_dir / "acestep_dataset.json"
    payload = {
        "metadata": {
            "name": manifest.get("name", dataset_dir.name),
            "source_reference": manifest.get("source_reference", ""),
            "tag_position": "prepend" if custom_tag else "append",
            "genre_ratio": 0,
            "custom_tag": custom_tag,
            "generated_by": "anvil-audio",
        },
        "samples": samples,
    }
    _write_json(output_path.expanduser().resolve(), payload)
    return output_path.expanduser().resolve()


def preprocess_for_acestep(
    *,
    dataset_dir: Path,
    tensor_output: Path,
    checkpoint_dir: Path | None = None,
    model_variant: str = "sft",
    max_duration: float = 240.0,
    device: str = "auto",
    precision: str = "fp32",
    custom_tag: str = "",
    lyrics: str = "[Instrumental]",
    genre: str = "",
) -> dict[str, Any]:
    """Run ACE-Step training_v2 preprocessing for an Anvil dataset."""
    dataset_json = write_acestep_dataset_json(
        dataset_dir,
        custom_tag=custom_tag,
        lyrics=lyrics,
        genre=genre,
    )
    checkpoint_root = checkpoint_dir or default_checkpoint_dir()
    try:
        from acestep.training_v2.preprocess import preprocess_audio_files
    except ImportError as exc:
        raise RuntimeError(
            "ACE-Step training_v2 preprocessing is unavailable. Install ACE-Step "
            "with `pip install anvil-audio[acestep]` or rerun `bash install.sh`."
        ) from exc

    result = preprocess_audio_files(
        audio_dir=str((dataset_dir / "clips").resolve()),
        output_dir=str(tensor_output.expanduser().resolve()),
        checkpoint_dir=str(checkpoint_root.expanduser().resolve()),
        variant=model_variant,
        max_duration=max_duration,
        dataset_json=str(dataset_json),
        device=device,
        precision=precision,
    )
    validate_preprocessed_tensors(tensor_output)
    return result


def build_train_command(config: LoRATrainConfig) -> list[str]:
    """Build the subprocess command for ACE-Step fixed LoRA training."""
    args: list[str] = []
    if config.plain:
        args.append("--plain")
    if config.yes:
        args.append("--yes")
    args.extend(
        [
            "--checkpoint-dir",
            str(config.checkpoint_dir.expanduser().resolve()),
            "--model-variant",
            config.model_variant,
            "--dataset-dir",
            str(config.tensor_dir.expanduser().resolve()),
            "--output-dir",
            str(config.output_dir.expanduser().resolve()),
            "--device",
            config.device,
            "--precision",
            config.precision,
            "--epochs",
            str(config.epochs),
            "--lr",
            str(config.learning_rate),
            "--batch-size",
            str(config.batch_size),
            "--gradient-accumulation",
            str(config.gradient_accumulation),
            "--rank",
            str(config.rank),
            "--alpha",
            str(config.alpha),
            "--dropout",
            str(config.dropout),
            "--attention-type",
            config.attention_type,
            "--cfg-ratio",
            str(config.cfg_ratio),
            "--save-every",
            str(config.save_every),
            "--seed",
            str(config.seed),
            "--target-modules",
            *config.target_modules,
        ]
    )
    if config.base_model:
        args.extend(["--base-model", config.base_model])
    if config.num_workers is not None:
        args.extend(["--num-workers", str(config.num_workers)])

    if config.basic_loop:
        runner = (
            "from acestep.training_v2 import trainer_fixed\n"
            "trainer_fixed._FABRIC_AVAILABLE = False\n"
            "from acestep.training_v2.cli.train_fixed import main\n"
            "raise SystemExit(main())\n"
        )
        return [sys.executable, "-c", runner, *args]

    return [sys.executable, "-m", "acestep.training_v2.cli.train_fixed", *args]


def run_lora_training(config: LoRATrainConfig) -> int:
    """Launch ACE-Step fixed LoRA training and return the subprocess code."""
    command = build_train_command(config)
    result = subprocess.run(
        command,
        check=False,
        cwd=str(_training_safe_root(config)),
    )
    if result.returncode != 0:
        return result.returncode
    if not adapter_output_exists(config.output_dir):
        print(
            "ACE-Step training exited without writing a final adapter; "
            "treating the run as failed.",
            file=sys.stderr,
        )
        return 1
    if not adapter_output_is_finite(config.output_dir):
        print(
            "ACE-Step training wrote a final adapter with non-finite weights; "
            "treating the run as failed.",
            file=sys.stderr,
        )
        return 1
    return 0


def adapter_output_exists(output_dir: Path) -> bool:
    """Return true when ACE-Step wrote an inference-ready adapter."""
    final_dir = output_dir.expanduser().resolve() / "final"
    peft = (
        (final_dir / "adapter_config.json").is_file()
        and (
            (final_dir / "adapter_model.safetensors").is_file()
            or (final_dir / "adapter_model.bin").is_file()
        )
    )
    lokr = (final_dir / "lokr_weights.safetensors").is_file()
    return peft or lokr


def adapter_output_is_finite(output_dir: Path) -> bool:
    """Return true when all floating adapter tensors are finite."""
    final_dir = output_dir.expanduser().resolve() / "final"
    safetensors_path = final_dir / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        safetensors_path = final_dir / "lokr_weights.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return _all_tensors_finite(load_file(safetensors_path))

    bin_path = final_dir / "adapter_model.bin"
    if bin_path.is_file():
        import torch

        return _all_tensors_finite(torch.load(bin_path, map_location="cpu"))
    return False


def validate_preprocessed_tensors(tensor_dir: Path) -> None:
    """Raise if ACE-Step preprocessing wrote non-finite tensor values."""
    import torch

    tensor_dir = tensor_dir.expanduser().resolve()
    bad: dict[str, list[str]] = {}
    for path in sorted(tensor_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if (
                torch.is_tensor(value)
                and torch.is_floating_point(value)
                and not bool(torch.isfinite(value).all())
            ):
                bad.setdefault(path.name, []).append(key)
    if bad:
        preview = "; ".join(
            f"{name}: {', '.join(keys)}" for name, keys in list(bad.items())[:5]
        )
        raise RuntimeError(
            "ACE-Step preprocessing wrote non-finite tensors "
            f"({preview}). Retry with `--precision fp32`."
        )


def _all_tensors_finite(value: Any) -> bool:
    import torch

    if torch.is_tensor(value):
        return not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _training_safe_root(config: LoRATrainConfig) -> Path:
    """Choose a CWD compatible with ACE-Step's safe-path checks."""
    tensor_parent = config.tensor_dir.expanduser().resolve().parent
    output_parent = config.output_dir.expanduser().resolve().parent
    common = Path(os.path.commonpath([str(tensor_parent), str(output_parent)]))
    if common == Path("/"):
        common = tensor_parent
    common.mkdir(parents=True, exist_ok=True)
    return common


def _is_instrumental(caption: str, lyrics: str) -> bool:
    text = f"{caption} {lyrics}".lower()
    if lyrics.strip() and lyrics.strip().lower() != "[instrumental]":
        return False
    return not any(word in text for word in ("vocal", "voice", "singer", "lyrics"))


def _sample_filename(index: int, rel_file: str) -> str:
    path = Path(rel_file)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    stem = stem or "audio"
    suffix = path.suffix or ".wav"
    return f"{index:05d}_{stem}{suffix}"


def _stage_sample_audio(
    dataset_dir: Path,
    *,
    source_audio: Path,
    sample_filename: str,
) -> Path:
    staged_dir = dataset_dir / ".acestep_audio"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_audio = staged_dir / sample_filename
    if staged_audio.exists():
        try:
            if staged_audio.samefile(source_audio):
                return staged_audio.resolve()
        except OSError:
            pass
        staged_audio.unlink()

    try:
        os.link(source_audio, staged_audio)
    except OSError:
        try:
            staged_audio.symlink_to(source_audio)
        except OSError:
            shutil.copy2(source_audio, staged_audio)
    return staged_audio.resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
