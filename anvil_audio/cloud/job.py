"""Portable cloud training job packages."""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CLOUD_JOB_VERSION = "0.1"
DEFAULT_RECIPE = "lora-balanced"
DEFAULT_REPO_URL = "https://github.com/PiMPStudios/anvil-audio-v2.git"
ASSET_NAMES = ("full-mix", "instrumental", "vocals", "drums", "bass", "other")


@dataclass(frozen=True, slots=True)
class LoRARecipe:
    """Cloud training recipe values passed to `anvil lora train`."""

    name: str
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    rank: int
    alpha: int
    dropout: float
    attention_type: str
    cfg_ratio: float
    save_every: int
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    precision: str = "auto"
    preprocess_precision: str = "fp32"
    max_duration: float = 240.0
    seed: int = 42


RECIPES: dict[str, LoRARecipe] = {
    "lora-smoke": LoRARecipe(
        name="lora-smoke",
        epochs=5,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=2,
        rank=16,
        alpha=32,
        dropout=0.1,
        attention_type="both",
        cfg_ratio=0.15,
        save_every=1,
    ),
    "lora-balanced": LoRARecipe(
        name="lora-balanced",
        epochs=100,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=4,
        rank=64,
        alpha=128,
        dropout=0.1,
        attention_type="both",
        cfg_ratio=0.15,
        save_every=10,
    ),
    "lora-quality": LoRARecipe(
        name="lora-quality",
        epochs=150,
        learning_rate=7.5e-5,
        batch_size=1,
        gradient_accumulation=4,
        rank=96,
        alpha=192,
        dropout=0.08,
        attention_type="both",
        cfg_ratio=0.15,
        save_every=10,
    ),
}


@dataclass(slots=True)
class CloudJobPackageConfig:
    """Configuration for building a portable training job folder."""

    training_bundle: Path
    output_dir: Path
    name: str | None = None
    base_model: str = "acestep-v1.5-sft"
    model_variant: str = "sft"
    recipe: str = DEFAULT_RECIPE
    primary_asset: str = "full-mix"
    max_hours: float = 6.0
    checkpoint_dir: str = "~/.cache/anvil-audio/acestep/checkpoints"
    repo_url: str | None = None
    repo_ref: str | None = None
    force: bool = False


@dataclass(slots=True)
class CloudJobPackageResult:
    """Result from building a cloud job package."""

    job_dir: Path
    job_path: Path
    dataset_dir: Path
    clip_count: int
    asset_count: int
    warnings: list[str] = field(default_factory=list)


def available_recipes() -> tuple[str, ...]:
    """Return the supported recipe names."""
    return tuple(sorted(RECIPES))


def create_cloud_job_package(config: CloudJobPackageConfig) -> CloudJobPackageResult:
    """Create a portable training job folder from `training_bundle.json`."""
    bundle_path = config.training_bundle.expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing training bundle: {bundle_path}")

    bundle = _read_json_object(bundle_path)
    recipe = RECIPES.get(config.recipe)
    if recipe is None:
        raise ValueError(
            f"Unknown cloud training recipe: {config.recipe}. "
            f"Available: {', '.join(available_recipes())}."
        )

    primary_asset = config.primary_asset.strip().lower()
    if primary_asset not in ASSET_NAMES:
        raise ValueError(
            f"Unsupported primary asset: {config.primary_asset}. "
            f"Allowed: {', '.join(ASSET_NAMES)}."
        )

    source_dataset_dir = Path(str(bundle.get("dataset_dir") or "")).expanduser()
    if not source_dataset_dir.is_absolute():
        source_dataset_dir = (bundle_path.parent / source_dataset_dir).resolve()
    else:
        source_dataset_dir = source_dataset_dir.resolve()
    if not source_dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing bundle dataset_dir: {source_dataset_dir}")

    job_dir = _resolve_job_dir(config, bundle)
    if job_dir.exists():
        if not config.force:
            raise FileExistsError(f"Cloud job already exists: {job_dir}")
        shutil.rmtree(job_dir)
    (job_dir / "inputs" / "dataset").mkdir(parents=True)
    (job_dir / "outputs").mkdir()
    (job_dir / "logs").mkdir()
    (job_dir / "scripts").mkdir()
    (job_dir / "work").mkdir()

    clips = _as_list(bundle.get("clips"))
    copied_records: list[dict[str, Any]] = []
    copied_assets = 0
    warnings: list[str] = []
    job_dataset_dir = job_dir / "inputs" / "dataset"
    for index, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            continue
        assets = clip.get("assets") if isinstance(clip.get("assets"), dict) else {}
        asset_rel = str(assets.get(primary_asset) or "")
        if not asset_rel:
            warnings.append(
                f"clip_{index:04d}: missing requested primary asset {primary_asset}"
            )
            continue
        source_asset = (source_dataset_dir / asset_rel).resolve()
        if not source_asset.is_file():
            warnings.append(f"clip_{index:04d}: missing asset file {asset_rel}")
            continue
        destination = job_dataset_dir / asset_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, destination)
        copied_assets += 1
        copied_records.append(_caption_record_from_bundle_clip(index, clip, asset_rel))

    if not copied_records:
        raise RuntimeError(
            f"No clips with primary asset {primary_asset} were available to package."
        )

    _copy_optional_metadata(source_dataset_dir, job_dataset_dir)
    _write_json(job_dataset_dir / "captions.json", copied_records)
    _write_json(
        job_dataset_dir / "dataset_manifest.json",
        _job_dataset_manifest(bundle, source_dataset_dir, primary_asset),
    )

    shutil.copy2(bundle_path, job_dir / "training_bundle.json")
    job_payload = _job_payload(
        config=config,
        bundle=bundle,
        bundle_path=bundle_path,
        job_dir=job_dir,
        recipe=recipe,
        primary_asset=primary_asset,
        source_dataset_dir=source_dataset_dir,
        copied_records=copied_records,
        copied_assets=copied_assets,
        warnings=warnings,
    )
    job_path = job_dir / "job.json"
    _write_json(job_path, job_payload)
    _write_scripts(job_dir, job_payload)

    return CloudJobPackageResult(
        job_dir=job_dir,
        job_path=job_path,
        dataset_dir=job_dataset_dir,
        clip_count=len(copied_records),
        asset_count=copied_assets,
        warnings=warnings,
    )


def _resolve_job_dir(config: CloudJobPackageConfig, bundle: dict[str, Any]) -> Path:
    output_dir = config.output_dir.expanduser()
    if output_dir.name:
        return output_dir.resolve()
    dataset_name = str(bundle.get("dataset_name") or "anvil_dataset")
    name = config.name or dataset_name
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "cloud_jobs" / f"{_slugify(name)}_{stamp}").resolve()


def _caption_record_from_bundle_clip(
    index: int, clip: dict[str, Any], asset_rel: str
) -> dict[str, Any]:
    caption = str(clip.get("caption") or clip.get("prompt") or f"clip {index}")
    return {
        "file": asset_rel,
        "source_file": str(clip.get("file") or ""),
        "caption": caption,
        "prompt": str(clip.get("prompt") or caption),
        "tags": _as_string_list(clip.get("tags")),
        "negative_tags": _as_string_list(clip.get("negative_tags")),
        "confidence": _as_float(clip.get("confidence"), default=0.0),
        "seconds_start": _as_float(clip.get("seconds_start"), default=0.0),
        "seconds_total": _as_float(clip.get("seconds_total"), default=0.0),
        "cloud_job_source_id": str(clip.get("id") or f"clip_{index:04d}"),
    }


def _copy_optional_metadata(source_dataset_dir: Path, job_dataset_dir: Path) -> None:
    for name in ("character_sheet.json", "dataset_config.json"):
        source = source_dataset_dir / name
        if source.is_file():
            shutil.copy2(source, job_dataset_dir / name)


def _job_dataset_manifest(
    bundle: dict[str, Any], source_dataset_dir: Path, primary_asset: str
) -> dict[str, Any]:
    return {
        "anvil_dataset_version": "cloud-job-0.1",
        "name": str(bundle.get("dataset_name") or source_dataset_dir.name),
        "source_dataset_dir": str(source_dataset_dir),
        "source_training_bundle": "training_bundle.json",
        "primary_asset": primary_asset,
        "generated_by": "anvil cloud package",
    }


def _job_payload(
    *,
    config: CloudJobPackageConfig,
    bundle: dict[str, Any],
    bundle_path: Path,
    job_dir: Path,
    recipe: LoRARecipe,
    primary_asset: str,
    source_dataset_dir: Path,
    copied_records: list[dict[str, Any]],
    copied_assets: int,
    warnings: list[str],
) -> dict[str, Any]:
    max_seconds = max(1, int(config.max_hours * 3600))
    repo_url = config.repo_url or _detect_git_remote() or DEFAULT_REPO_URL
    repo_ref = config.repo_ref or _detect_git_ref() or "main"
    return {
        "anvil_cloud_job_version": CLOUD_JOB_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "name": job_dir.name,
        "source_training_bundle": str(bundle_path),
        "source_dataset_dir": str(source_dataset_dir),
        "dataset_name": str(bundle.get("dataset_name") or source_dataset_dir.name),
        "primary_asset": primary_asset,
        "clip_count": len(copied_records),
        "asset_count": copied_assets,
        "warnings": warnings,
        "training": {
            "base_model": config.base_model,
            "model_variant": config.model_variant,
            "recipe": recipe.name,
            "max_hours": config.max_hours,
            "max_seconds": max_seconds,
            "checkpoint_dir": config.checkpoint_dir,
            "device": "cuda",
            "lyrics": "[Instrumental]",
            "genre": str(bundle.get("dataset_name") or ""),
            "recipe_args": {
                "epochs": recipe.epochs,
                "learning_rate": recipe.learning_rate,
                "batch_size": recipe.batch_size,
                "gradient_accumulation": recipe.gradient_accumulation,
                "rank": recipe.rank,
                "alpha": recipe.alpha,
                "dropout": recipe.dropout,
                "attention_type": recipe.attention_type,
                "cfg_ratio": recipe.cfg_ratio,
                "save_every": recipe.save_every,
                "target_modules": list(recipe.target_modules),
                "precision": recipe.precision,
                "preprocess_precision": recipe.preprocess_precision,
                "max_duration": recipe.max_duration,
                "seed": recipe.seed,
            },
        },
        "runtime": {
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "install_spec": f"anvil-audio[acestep] @ git+{repo_url}@{repo_ref}",
        },
        "paths": {
            "dataset_dir": "inputs/dataset",
            "tensor_dir": "work/tensors",
            "output_dir": "outputs/lora",
            "logs_dir": "logs",
        },
    }


def _write_scripts(job_dir: Path, job: dict[str, Any]) -> None:
    scripts = job_dir / "scripts"
    _write_executable(scripts / "bootstrap.sh", _bootstrap_script(job))
    _write_executable(scripts / "run_training.sh", _run_training_script(job))
    _write_executable(scripts / "collect.sh", _collect_script())


def _bootstrap_script(job: dict[str, Any]) -> str:
    install_spec = str(job["runtime"]["install_spec"])
    return f"""#!/usr/bin/env bash
set -euo pipefail

JOB_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "$JOB_ROOT"

PYTHON_BIN="${{PYTHON_BIN:-python3}}"
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
ANVIL_AUDIO_INSTALL="${{ANVIL_AUDIO_INSTALL:-{install_spec}}}"
python -m pip install "$ANVIL_AUDIO_INSTALL"
anvil setup || true
"""


def _run_training_script(job: dict[str, Any]) -> str:
    training = job["training"]
    recipe = training["recipe_args"]
    checkpoint_dir = str(training["checkpoint_dir"])
    target_modules = " ".join(str(item) for item in recipe["target_modules"])
    return f"""#!/usr/bin/env bash
set -euo pipefail

JOB_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "$JOB_ROOT"

source .venv/bin/activate
mkdir -p work/tensors outputs/lora logs

DATASET_DIR="$JOB_ROOT/inputs/dataset"
TENSOR_DIR="$JOB_ROOT/work/tensors"
OUTPUT_DIR="$JOB_ROOT/outputs/lora"
CHECKPOINT_DIR="${{ANVIL_CHECKPOINT_DIR:-{checkpoint_dir}}}"
MAX_SECONDS="${{ANVIL_MAX_SECONDS:-{int(training["max_seconds"])}}}"

echo "Preprocessing dataset: $DATASET_DIR"
anvil lora preprocess "$DATASET_DIR" \\
  --output-dir "$TENSOR_DIR" \\
  --checkpoint-dir "$CHECKPOINT_DIR" \\
  --model-variant {training["model_variant"]} \\
  --max-duration {recipe["max_duration"]} \\
  --device cuda \\
  --precision {recipe["preprocess_precision"]} \\
  --lyrics "[Instrumental]" \\
  --genre "{training["genre"]}" 2>&1 | tee logs/preprocess.log

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD=(timeout "$MAX_SECONDS")
else
  TIMEOUT_CMD=()
fi

echo "Training LoRA recipe: {training["recipe"]}"
"${{TIMEOUT_CMD[@]}}" anvil lora train "$TENSOR_DIR" \\
  --output-dir "$OUTPUT_DIR" \\
  --checkpoint-dir "$CHECKPOINT_DIR" \\
  --model-variant {training["model_variant"]} \\
  --device cuda \\
  --precision {recipe["precision"]} \\
  --epochs {recipe["epochs"]} \\
  --lr {recipe["learning_rate"]} \\
  --batch-size {recipe["batch_size"]} \\
  --gradient-accumulation {recipe["gradient_accumulation"]} \\
  --rank {recipe["rank"]} \\
  --alpha {recipe["alpha"]} \\
  --dropout {recipe["dropout"]} \\
  --attention-type {recipe["attention_type"]} \\
  --cfg-ratio {recipe["cfg_ratio"]} \\
  --save-every {recipe["save_every"]} \\
  --seed {recipe["seed"]} \\
  --target-modules {target_modules} 2>&1 | tee logs/train.log
"""


def _collect_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

JOB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$JOB_ROOT"

mkdir -p outputs
tar -czf outputs/anvil_cloud_results.tar.gz job.json training_bundle.json logs outputs/lora
echo "Results archive: $JOB_ROOT/outputs/anvil_cloud_results.tar.gz"
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _detect_git_remote() -> str | None:
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if not value:
        return None
    if value.startswith("git@github.com:"):
        repo = value.removeprefix("git@github.com:").removesuffix(".git")
        return f"https://github.com/{repo}.git"
    return value


def _detect_git_ref() -> str | None:
    for command in (
        ["git", "branch", "--show-current"],
        ["git", "rev-parse", "--short", "HEAD"],
    ):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        value = result.stdout.strip()
        if value:
            return value
    return None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "anvil-cloud-job"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
