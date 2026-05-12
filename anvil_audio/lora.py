"""LoRA adapter registry and import helpers for Anvil Audio.

Adapters live outside the git checkout by default:

    ~/.cache/anvil-audio/lora/adapters/<adapter-id>/

The registry is intentionally lightweight. It records where an adapter came
from and whether ACE-Step can load it directly. Runtime injection is still
delegated to ACE-Step's own handler so we stay compatible with upstream PEFT
LoRA and LoKr/LyCORIS formats.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

AdapterFormat = Literal["peft", "lokr", "anvil-native", "unknown"]

_METADATA_FILENAME = "anvil_lora.json"
_PEFT_CONFIG = "adapter_config.json"
_PEFT_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_LOKR_WEIGHT = "lokr_weights.safetensors"
_NATIVE_METADATA = "adapter.json"
_NATIVE_WEIGHTS = "adapter.safetensors"


@dataclass(slots=True)
class LoRAAdapterEntry:
    """Metadata for one locally registered adapter."""

    id: str
    name: str
    path: str
    format: AdapterFormat
    loadable: bool
    source: str = "local"
    base_model: str = "acestep-v1.5"
    repo_id: str = ""
    revision: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoRAAdapterEntry":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            path=str(data["path"]),
            format=str(data.get("format", "unknown")),  # type: ignore[arg-type]
            loadable=bool(data.get("loadable", False)),
            source=str(data.get("source", "local")),
            base_model=str(data.get("base_model", "acestep-v1.5")),
            repo_id=str(data.get("repo_id", "")),
            revision=str(data.get("revision", "")),
            created_at=str(data.get("created_at", "")),
            notes=[str(note) for note in data.get("notes", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lora_cache_root() -> Path:
    """Return the root directory used for adapter storage."""
    raw = os.environ.get("ANVIL_AUDIO_LORA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".cache" / "anvil-audio" / "lora"


def adapters_dir() -> Path:
    """Return the directory containing registered adapter folders."""
    return lora_cache_root() / "adapters"


def list_adapters() -> list[LoRAAdapterEntry]:
    """Return all locally registered adapters, newest first."""
    root = adapters_dir()
    if not root.exists():
        return []
    entries: list[LoRAAdapterEntry] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / _METADATA_FILENAME
        if not meta_path.is_file():
            continue
        try:
            entries.append(LoRAAdapterEntry.from_dict(_read_json(meta_path)))
        except (KeyError, TypeError, json.JSONDecodeError, OSError):
            continue
    return sorted(entries, key=lambda entry: entry.created_at, reverse=True)


def get_adapter(reference: str) -> LoRAAdapterEntry | None:
    """Find an adapter by id or display name."""
    wanted = reference.strip()
    if not wanted:
        return None
    wanted_lower = wanted.lower()
    for entry in list_adapters():
        if entry.id == wanted or entry.name == wanted:
            return entry
        if entry.id.lower() == wanted_lower or entry.name.lower() == wanted_lower:
            return entry
    return None


def resolve_adapter_reference(reference: str) -> tuple[Path, LoRAAdapterEntry | None]:
    """Resolve a registry id/name or direct filesystem path to an adapter path."""
    entry = get_adapter(reference)
    if entry is not None:
        if not entry.loadable:
            notes = "; ".join(entry.notes) if entry.notes else "not loadable"
            raise ValueError(
                f"Adapter '{reference}' is registered but cannot be loaded by "
                f"ACE-Step directly ({entry.format}: {notes})."
            )
        return Path(entry.path).expanduser().resolve(), entry

    path = Path(reference).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {reference}")
    adapter_format, loadable, notes = detect_adapter_format(path)
    if not loadable:
        details = "; ".join(notes) if notes else "unsupported adapter layout"
        raise ValueError(f"LoRA adapter is not loadable by ACE-Step: {path} ({details})")
    return path.resolve(), None


def import_local_adapter(
    source_path: Path,
    *,
    name: str | None = None,
    base_model: str = "acestep-v1.5",
    copy: bool = True,
    force: bool = False,
) -> LoRAAdapterEntry:
    """Register a local adapter directory or LoKr safetensors file."""
    source = source_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {source}")

    default_name = source.stem if source.is_file() else source.name
    adapter_name = (name or default_name).strip()
    adapter_id = _unique_adapter_id(adapter_name, force=force)
    target = adapters_dir() / adapter_id
    if target.exists() and force:
        shutil.rmtree(target)
    elif target.exists():
        raise FileExistsError(f"Adapter already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / source.name)
    else:
        target = source

    return _write_entry(
        adapter_id=adapter_id,
        name=adapter_name,
        path=target,
        source="local",
        base_model=base_model,
    )


def import_hf_adapter(
    repo_id: str,
    *,
    name: str | None = None,
    revision: str | None = None,
    base_model: str = "acestep-v1.5",
    force: bool = False,
) -> LoRAAdapterEntry:
    """Download static adapter artifacts from HuggingFace and register them."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is core.
        raise RuntimeError(
            "huggingface_hub is required to import adapters from HuggingFace."
        ) from exc

    adapter_name = (name or repo_id.rsplit("/", 1)[-1]).strip()
    adapter_id = _unique_adapter_id(adapter_name, force=force)
    target = adapters_dir() / adapter_id
    if target.exists() and force:
        shutil.rmtree(target)
    elif target.exists():
        raise FileExistsError(f"Adapter already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "lokr_weights.safetensors",
            "adapter.json",
            "adapter.safetensors",
            "*.md",
            "README*",
        ],
    )

    return _write_entry(
        adapter_id=adapter_id,
        name=adapter_name,
        path=target,
        source="huggingface",
        base_model=base_model,
        repo_id=repo_id,
        revision=revision or "",
    )


def detect_adapter_format(path: Path) -> tuple[AdapterFormat, bool, list[str]]:
    """Inspect *path* and report format plus ACE-Step loadability."""
    path = path.expanduser().resolve()
    notes: list[str] = []

    if path.is_file():
        if path.name == _LOKR_WEIGHT or _safetensors_has_lokr_config(path):
            return "lokr", True, notes
        return "unknown", False, ["file adapters must be LoKr .safetensors files"]

    if not path.is_dir():
        return "unknown", False, ["path is neither a file nor a directory"]

    if (path / _PEFT_CONFIG).is_file():
        if any((path / name).is_file() for name in _PEFT_WEIGHT_FILES):
            return "peft", True, notes
        notes.append("adapter_config.json found but adapter_model weights are missing")
        return "peft", False, notes

    if (path / _LOKR_WEIGHT).is_file():
        return "lokr", True, notes
    if any(_safetensors_has_lokr_config(candidate) for candidate in path.glob("*.safetensors")):
        return "lokr", True, notes

    if (path / _NATIVE_METADATA).is_file() and (path / _NATIVE_WEIGHTS).is_file():
        return (
            "anvil-native",
            False,
            [
                "AnvilApp MLX adapter format is registered for tracking, "
                "but ACE-Step's Python runtime expects PEFT LoRA or LoKr."
            ],
        )

    return (
        "unknown",
        False,
        [
            "expected PEFT adapter_config.json plus adapter_model weights, "
            "or LoKr lokr_weights.safetensors"
        ],
    )


def _write_entry(
    *,
    adapter_id: str,
    name: str,
    path: Path,
    source: str,
    base_model: str,
    repo_id: str = "",
    revision: str = "",
) -> LoRAAdapterEntry:
    adapter_format, loadable, notes = detect_adapter_format(path)
    entry = LoRAAdapterEntry(
        id=adapter_id,
        name=name,
        path=str(path.expanduser().resolve()),
        format=adapter_format,
        loadable=loadable,
        source=source,
        base_model=base_model,
        repo_id=repo_id,
        revision=revision,
        notes=notes,
    )
    metadata_dir = adapters_dir() / adapter_id
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_dir / _METADATA_FILENAME, entry.to_dict())
    return entry


def _unique_adapter_id(name: str, *, force: bool) -> str:
    base = _slugify(name)
    if force:
        return base
    root = adapters_dir()
    if not (root / base).exists():
        return base
    for idx in range(2, 10_000):
        candidate = f"{base}-{idx}"
        if not (root / candidate).exists():
            return candidate
    raise RuntimeError(f"Could not create a unique adapter id for {name!r}")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "adapter"


def _safetensors_has_lokr_config(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return False
    raw = metadata.get("lokr_config")
    return isinstance(raw, str) and bool(raw.strip())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
