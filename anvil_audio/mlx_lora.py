"""PEFT LoRA bridge for ACE-Step's native MLX DiT decoder.

The upstream ACE-Step LoRA path wraps the PyTorch decoder with PEFT.  On Apple
Silicon that forces the large DiT out of the native MLX path, which is exactly
where XL checkpoints tend to run out of memory.  This module keeps the base DiT
inside MLX and applies PEFT LoRA adapter matrices directly to matching MLX
``nn.Linear`` projections at inference time.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_PEFT_CONFIG = "adapter_config.json"
_PEFT_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_LORA_KEY_RE = re.compile(
    r"^(?P<module>.+?)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)
_PEFT_PREFIXES = (
    "base_model.model.",
    "base_model.",
    "model.",
    "decoder.",
    "transformer.",
    "dit.",
)


@dataclass(slots=True)
class MLXLoRAModule:
    """One PEFT LoRA contribution targeting a single MLX Linear module."""

    module_path: str
    lora_a: np.ndarray
    lora_b: np.ndarray
    rank: int
    alpha: float
    scale: float
    user_scale: float


@dataclass(slots=True)
class MLXLoRAAdapter:
    """Loaded PEFT adapter normalized for MLX application."""

    name: str
    path: Path
    modules: list[MLXLoRAModule]
    peft_config: dict[str, Any]


def is_peft_lora_dir(path: str | Path) -> bool:
    """Return whether *path* looks like a PEFT LoRA adapter directory."""
    adapter_path = Path(path).expanduser()
    return (
        adapter_path.is_dir()
        and (adapter_path / _PEFT_CONFIG).is_file()
        and any((adapter_path / name).is_file() for name in _PEFT_WEIGHT_FILES)
    )


def normalize_peft_module_path(module_path: str) -> str:
    """Map a PEFT state-dict module prefix to an ACE-Step MLX module path."""
    normalized = module_path.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _PEFT_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized.replace(".base_layer", "")


def parse_peft_lora_key(key: str) -> tuple[str, str] | None:
    """Return ``(module_path, side)`` for PEFT LoRA A/B keys, else ``None``."""
    match = _LORA_KEY_RE.match(key)
    if match is None:
        return None
    return normalize_peft_module_path(match.group("module")), match.group("side")


def load_peft_lora_adapter(
    adapter_path: str | Path,
    *,
    adapter_name: str | None = None,
    scale: float = 1.0,
) -> MLXLoRAAdapter:
    """Load and normalize a PEFT LoRA adapter without importing PEFT or torch.

    ``adapter_model.safetensors`` is preferred so the bridge can remain cheap
    and CPU-light.  ``adapter_model.bin`` is accepted as a fallback because some
    older PEFT exports use it.
    """
    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"MLX LoRA expects a PEFT adapter directory: {path}")

    config_path = path / _PEFT_CONFIG
    if not config_path.is_file():
        raise ValueError(f"Missing PEFT adapter_config.json: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PEFT adapter_config.json: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"PEFT adapter_config.json must be an object: {config_path}")

    weights = _load_peft_weights(path)
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, value in weights.items():
        parsed = parse_peft_lora_key(key)
        if parsed is None:
            continue
        module_path, side = parsed
        grouped.setdefault(module_path, {})[side] = _as_numpy(value)

    modules: list[MLXLoRAModule] = []
    for module_path, pair in sorted(grouped.items()):
        lora_a = pair.get("A")
        lora_b = pair.get("B")
        if lora_a is None or lora_b is None:
            continue
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {module_path!r} must be rank-2 matrices "
                f"(got A={lora_a.shape}, B={lora_b.shape})"
            )
        rank = int(lora_a.shape[0])
        if rank <= 0 or int(lora_b.shape[1]) != rank:
            raise ValueError(
                f"LoRA tensor rank mismatch for {module_path!r}: "
                f"A={lora_a.shape}, B={lora_b.shape}"
            )
        alpha = _module_alpha(config, module_path, default=float(rank))
        scale_base = alpha / math.sqrt(rank) if bool(config.get("use_rslora")) else alpha / rank
        modules.append(
            MLXLoRAModule(
                module_path=module_path,
                lora_a=lora_a.astype(np.float32, copy=False),
                lora_b=lora_b.astype(np.float32, copy=False),
                rank=rank,
                alpha=float(alpha),
                scale=float(scale) * float(scale_base),
                user_scale=float(scale),
            )
        )

    if not modules:
        raise ValueError(f"No PEFT LoRA A/B weights found in adapter: {path}")

    return MLXLoRAAdapter(
        name=(adapter_name or path.name).strip() or path.name,
        path=path,
        modules=modules,
        peft_config=config,
    )


def apply_lora_stack_to_mlx_decoder(
    mlx_decoder: Any,
    adapters: Sequence[Any],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Apply a stack of PEFT LoRA adapters to an ACE-Step MLX decoder."""
    if mlx_decoder is None:
        raise RuntimeError("ACE-Step MLX decoder is not initialized.")

    clear_mlx_lora_stack(mlx_decoder)
    if not adapters:
        return get_mlx_lora_status(mlx_decoder)

    loaded: list[MLXLoRAAdapter] = []
    for item in adapters:
        loaded.append(
            load_peft_lora_adapter(
                _entry_value(item, "path"),
                adapter_name=_entry_value(item, "adapter_name", None),
                scale=float(_entry_value(item, "scale", 1.0)),
            )
        )

    applied_modules = 0
    try:
        for adapter in loaded:
            applied_modules += _apply_adapter_to_decoder(mlx_decoder, adapter)
    except Exception:
        clear_mlx_lora_stack(mlx_decoder)
        raise

    set_mlx_lora_enabled(mlx_decoder, enabled)
    status = get_mlx_lora_status(mlx_decoder)
    status.update(
        {
            "message": (
                f"Applied {len(loaded)} LoRA adapter"
                f"{'' if len(loaded) == 1 else 's'} to MLX DiT"
            ),
            "applied_modules": applied_modules,
        }
    )
    return status


def clear_mlx_lora_stack(mlx_decoder: Any) -> None:
    """Remove active LoRA contributions while keeping already-wrapped modules."""
    for wrapper in _iter_wrappers(mlx_decoder):
        wrapper.clear_adapters()
    setattr(mlx_decoder, "_anvil_mlx_lora_loaded", False)
    setattr(mlx_decoder, "_anvil_mlx_lora_active", False)
    setattr(mlx_decoder, "_anvil_mlx_lora_adapters", [])


def set_mlx_lora_enabled(mlx_decoder: Any, enabled: bool) -> dict[str, Any]:
    """Enable or disable all MLX LoRA wrappers attached to *mlx_decoder*."""
    wrappers = list(_iter_wrappers(mlx_decoder))
    for wrapper in wrappers:
        wrapper.set_enabled(enabled)
    loaded = any(wrapper.adapter_count for wrapper in wrappers)
    setattr(mlx_decoder, "_anvil_mlx_lora_loaded", loaded)
    setattr(mlx_decoder, "_anvil_mlx_lora_active", bool(enabled) and loaded)
    return get_mlx_lora_status(mlx_decoder)


def get_mlx_lora_status(mlx_decoder: Any) -> dict[str, Any]:
    """Return compact status for MLX LoRA wrappers on *mlx_decoder*."""
    wrappers = list(_iter_wrappers(mlx_decoder))
    adapters: dict[str, float] = {}
    modules: list[str] = []
    active = False
    for module_path, wrapper in _wrapper_registry(mlx_decoder).items():
        if wrapper.adapter_count:
            modules.append(module_path)
        active = active or bool(getattr(wrapper, "_anvil_lora_enabled", False))
        for adapter in getattr(wrapper, "_anvil_lora_adapters", []):
            name = str(adapter["name"])
            adapters[name] = float(adapter["user_scale"])

    loaded = bool(adapters)
    return {
        "backend": "mlx",
        "loaded": loaded,
        "active": active and loaded,
        "adapters": list(adapters.keys()),
        "scales": adapters,
        "modules": modules,
        "wrapped_modules": len(wrappers),
    }


def _apply_adapter_to_decoder(mlx_decoder: Any, adapter: MLXLoRAAdapter) -> int:
    resolved = []
    missing: list[str] = []
    for module in adapter.modules:
        try:
            parent, child_key, target = _resolve_parent_and_child(
                mlx_decoder,
                module.module_path,
            )
        except (AttributeError, IndexError, TypeError):
            missing.append(module.module_path)
            continue
        resolved.append((module, parent, child_key, target))

    if missing:
        preview = ", ".join(missing[:5])
        more = f", +{len(missing) - 5} more" if len(missing) > 5 else ""
        raise RuntimeError(
            "PEFT LoRA targets are not present in the ACE-Step MLX decoder: "
            f"{preview}{more}"
        )

    registry = _wrapper_registry(mlx_decoder)
    for module, parent, child_key, target in resolved:
        wrapper = _ensure_wrapper(parent, child_key, target)
        wrapper.add_adapter(
            name=adapter.name,
            lora_a_np=module.lora_a,
            lora_b_np=module.lora_b,
            scale=module.scale,
            user_scale=module.user_scale,
        )
        registry[module.module_path] = wrapper

    setattr(mlx_decoder, "_anvil_mlx_lora_loaded", True)
    setattr(mlx_decoder, "_anvil_mlx_lora_adapters", list(_adapter_names(registry.values())))
    return len(resolved)


def _load_peft_weights(path: Path) -> Mapping[str, Any]:
    safetensors_path = path / "adapter_model.safetensors"
    if safetensors_path.is_file():
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to load PEFT LoRA safetensors for MLX."
            ) from exc
        return load_file(str(safetensors_path))

    bin_path = path / "adapter_model.bin"
    if bin_path.is_file():
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required to load PEFT adapter_model.bin files.") from exc
        payload = torch.load(str(bin_path), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"adapter_model.bin did not contain a state dict: {bin_path}")
        return payload

    raise FileNotFoundError(
        f"Missing PEFT adapter weights in {path} "
        f"(expected one of {', '.join(_PEFT_WEIGHT_FILES)})"
    )


def _module_alpha(config: Mapping[str, Any], module_path: str, *, default: float) -> float:
    alpha_pattern = config.get("alpha_pattern")
    if isinstance(alpha_pattern, Mapping):
        for key, value in alpha_pattern.items():
            if module_path.endswith(str(key)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    break
    try:
        return float(config.get("lora_alpha", default))
    except (TypeError, ValueError):
        return default


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(value)


def _resolve_parent_and_child(root: Any, module_path: str) -> tuple[Any, str, Any]:
    parts = [part for part in module_path.split(".") if part]
    if not parts:
        raise AttributeError("empty module path")
    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    child_key = parts[-1]
    return parent, child_key, _get_child(parent, child_key)


def _get_child(parent: Any, key: str) -> Any:
    if key.isdigit():
        return parent[int(key)]
    return getattr(parent, key)


def _set_child(parent: Any, key: str, value: Any) -> None:
    if key.isdigit():
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def _ensure_wrapper(parent: Any, child_key: str, target: Any) -> Any:
    if getattr(target, "_anvil_mlx_lora_wrapper", False):
        return target
    if not callable(target) or not hasattr(target, "weight"):
        raise RuntimeError(f"MLX LoRA target is not a Linear-like module: {child_key}")
    wrapper_cls = _wrapper_class()
    wrapper = wrapper_cls(target)
    _set_child(parent, child_key, wrapper)
    return wrapper


def _wrapper_class() -> type[Any]:
    import mlx.core as mx
    import mlx.nn as nn

    class AnvilMLXLoRALinear(nn.Module):  # type: ignore[misc]
        """Linear wrapper that adds one or more PEFT LoRA contributions."""

        _anvil_mlx_lora_wrapper = True

        def __init__(self, base_layer: Any):
            super().__init__()
            self.base_layer = base_layer
            self._anvil_lora_adapters: list[dict[str, Any]] = []
            self._anvil_lora_enabled = True

        @property
        def weight(self) -> Any:
            return self.base_layer.weight

        @property
        def bias(self) -> Any:
            return getattr(self.base_layer, "bias", None)

        @property
        def adapter_count(self) -> int:
            return len(self._anvil_lora_adapters)

        def clear_adapters(self) -> None:
            self._anvil_lora_adapters.clear()

        def set_enabled(self, enabled: bool) -> None:
            self._anvil_lora_enabled = bool(enabled)

        def add_adapter(
            self,
            *,
            name: str,
            lora_a_np: np.ndarray,
            lora_b_np: np.ndarray,
            scale: float,
            user_scale: float,
        ) -> None:
            base_shape = tuple(self.base_layer.weight.shape)
            if len(base_shape) != 2:
                raise RuntimeError(f"MLX LoRA target weight must be rank-2: {base_shape}")
            if lora_a_np.shape[1] != base_shape[1] or lora_b_np.shape[0] != base_shape[0]:
                raise RuntimeError(
                    "LoRA tensor shape does not match MLX Linear weight "
                    f"(weight={base_shape}, A={lora_a_np.shape}, B={lora_b_np.shape})"
                )
            if lora_b_np.shape[1] != lora_a_np.shape[0]:
                raise RuntimeError(
                    f"LoRA A/B rank mismatch: A={lora_a_np.shape}, B={lora_b_np.shape}"
                )

            dtype = getattr(self.base_layer.weight, "dtype", None)
            lora_a = mx.array(lora_a_np, dtype=dtype) if dtype is not None else mx.array(lora_a_np)
            lora_b = mx.array(lora_b_np, dtype=dtype) if dtype is not None else mx.array(lora_b_np)
            mx.eval(lora_a, lora_b)
            self._anvil_lora_adapters.append(
                {
                    "name": name,
                    "a": lora_a,
                    "b": lora_b,
                    "scale": float(scale),
                    "user_scale": float(user_scale),
                }
            )

        def __call__(self, x: Any) -> Any:
            output = self.base_layer(x)
            if not self._anvil_lora_enabled:
                return output
            for adapter in self._anvil_lora_adapters:
                output = output + ((x @ adapter["a"].T) @ adapter["b"].T) * adapter["scale"]
            return output

    return AnvilMLXLoRALinear


def _wrapper_registry(mlx_decoder: Any) -> dict[str, Any]:
    registry = getattr(mlx_decoder, "_anvil_mlx_lora_wrappers", None)
    if not isinstance(registry, dict):
        registry = {}
        setattr(mlx_decoder, "_anvil_mlx_lora_wrappers", registry)
    return registry


def _iter_wrappers(mlx_decoder: Any) -> Iterable[Any]:
    return _wrapper_registry(mlx_decoder).values()


def _adapter_names(wrappers: Iterable[Any]) -> Iterable[str]:
    seen: set[str] = set()
    for wrapper in wrappers:
        for adapter in getattr(wrapper, "_anvil_lora_adapters", []):
            name = str(adapter["name"])
            if name not in seen:
                seen.add(name)
                yield name


def _entry_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)
