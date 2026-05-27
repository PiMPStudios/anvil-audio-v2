"""Shared memory cleanup and pressure-monitoring helpers."""

from __future__ import annotations

import gc
import os
import subprocess
from typing import Any


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _device_type(device: object | None) -> str | None:
    if device is None:
        return None
    return str(device).split(":", 1)[0].lower()


def flush_memory_caches(
    device: object | None = None,
    *,
    include_gc: bool = True,
    include_torch: bool = True,
    include_mlx: bool = True,
    synchronize: bool = True,
) -> dict[str, Any]:
    """Best-effort Python, torch, and MLX cache cleanup.

    Args:
        device: Optional device hint. When omitted, all known accelerator cache
            backends are flushed. When provided, only caches relevant to that
            device family are flushed.
        include_gc: Run Python garbage collection before accelerator cleanup.
        include_torch: Clear CUDA/MPS caches when torch is importable.
        include_mlx: Clear MLX/Metal caches when MLX is importable.
        synchronize: Synchronize accelerators when the backend exposes a cheap
            synchronization hook.

    Returns:
        A compact dict describing the cleanup that was attempted.
    """
    actions: list[str] = []
    collected = gc.collect() if include_gc else None
    dtype = _device_type(device)

    if include_torch and dtype in {None, "cuda", "mps"}:
        try:
            import torch

            if dtype in {None, "cuda"} and torch.cuda.is_available():
                torch.cuda.empty_cache()
                actions.append("torch.cuda.empty_cache")
                if synchronize:
                    try:
                        torch.cuda.synchronize()
                        actions.append("torch.cuda.synchronize")
                    except RuntimeError:
                        pass

            mps = getattr(torch, "mps", None)
            mps_empty_cache = getattr(mps, "empty_cache", None)
            if dtype in {None, "mps"} and callable(mps_empty_cache):
                mps_empty_cache()
                actions.append("torch.mps.empty_cache")
                mps_synchronize = getattr(mps, "synchronize", None)
                if synchronize and callable(mps_synchronize):
                    mps_synchronize()
                    actions.append("torch.mps.synchronize")
        except Exception as exc:
            actions.append(f"torch cleanup skipped: {exc}")

    if include_mlx and dtype in {None, "mps", "mlx", "metal"}:
        try:
            import mlx.core as mx

            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
                actions.append("mlx.clear_cache")

            metal = getattr(mx, "metal", None)
            metal_clear_cache = getattr(metal, "clear_cache", None)
            if callable(metal_clear_cache):
                metal_clear_cache()
                actions.append("mlx.metal.clear_cache")
        except Exception as exc:
            actions.append(f"mlx cleanup skipped: {exc}")

    return {"gc_collected": collected, "actions": actions}


def process_rss_mb() -> float | None:
    """Return current process RSS in MB when it can be measured cheaply."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    try:
        raw = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        return int(raw) / 1024 if raw else None
    except Exception:
        return None


def system_memory_status() -> dict[str, Any]:
    """Return system memory details when psutil is installed."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_percent": vm.percent,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def torch_memory_status() -> dict[str, Any]:
    """Return accelerator memory stats exposed by torch."""
    status: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            cuda_status = {
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
                "peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / (1024**2), 2
                ),
            }
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                cuda_status["free_mb"] = round(free_bytes / (1024**2), 2)
                cuda_status["total_mb"] = round(total_bytes / (1024**2), 2)
            except Exception:
                pass
            status["cuda"] = cuda_status

        mps_stats: dict[str, Any] = {}
        mps = getattr(torch, "mps", None)
        for name in (
            "current_allocated_memory",
            "driver_allocated_memory",
            "recommended_max_memory",
        ):
            fn = getattr(mps, name, None)
            if callable(fn):
                try:
                    mps_stats[f"{name}_mb"] = round(fn() / (1024**2), 2)
                except Exception:
                    pass
        if mps_stats:
            status["mps"] = mps_stats
    except Exception as exc:
        status["error"] = str(exc)
    return status


def mlx_memory_status() -> dict[str, Any]:
    """Return MLX memory stats when MLX is importable."""
    try:
        import mlx.core as mx
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    status: dict[str, Any] = {"available": True}
    for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        fn = getattr(mx, name, None)
        if callable(fn):
            try:
                status[f"{name.removeprefix('get_')}_mb"] = round(
                    fn() / (1024**2), 2
                )
            except Exception:
                pass
    return status


def memory_snapshot() -> dict[str, Any]:
    """Return process, system, torch, and MLX memory status."""
    return {
        "process_rss_mb": process_rss_mb(),
        "system_memory": system_memory_status(),
        "torch_memory": torch_memory_status(),
        "mlx_memory": mlx_memory_status(),
    }


def memory_pressure_status(
    *,
    env_prefix: str = "ANVIL_MEMORY",
    rss_limit_mb: float | None = None,
    system_available_limit_mb: float | None = None,
    system_used_percent_limit: float | None = None,
    mps_limit_ratio: float | None = None,
    cuda_reserved_ratio: float | None = None,
) -> dict[str, Any]:
    """Evaluate configured memory-pressure thresholds.

    Thresholds default to env vars named from ``env_prefix``:
    ``<prefix>_RSS_LIMIT_MB``, ``<prefix>_SYSTEM_AVAILABLE_LIMIT_MB``,
    ``<prefix>_SYSTEM_USED_PERCENT_LIMIT``, ``<prefix>_MPS_LIMIT_RATIO``, and
    ``<prefix>_CUDA_RESERVED_RATIO``. A threshold of 0 disables that check.
    """
    thresholds = {
        "rss_limit_mb": (
            _env_float(f"{env_prefix}_RSS_LIMIT_MB")
            if rss_limit_mb is None
            else max(0.0, float(rss_limit_mb))
        ),
        "system_available_limit_mb": (
            _env_float(f"{env_prefix}_SYSTEM_AVAILABLE_LIMIT_MB")
            if system_available_limit_mb is None
            else max(0.0, float(system_available_limit_mb))
        ),
        "system_used_percent_limit": (
            _env_float(f"{env_prefix}_SYSTEM_USED_PERCENT_LIMIT")
            if system_used_percent_limit is None
            else max(0.0, float(system_used_percent_limit))
        ),
        "mps_limit_ratio": (
            _env_float(f"{env_prefix}_MPS_LIMIT_RATIO")
            if mps_limit_ratio is None
            else max(0.0, float(mps_limit_ratio))
        ),
        "cuda_reserved_ratio": (
            _env_float(f"{env_prefix}_CUDA_RESERVED_RATIO")
            if cuda_reserved_ratio is None
            else max(0.0, float(cuda_reserved_ratio))
        ),
    }
    snapshot = memory_snapshot()
    reasons: list[str] = []

    rss_mb = snapshot.get("process_rss_mb")
    if (
        thresholds["rss_limit_mb"] > 0
        and rss_mb is not None
        and rss_mb >= thresholds["rss_limit_mb"]
    ):
        reasons.append(
            f"process_rss_mb {rss_mb:.1f} >= {thresholds['rss_limit_mb']:.1f}"
        )

    system = snapshot.get("system_memory", {})
    available_gb = system.get("available_gb")
    if thresholds["system_available_limit_mb"] > 0 and available_gb is not None:
        available_mb = float(available_gb) * 1024
        if available_mb <= thresholds["system_available_limit_mb"]:
            reasons.append(
                "system_available_mb "
                f"{available_mb:.1f} <= {thresholds['system_available_limit_mb']:.1f}"
            )

    used_percent = system.get("used_percent")
    if (
        thresholds["system_used_percent_limit"] > 0
        and used_percent is not None
        and float(used_percent) >= thresholds["system_used_percent_limit"]
    ):
        reasons.append(
            "system_used_percent "
            f"{float(used_percent):.1f} >= {thresholds['system_used_percent_limit']:.1f}"
        )

    torch_status = snapshot.get("torch_memory", {})
    mps = torch_status.get("mps", {})
    mps_recommended = mps.get("recommended_max_memory_mb")
    mps_driver = mps.get("driver_allocated_memory_mb")
    if (
        thresholds["mps_limit_ratio"] > 0
        and mps_recommended
        and mps_driver is not None
        and float(mps_driver) / float(mps_recommended) >= thresholds["mps_limit_ratio"]
    ):
        reasons.append(
            "mps_driver_ratio "
            f"{float(mps_driver) / float(mps_recommended):.2f} "
            f">= {thresholds['mps_limit_ratio']:.2f}"
        )

    cuda = torch_status.get("cuda", {})
    cuda_reserved = cuda.get("reserved_mb")
    cuda_total = cuda.get("total_mb")
    if (
        thresholds["cuda_reserved_ratio"] > 0
        and cuda_total
        and cuda_reserved is not None
        and float(cuda_reserved) / float(cuda_total) >= thresholds["cuda_reserved_ratio"]
    ):
        reasons.append(
            "cuda_reserved_ratio "
            f"{float(cuda_reserved) / float(cuda_total):.2f} "
            f">= {thresholds['cuda_reserved_ratio']:.2f}"
        )

    return {
        "pressure": bool(reasons),
        "reasons": reasons,
        "thresholds": thresholds,
        "memory": snapshot,
    }


def cleanup_if_memory_pressure(
    *,
    reason: str,
    env_prefix: str = "ANVIL_MEMORY",
) -> dict[str, Any]:
    """Flush caches when configured memory-pressure thresholds are exceeded."""
    pressure = memory_pressure_status(env_prefix=env_prefix)
    if not pressure["pressure"]:
        return {"triggered": False, "reason": reason, "pressure": pressure}

    return {
        "triggered": True,
        "reason": reason,
        "pressure": pressure,
        "cleanup": flush_memory_caches(),
    }


def estimate_value_size_mb(value: Any) -> float:
    """Estimate tensor/array memory footprint in MB for cleanup thresholds."""
    if value is None:
        return 0.0
    if isinstance(value, dict):
        return sum(estimate_value_size_mb(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(estimate_value_size_mb(item) for item in value)

    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return float(numel() * element_size()) / (1024**2)
        except Exception:
            return 0.0

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return float(nbytes) / (1024**2)
        except Exception:
            return 0.0

    return 0.0


def estimate_values_size_mb(*values: Any) -> float:
    """Estimate total tensor/array memory footprint in MB."""
    return sum(estimate_value_size_mb(value) for value in values)
