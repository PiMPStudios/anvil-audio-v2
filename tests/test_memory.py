import torch

from anvil_audio.utils import memory


def test_estimate_values_size_mb_counts_tensors():
    tensor = torch.zeros(2, 4, dtype=torch.float32)

    size_mb = memory.estimate_values_size_mb(tensor)

    assert size_mb == 32 / (1024**2)


def test_memory_pressure_status_reports_rss_threshold(monkeypatch):
    monkeypatch.setattr(memory, "process_rss_mb", lambda: 200.0)
    monkeypatch.setattr(memory, "system_memory_status", lambda: {})
    monkeypatch.setattr(memory, "torch_memory_status", lambda: {})
    monkeypatch.setattr(memory, "mlx_memory_status", lambda: {})

    status = memory.memory_pressure_status(rss_limit_mb=100.0)

    assert status["pressure"] is True
    assert status["reasons"] == ["process_rss_mb 200.0 >= 100.0"]
    assert status["memory"]["process_rss_mb"] == 200.0


def test_cleanup_if_memory_pressure_flushes_when_threshold_trips(monkeypatch):
    calls = []
    pressure = {"pressure": True, "reasons": ["test"], "thresholds": {}, "memory": {}}

    monkeypatch.setattr(memory, "memory_pressure_status", lambda **_kwargs: pressure)
    monkeypatch.setattr(
        memory,
        "flush_memory_caches",
        lambda: calls.append("flush") or {"actions": ["flush"]},
    )

    result = memory.cleanup_if_memory_pressure(reason="test")

    assert result["triggered"] is True
    assert result["cleanup"] == {"actions": ["flush"]}
    assert calls == ["flush"]
