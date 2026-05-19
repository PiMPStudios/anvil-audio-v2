"""Tests for ACEStepPipeline without project_root."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_fake_acestep_handler(monkeypatch, calls):
    class FakeAceStepHandler:
        def initialize_service(self, **kwargs):
            calls["initialize_service"] = kwargs
            return "ok", True

    acestep_pkg = types.ModuleType("acestep")
    handler_mod = types.ModuleType("acestep.handler")
    handler_mod.AceStepHandler = FakeAceStepHandler

    monkeypatch.setitem(sys.modules, "acestep", acestep_pkg)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)


def test_import_error_without_project_root_gives_clear_message(monkeypatch):
    """When acestep isn't installed and project_root=None, error is actionable."""
    for key in list(sys.modules.keys()):
        if key.startswith("acestep"):
            monkeypatch.delitem(sys.modules, key)

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "acestep.handler":
            raise ImportError("No module named 'acestep'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    with pytest.raises(ImportError, match="pip install anvil-audio\\[acestep\\]"):
        ACEStepPipeline(project_root=None)


def test_project_root_none_skips_sys_path_injection(monkeypatch):
    """project_root=None does not mutate sys.path."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "acestep.handler":
            raise ImportError("No module named 'acestep'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    path_before = set(sys.path)
    try:
        ACEStepPipeline(project_root=None)
    except ImportError:
        pass

    new_entries = [p for p in sys.path if p not in path_before]
    assert not new_entries, f"sys.path was mutated with project_root=None: {new_entries}"


def test_missing_xl_checkpoint_blocks_auto_download(monkeypatch, tmp_path):
    """XL variants are optional and should not auto-download on model load."""
    calls = {}
    _install_fake_acestep_handler(monkeypatch, calls)
    monkeypatch.delenv("ANVIL_ACESTEP_ALLOW_XL_AUTO_DOWNLOAD", raising=False)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    with pytest.raises(RuntimeError, match="large optional downloads"):
        ACEStepPipeline(
            project_root=str(tmp_path),
            config_path="acestep-v15-xl-sft",
            device="cpu",
        )

    assert "initialize_service" not in calls


def test_installed_xl_checkpoint_allows_load_and_forwards_init_options(
    monkeypatch,
    tmp_path,
):
    """Installed XL checkpoints can load, including registry-provided memory knobs."""
    calls = {}
    _install_fake_acestep_handler(monkeypatch, calls)
    monkeypatch.delenv("ANVIL_ACESTEP_ALLOW_XL_AUTO_DOWNLOAD", raising=False)

    checkpoint_dir = tmp_path / "checkpoints" / "acestep-v15-xl-turbo"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"placeholder")

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    ACEStepPipeline(
        project_root=str(tmp_path),
        config_path="acestep-v15-xl-turbo",
        device="cpu",
        default_params={
            "offload_to_cpu": True,
            "offload_dit_to_cpu": True,
            "quantization": "int8_weight_only",
            "prefer_source": "huggingface",
            "vae_checkpoint": "scragvae",
        },
    )

    init_kwargs = calls["initialize_service"]
    assert init_kwargs["offload_to_cpu"] is True
    assert init_kwargs["offload_dit_to_cpu"] is True
    assert init_kwargs["quantization"] == "int8_weight_only"
    assert init_kwargs["prefer_source"] == "huggingface"
    assert init_kwargs["vae_checkpoint"] == "scragvae"


def test_blank_lyrics_use_direct_sft_conditioning_defaults(monkeypatch, tmp_path):
    """Blank-lyrics SFT requests should match AnvilApp's direct DiT path."""
    calls = {}

    class FakeAceStepHandler:
        def initialize_service(self, **kwargs):
            calls["initialize_service"] = kwargs
            return "ok", True

    class FakeLLMHandler:
        llm_initialized = True

        def initialize(self, **kwargs):
            calls["initialize_lm"] = kwargs
            return "lm ok", True

    class FakeGenerationParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_generate_music(dit_handler, llm_handler, params, config, save_dir=None, progress=None):
        calls["generate_music"] = {
            "llm_handler": llm_handler,
            "params": params,
            "config": config,
            "save_dir": save_dir,
        }
        return SimpleNamespace(
            success=True,
            error=None,
            status_message="ok",
            audios=[{"tensor": torch.zeros(2, 16), "sample_rate": 48000}],
        )

    acestep_pkg = types.ModuleType("acestep")
    handler_mod = types.ModuleType("acestep.handler")
    handler_mod.AceStepHandler = FakeAceStepHandler
    lm_mod = types.ModuleType("acestep.llm_inference")
    lm_mod.LLMHandler = FakeLLMHandler
    inference_mod = types.ModuleType("acestep.inference")
    inference_mod.GenerationParams = FakeGenerationParams
    inference_mod.GenerationConfig = FakeGenerationConfig
    inference_mod.generate_music = fake_generate_music

    monkeypatch.setitem(sys.modules, "acestep", acestep_pkg)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", lm_mod)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference_mod)
    monkeypatch.setenv("ANVIL_ACESTEP_USE_MLX_DIT", "0")

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    pipe = ACEStepPipeline(
        project_root=str(tmp_path),
        config_path="acestep-v15-sft",
        device="cpu",
        lm_model_path="acestep-5Hz-lm-4B",
        default_params={
            "steps": 50,
            "cfg_scale": 7.5,
            "shift": 3.0,
            "lm_cfg_scale": 2.0,
            "thinking": False,
            "use_cot_metas": False,
            "use_cot_caption": False,
            "use_cot_language": False,
            "dcw_enabled": False,
            "velocity_norm_threshold": 0.0,
            "velocity_ema_factor": 0.0,
        },
    )
    audio = pipe.generate(
        [
            {
                "prompt": "instrumental rock",
                "lyrics": "",
                "negative_prompt": "muddy mix, clipped vocals",
                "seconds_total": 10,
            }
        ],
        seed=123,
    )

    assert audio.shape == (1, 2, 16)
    assert calls["initialize_service"]["use_mlx_dit"] is False
    assert "initialize_lm" not in calls
    assert calls["generate_music"]["llm_handler"] is None
    params = calls["generate_music"]["params"]
    assert params.lyrics == ""
    assert params.vocal_language == "en"
    assert params.thinking is False
    assert params.use_cot_metas is False
    assert params.use_cot_caption is False
    assert params.use_cot_language is False
    assert params.dcw_enabled is False
    assert params.velocity_norm_threshold == 0.0
    assert params.velocity_ema_factor == 0.0
    assert params.lm_cfg_scale == 2.0
    assert params.lm_negative_prompt == "muddy mix, clipped vocals"
    assert params.inference_steps == 50
    assert params.guidance_scale == 7.5
    assert params.shift == 3.0
    config = calls["generate_music"]["config"]
    assert config.use_random_seed is False
    assert config.seeds == [123]


def test_lm_planner_loads_lazily_when_thinking_requested(monkeypatch, tmp_path):
    """ACE-Step LM should load only when the generation path needs it."""
    calls = {}

    class FakeAceStepHandler:
        def initialize_service(self, **kwargs):
            calls["initialize_service"] = kwargs
            return "ok", True

    class FakeLLMHandler:
        def initialize(self, **kwargs):
            calls["initialize_lm"] = kwargs
            return "lm ok", True

    class FakeGenerationParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_generate_music(dit_handler, llm_handler, params, config, save_dir=None, progress=None):
        calls["generate_music"] = {
            "llm_handler": llm_handler,
            "params": params,
            "config": config,
            "save_dir": save_dir,
        }
        return SimpleNamespace(
            success=True,
            error=None,
            status_message="ok",
            audios=[{"tensor": torch.zeros(2, 16), "sample_rate": 48000}],
        )

    acestep_pkg = types.ModuleType("acestep")
    handler_mod = types.ModuleType("acestep.handler")
    handler_mod.AceStepHandler = FakeAceStepHandler
    lm_mod = types.ModuleType("acestep.llm_inference")
    lm_mod.LLMHandler = FakeLLMHandler
    inference_mod = types.ModuleType("acestep.inference")
    inference_mod.GenerationParams = FakeGenerationParams
    inference_mod.GenerationConfig = FakeGenerationConfig
    inference_mod.generate_music = fake_generate_music

    monkeypatch.setitem(sys.modules, "acestep", acestep_pkg)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", lm_mod)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference_mod)
    monkeypatch.setenv("ANVIL_ACESTEP_USE_MLX_DIT", "0")

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    pipe = ACEStepPipeline(
        project_root=str(tmp_path),
        config_path="acestep-v15-turbo",
        device="cpu",
        lm_model_path="acestep-5Hz-lm-1.7B",
        default_params={
            "steps": 8,
            "cfg_scale": 1.0,
            "thinking": True,
            "use_cot_metas": True,
            "use_cot_caption": False,
            "use_cot_language": True,
            "dcw_enabled": False,
        },
    )

    assert "initialize_lm" not in calls

    audio = pipe.generate(
        [{"prompt": "anthemic rock", "lyrics": "[Instrumental]", "seconds_total": 10}],
        seed=123,
    )

    assert audio.shape == (1, 2, 16)
    assert calls["initialize_lm"]["lm_model_path"].endswith("acestep-5Hz-lm-1.7B")
    assert calls["generate_music"]["llm_handler"] is not None
    assert calls["generate_music"]["params"].thinking is True


def test_apply_lora_adapter_delegates_to_acestep_handler(monkeypatch, tmp_path):
    """ACEStepPipeline should use upstream handler APIs for LoRA loading."""
    calls = {"add_lora": []}

    class FakeAceStepHandler:
        def initialize_service(self, **kwargs):
            calls["initialize_service"] = kwargs
            return "ok", True

        def add_lora(self, path, adapter_name=None):
            calls["add_lora"].append((path, adapter_name))
            return "✅ loaded"

        def set_active_lora_adapter(self, adapter_name):
            calls["active"] = adapter_name
            return "✅ active"

        def set_lora_scale(self, adapter_name, scale):
            calls["scale"] = (adapter_name, scale)
            return "✅ scale"

        def set_use_lora(self, enabled):
            calls["enabled"] = enabled
            return "✅ enabled"

        def get_lora_status(self):
            return {"loaded": True, "active": True}

    acestep_pkg = types.ModuleType("acestep")
    handler_mod = types.ModuleType("acestep.handler")
    handler_mod.AceStepHandler = FakeAceStepHandler
    monkeypatch.setitem(sys.modules, "acestep", acestep_pkg)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    pipe = ACEStepPipeline(project_root=str(tmp_path), device="cpu", use_mlx_dit=False)

    status = pipe.apply_lora_adapter(
        str(adapter_dir),
        adapter_name="style",
        scale=0.6,
    )
    second = pipe.apply_lora_adapter(
        str(adapter_dir),
        adapter_name="style",
        scale=0.4,
    )

    assert calls["add_lora"] == [(str(adapter_dir.resolve()), "style")]
    assert calls["active"] == "style"
    assert calls["scale"] == ("style", 0.4)
    assert calls["enabled"] is True
    assert status["adapter_name"] == "style"
    assert second["message"] == "Adapter already loaded: style"


def test_apply_lora_adapter_rejects_native_mlx_dit(monkeypatch, tmp_path):
    """Native MLX DiT currently bypasses ACE-Step's PyTorch LoRA injection."""

    class FakeAceStepHandler:
        def initialize_service(self, **kwargs):
            return "ok", True

    acestep_pkg = types.ModuleType("acestep")
    handler_mod = types.ModuleType("acestep.handler")
    handler_mod.AceStepHandler = FakeAceStepHandler
    monkeypatch.setitem(sys.modules, "acestep", acestep_pkg)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    pipe = ACEStepPipeline(project_root=str(tmp_path), device="cpu", use_mlx_dit=True)

    with pytest.raises(RuntimeError, match="PyTorch DiT backend"):
        pipe.apply_lora_adapter(str(adapter_dir))
