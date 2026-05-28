import json
import stat

import pytest

from anvil_audio.cloud import CloudJobPackageConfig, SSHRunConfig
from anvil_audio.cloud.gpufindr import (
    GPUFindrSearch,
    GPUOffer,
    build_gpufindr_url,
    filter_gpufindr_offers,
)
from anvil_audio.cloud.job import create_cloud_job_package
from anvil_audio.cloud.runpod import (
    RunPodLaunchConfig,
    build_launch_request,
    build_status_request,
    launch_pod,
    ssh_target_from_pod,
    terminate_pod,
)
from anvil_audio.cloud.ssh import plan_ssh_run


def test_create_cloud_job_package_copies_primary_assets(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    output_dir = tmp_path / "jobs" / "dark_blues_job"

    result = create_cloud_job_package(
        CloudJobPackageConfig(
            training_bundle=bundle,
            output_dir=output_dir,
            primary_asset="instrumental",
            repo_url="https://example.test/anvil.git",
            repo_ref="feature-cloud",
        )
    )

    assert result.job_dir == output_dir.resolve()
    assert result.clip_count == 1
    assert result.asset_count == 1
    assert (output_dir / "inputs/dataset/stems/clip_0001/instrumental.wav").is_file()

    captions = json.loads(
        (output_dir / "inputs/dataset/captions.json").read_text(encoding="utf-8")
    )
    assert captions[0]["file"] == "stems/clip_0001/instrumental.wav"
    assert captions[0]["source_file"] == "clips/clip_0001.wav"

    job = json.loads((output_dir / "job.json").read_text(encoding="utf-8"))
    assert job["primary_asset"] == "instrumental"
    assert job["runtime"]["repo_ref"] == "feature-cloud"
    assert job["training"]["recipe"] == "lora-balanced"
    assert job["training"]["lyrics"] == "[Instrumental]"
    assert job["training"]["lyrics_source"] == "constant"
    assert job["training"]["checkpoint_models"] == ["main", "acestep-v15-sft"]

    bootstrap = output_dir / "scripts/bootstrap.sh"
    assert bootstrap.stat().st_mode & stat.S_IXUSR
    bootstrap_text = bootstrap.read_text(encoding="utf-8")
    assert "third_parts/nano-vllm" in bootstrap_text
    assert "venv --system-site-packages .venv" in bootstrap_text
    assert "download_main_model" in bootstrap_text
    assert "download_submodel" in bootstrap_text
    assert (
        'REQUIRED_CHECKPOINT_MODELS=\'["main", "acestep-v15-sft"]\'' in bootstrap_text
    )
    assert 'pip install -c "$CONSTRAINTS_FILE" "$ANVIL_AUDIO_INSTALL"' in bootstrap_text
    assert "scikit-learn<1.8" in bootstrap_text
    assert "--force-reinstall" in bootstrap_text
    assert "--no-cache-dir" in bootstrap_text
    assert (
        'pip install "$ACESTEP_INSTALL" -c "$CONSTRAINTS_FILE" --no-deps'
        in bootstrap_text
    )

    training_text = (output_dir / "scripts/run_training.sh").read_text(encoding="utf-8")
    assert 'rm -rf "$TENSOR_DIR"' in training_text
    assert "--lyrics '[Instrumental]'" in training_text
    assert "--lyrics-source constant" in training_text

    collect_text = (output_dir / "scripts/collect.sh").read_text(encoding="utf-8")
    assert "anvil_cloud_collect" in collect_text
    assert "outputs/lora/final" in collect_text
    assert "outputs/lora/checkpoints.txt" in collect_text
    assert 'tar -czf "$ARCHIVE" -C "$COLLECT_DIR" .' in collect_text
    assert "tar -czf outputs/anvil_cloud_results.tar.gz job.json" not in collect_text


def test_create_cloud_job_package_accepts_custom_training_lyrics(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    output_dir = tmp_path / "jobs" / "voice_job"

    create_cloud_job_package(
        CloudJobPackageConfig(
            training_bundle=bundle,
            output_dir=output_dir,
            primary_asset="instrumental",
            training_lyrics="vocal stem, expressive male vocal",
        )
    )

    job = json.loads((output_dir / "job.json").read_text(encoding="utf-8"))
    assert job["training"]["lyrics"] == "vocal stem, expressive male vocal"

    training_text = (output_dir / "scripts/run_training.sh").read_text(encoding="utf-8")
    assert "--lyrics 'vocal stem, expressive male vocal'" in training_text


def test_create_cloud_job_package_preserves_transcripts_for_lyrics_source(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["clips"][0]["transcript"] = "I keep the light low"
    payload["clips"][0]["transcription"] = {
        "backend": "fake-whisper",
        "text": "I keep the light low",
    }
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "jobs" / "voice_job"

    create_cloud_job_package(
        CloudJobPackageConfig(
            training_bundle=bundle,
            output_dir=output_dir,
            primary_asset="instrumental",
            training_lyrics="fallback vocal marker",
            training_lyrics_source="transcript",
        )
    )

    captions = json.loads(
        (output_dir / "inputs/dataset/captions.json").read_text(encoding="utf-8")
    )
    assert captions[0]["transcript"] == "I keep the light low"
    assert captions[0]["transcription"]["backend"] == "fake-whisper"

    job = json.loads((output_dir / "job.json").read_text(encoding="utf-8"))
    assert job["training"]["lyrics"] == "fallback vocal marker"
    assert job["training"]["lyrics_source"] == "transcript"

    training_text = (output_dir / "scripts/run_training.sh").read_text(encoding="utf-8")
    assert "--lyrics-source transcript" in training_text


def test_create_cloud_job_package_fails_without_primary_assets(tmp_path):
    bundle = _write_training_bundle(tmp_path)

    with pytest.raises(RuntimeError, match="No clips with primary asset vocals"):
        create_cloud_job_package(
            CloudJobPackageConfig(
                training_bundle=bundle,
                output_dir=tmp_path / "job",
                primary_asset="vocals",
            )
        )


def test_plan_ssh_run_builds_dry_run_commands(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    result = create_cloud_job_package(
        CloudJobPackageConfig(training_bundle=bundle, output_dir=tmp_path / "job")
    )

    commands = plan_ssh_run(
        SSHRunConfig(
            job_dir=result.job_dir,
            host="ubuntu@203.0.113.10",
            port=2222,
            dry_run=True,
            collect=True,
        )
    )

    assert commands[0][:4] == ["ssh", "-p", "2222", "ubuntu@203.0.113.10"]
    assert commands[1][0] == "rsync"
    assert commands[1].count("--exclude") == 4
    assert ".venv/" in commands[1]
    assert "outputs/" in commands[1]
    assert "scripts/bootstrap.sh" in commands[2][-1]
    assert "scripts/run_training.sh" in commands[3][-1]
    assert "scripts/collect.sh" in commands[4][-1]
    assert commands[5][:2] == ["mkdir", "-p"]
    assert commands[-1][0] == "rsync"
    assert "outputs/anvil_cloud_collect/" in commands[-1][-2]


def test_gpufindr_url_includes_remote_filters():
    url = build_gpufindr_url(
        GPUFindrSearch(
            gpu="h200",
            source="lambda",
            location="us",
            max_price=4.0,
            sort="total_cost_ph.asc",
            limit=5,
        )
    )

    assert url.startswith("https://gpufindr.com/gpus?")
    assert "source=lambda" in url
    assert "location=us" in url
    assert "max_price=4.0" in url
    assert "sort=total_cost_ph.asc" in url
    assert "offset=" not in url

    page_two = build_gpufindr_url(GPUFindrSearch(), offset=1000)
    assert "offset=1000" in page_two


def test_gpufindr_filter_keeps_matching_viable_offers():
    offers = [
        GPUOffer(
            id="1",
            source="lambda",
            location="us-east",
            name="H200",
            num_gpus=1,
            vram_gb=96,
            total_cost_ph=1.49,
            reliability=0.99,
            flops_per_dollar_ph=44.0,
            gpu_mem_bw_gbps=4800,
            url="https://example.test/h200",
        ),
        GPUOffer(
            id="2",
            source="vast",
            location="us",
            name="RTX 4090",
            num_gpus=1,
            vram_gb=24,
            total_cost_ph=0.45,
            reliability=0.98,
            flops_per_dollar_ph=100.0,
            gpu_mem_bw_gbps=1000,
            url="https://example.test/4090",
        ),
    ]

    result = filter_gpufindr_offers(
        offers,
        GPUFindrSearch(gpu="h200", min_vram_gb=80, min_gpus=1),
    )

    assert [offer.id for offer in result] == ["1"]


def test_runpod_launch_request_uses_job_and_ssh_defaults(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    job = create_cloud_job_package(
        CloudJobPackageConfig(training_bundle=bundle, output_dir=tmp_path / "job")
    )

    request = build_launch_request(
        RunPodLaunchConfig(
            job_dir=job.job_dir,
            gpu_type="NVIDIA H200",
            name="anvil-test",
            allowed_cuda_versions=("12.1", "12.2"),
        )
    )

    variables = request.variables["input"]
    assert variables["gpuTypeId"] == "NVIDIA H200"
    assert variables["name"] == "anvil-test"
    assert variables["ports"] == "22/tcp"
    assert variables["startSsh"] is True
    assert variables["templateId"] == "runpod-torch-v280"
    assert "dockerArgs" not in variables
    assert "imageName" not in variables
    assert variables["allowedCudaVersions"] == ["12.1", "12.2"]
    assert {"key": "ANVIL_CLOUD_JOB_NAME", "value": "job"} in variables["env"]


def test_runpod_launch_request_can_use_raw_image(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    job = create_cloud_job_package(
        CloudJobPackageConfig(training_bundle=bundle, output_dir=tmp_path / "job")
    )

    request = build_launch_request(
        RunPodLaunchConfig(
            job_dir=job.job_dir,
            gpu_type="H200",
            template_id=None,
            image_name="runpod/pytorch:test",
        )
    )

    variables = request.variables["input"]
    assert variables["imageName"] == "runpod/pytorch:test"
    assert variables["dockerArgs"] == "sleep infinity"
    assert "templateId" not in variables


def test_runpod_launch_request_minimal_omits_extra_constraints(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    job = create_cloud_job_package(
        CloudJobPackageConfig(training_bundle=bundle, output_dir=tmp_path / "job")
    )

    request = build_launch_request(
        RunPodLaunchConfig(
            job_dir=job.job_dir,
            gpu_type="H200",
            minimal=True,
        )
    )

    variables = request.variables["input"]
    assert variables["gpuTypeId"] == "H200"
    assert variables["templateId"] == "runpod-torch-v280"
    assert variables["startSsh"] is True
    assert "containerDiskInGb" not in variables
    assert "volumeInGb" not in variables
    assert "minVcpuCount" not in variables
    assert "minMemoryInGb" not in variables


def test_runpod_launch_dry_run_does_not_require_api_key(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    job = create_cloud_job_package(
        CloudJobPackageConfig(training_bundle=bundle, output_dir=tmp_path / "job")
    )

    result = launch_pod(
        RunPodLaunchConfig(
            job_dir=job.job_dir,
            gpu_type="NVIDIA H200",
            dry_run=True,
        )
    )

    assert result["dry_run"] is True
    assert "podFindAndDeployOnDemand" in result["request"]["query"]


def test_runpod_terminate_dry_run_uses_pod_id():
    result = terminate_pod("pod123", dry_run=True)

    assert result["dry_run"] is True
    assert result["request"]["variables"] == {"input": {"podId": "pod123"}}


def test_runpod_status_query_uses_pod_id_filter():
    request = build_status_request("pod123")

    assert request.variables == {"input": {"podId": "pod123"}}


def test_runpod_ssh_target_from_runtime_ports():
    target = ssh_target_from_pod(
        {
            "runtime": {
                "ports": [
                    {
                        "ip": "203.0.113.10",
                        "privatePort": 22,
                        "publicPort": 32061,
                        "type": "tcp",
                    }
                ]
            }
        }
    )

    assert target == "root@203.0.113.10 -p 32061"


def test_runpod_ssh_target_handles_pending_runtime():
    assert ssh_target_from_pod({"runtime": None}) is None


def _write_training_bundle(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "clips").mkdir(parents=True)
    (dataset / "stems/clip_0001").mkdir(parents=True)
    (dataset / "clips/clip_0001.wav").write_bytes(b"mix")
    (dataset / "stems/clip_0001/instrumental.wav").write_bytes(b"inst")
    (dataset / "captions.json").write_text(
        json.dumps(
            [
                {
                    "file": "clips/clip_0001.wav",
                    "caption": "dark blues guitar",
                    "prompt": "dark blues guitar",
                }
            ]
        ),
        encoding="utf-8",
    )
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"name": "dark_blues"}),
        encoding="utf-8",
    )
    bundle = dataset / "training_bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "anvil_training_bundle_version": "1.0",
                "dataset_name": "dark_blues",
                "dataset_dir": str(dataset),
                "clip_count": 1,
                "asset_count": 2,
                "clips": [
                    {
                        "id": "clip_0001",
                        "file": "clips/clip_0001.wav",
                        "caption": "dark blues guitar",
                        "prompt": "dark blues guitar",
                        "tags": ["dark blues"],
                        "negative_tags": ["muddy mix"],
                        "confidence": 0.8,
                        "seconds_start": 0.0,
                        "seconds_total": 35.0,
                        "assets": {
                            "full-mix": "clips/clip_0001.wav",
                            "instrumental": "stems/clip_0001/instrumental.wav",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return bundle
