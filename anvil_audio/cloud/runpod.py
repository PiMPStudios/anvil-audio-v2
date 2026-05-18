"""RunPod provider adapter for cloud training pods."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
DEFAULT_RUNPOD_TEMPLATE_ID = "runpod-torch-v280"
DEFAULT_RUNPOD_PORTS = "22/tcp"
DEFAULT_RUNPOD_VOLUME_MOUNT = "/workspace"


@dataclass(slots=True)
class RunPodLaunchConfig:
    """Configuration for launching a RunPod pod for an Anvil job."""

    job_dir: Path
    gpu_type: str
    name: str | None = None
    gpu_count: int = 1
    image_name: str = DEFAULT_RUNPOD_IMAGE
    template_id: str | None = DEFAULT_RUNPOD_TEMPLATE_ID
    cloud_type: str = "ALL"
    volume_gb: int = 100
    container_disk_gb: int = 80
    min_vcpu_count: int = 8
    min_memory_gb: int = 48
    ports: str = DEFAULT_RUNPOD_PORTS
    volume_mount_path: str = DEFAULT_RUNPOD_VOLUME_MOUNT
    docker_args: str = "sleep infinity"
    allowed_cuda_versions: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    minimal: bool = False
    api_key: str | None = None
    dry_run: bool = False


@dataclass(slots=True)
class RunPodRequest:
    """Prepared RunPod GraphQL request."""

    query: str
    variables: dict[str, Any]


def api_key_from_env() -> str | None:
    """Return the RunPod API key from the standard Anvil environment variable."""
    return os.environ.get("RUNPOD_API_KEY")


def build_launch_request(config: RunPodLaunchConfig) -> RunPodRequest:
    """Build the GraphQL request for RunPod on-demand pod launch."""
    job_dir = config.job_dir.expanduser().resolve()
    _validate_job_dir(job_dir)
    input_payload: dict[str, Any] = {
        "cloudType": config.cloud_type,
        "gpuCount": max(1, int(config.gpu_count)),
        "gpuTypeId": config.gpu_type,
        "name": config.name or f"anvil-{job_dir.name}",
        "startSsh": True,
    }
    if not config.minimal:
        input_payload.update(
            {
                "volumeInGb": max(1, int(config.volume_gb)),
                "containerDiskInGb": max(1, int(config.container_disk_gb)),
                "minVcpuCount": max(1, int(config.min_vcpu_count)),
                "minMemoryInGb": max(1, int(config.min_memory_gb)),
                "ports": config.ports,
                "volumeMountPath": config.volume_mount_path,
                "env": _env_list(config.env | _default_job_env(job_dir)),
            }
        )
    if config.template_id:
        input_payload["templateId"] = config.template_id
    else:
        input_payload["imageName"] = config.image_name
        input_payload["dockerArgs"] = config.docker_args
    if config.allowed_cuda_versions:
        input_payload["allowedCudaVersions"] = list(config.allowed_cuda_versions)
    return RunPodRequest(
        query=(
            "mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput) "
            "{ podFindAndDeployOnDemand(input: $input) { "
            "id name imageName machineId desiredStatus costPerHr ports "
            "machine { podHostId } "
            "runtime { ports { ip isIpPublic privatePort publicPort type } } "
            "} }"
        ),
        variables={"input": input_payload},
    )


def launch_pod(config: RunPodLaunchConfig) -> dict[str, Any]:
    """Launch a RunPod pod or return a dry-run request payload."""
    request = build_launch_request(config)
    if config.dry_run:
        return {"dry_run": True, "request": _request_payload(request)}
    payload = _runpod_graphql(request, api_key=config.api_key or api_key_from_env())
    return _graphql_data(payload, "podFindAndDeployOnDemand")


def pod_status(pod_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """Return status and runtime connection information for a RunPod pod."""
    request = build_status_request(pod_id)
    payload = _runpod_graphql(request, api_key=api_key or api_key_from_env())
    return _graphql_data(payload, "pod")


def build_status_request(pod_id: str) -> RunPodRequest:
    """Build the GraphQL request for RunPod pod status."""
    return RunPodRequest(
        query=(
            "query pod($input: PodFilter) { pod(input: $input) { "
            "id name imageName machineId desiredStatus costPerHr ports "
            "machine { podHostId } "
            "runtime { ports { ip isIpPublic privatePort publicPort type } } "
            "} }"
        ),
        variables={"input": {"podId": pod_id}},
    )


def terminate_pod(pod_id: str, *, api_key: str | None = None, dry_run: bool = False) -> Any:
    """Terminate a RunPod pod."""
    request = RunPodRequest(
        query="mutation podTerminate($input: PodTerminateInput!) { podTerminate(input: $input) }",
        variables={"input": {"podId": pod_id}},
    )
    if dry_run:
        return {"dry_run": True, "request": _request_payload(request)}
    payload = _runpod_graphql(request, api_key=api_key or api_key_from_env())
    return _graphql_data(payload, "podTerminate")


def ssh_target_from_pod(pod: dict[str, Any]) -> str | None:
    """Return `root@ip -p port` guidance when RunPod exposes SSH runtime info."""
    runtime = pod.get("runtime")
    if not isinstance(runtime, dict):
        return None
    ports = runtime.get("ports")
    if not isinstance(ports, list):
        return None
    for item in ports:
        if not isinstance(item, dict):
            continue
        private_port = str(item.get("privatePort") or "")
        port_type = str(item.get("type") or "").lower()
        if private_port != "22" and "tcp" not in port_type:
            continue
        ip = str(item.get("ip") or "")
        public_port = item.get("publicPort")
        if ip and public_port:
            return f"root@{ip} -p {public_port}"
    return None


def _runpod_graphql(request: RunPodRequest, *, api_key: str | None) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is required for non-dry-run RunPod commands.")
    url = f"{RUNPOD_GRAPHQL_URL}?{urlencode({'api_key': api_key})}"
    body = json.dumps(_request_payload(request)).encode("utf-8")
    http_request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "anvil-audio/1.0"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=60) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"RunPod request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"RunPod request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RunPod returned an unexpected payload.")
    if payload.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {payload['errors']}")
    return payload


def _graphql_data(payload: dict[str, Any], field: str) -> Any:
    data = payload.get("data")
    if not isinstance(data, dict) or field not in data:
        raise RuntimeError(f"RunPod response did not include data.{field}.")
    return data[field]


def _request_payload(request: RunPodRequest) -> dict[str, Any]:
    return {"query": request.query, "variables": request.variables}


def _validate_job_dir(job_dir: Path) -> None:
    manifest = job_dir / "job.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing cloud job manifest: {manifest}")


def _default_job_env(job_dir: Path) -> dict[str, str]:
    return {
        "ANVIL_CLOUD_JOB_NAME": job_dir.name,
        "ANVIL_CLOUD_JOB_ROOT": DEFAULT_RUNPOD_VOLUME_MOUNT,
    }


def _env_list(env: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"key": str(key), "value": str(value)}
        for key, value in sorted(env.items())
        if str(key).strip()
    ]
