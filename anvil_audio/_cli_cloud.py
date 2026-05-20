"""Cloud training CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from anvil_audio.cloud import (
    CloudJobPackageConfig,
    GPUFindrSearch,
    RunPodLaunchConfig,
    SSHRunConfig,
    create_cloud_job_package,
    fetch_gpufindr_offers,
    format_command,
    launch_pod,
    pod_status,
    run_ssh_job,
    ssh_target_from_pod,
    terminate_pod,
)
from anvil_audio.cloud.runpod import DEFAULT_RUNPOD_IMAGE, DEFAULT_RUNPOD_TEMPLATE_ID
from anvil_audio.cloud.job import ASSET_NAMES, available_recipes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil cloud",
        description="Package and run provider-agnostic remote training jobs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "doctor",
        help="Check local tools needed for cloud packaging and SSH runs.",
    )

    search = subparsers.add_parser(
        "search",
        help="Search live GPU availability before signing up with a provider.",
    )
    search.add_argument(
        "--gpu",
        default="",
        help="Case-insensitive GPU name filter, e.g. h200, h100, l40s.",
    )
    search.add_argument(
        "--source",
        default="",
        help="Provider filter, e.g. lambda, runpod, vast, tensordock.",
    )
    search.add_argument(
        "--location",
        default="",
        help="Location substring filter, e.g. us, tor, eu.",
    )
    search.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Max hourly total price in USD.",
    )
    search.add_argument(
        "--min-vram-gb",
        type=float,
        default=0.0,
        help="Minimum VRAM per listed offer in GB.",
    )
    search.add_argument(
        "--min-gpus",
        type=int,
        default=1,
        help="Minimum GPUs in the listed offer.",
    )
    search.add_argument(
        "--min-reliability",
        type=float,
        default=0.0,
        help="Minimum provider reliability score when available.",
    )
    search.add_argument(
        "--sort",
        default="total_cost_ph.asc",
        help="GPUFindr sort, e.g. total_cost_ph.asc or flops_per_dollar_ph.desc.",
    )
    search.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Max rows to print after local filters. Default: 15.",
    )
    search.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max GPUFindr pages to scan before local filters. Default: 5.",
    )

    package = subparsers.add_parser(
        "package",
        help="Build a portable cloud training job folder from training_bundle.json.",
    )
    package.add_argument("training_bundle", help="Path to training_bundle.json.")
    package.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the portable job package.",
    )
    package.add_argument("--name", default=None, help="Optional job name.")
    package.add_argument(
        "--base-model",
        default="acestep-v1.5-sft",
        help="Human-readable base model label stored in job.json.",
    )
    package.add_argument(
        "--model-variant",
        default="sft",
        choices=("turbo", "base", "sft", "xl_turbo", "xl_base", "xl_sft"),
        help="ACE-Step model variant passed to preprocessing/training.",
    )
    package.add_argument(
        "--recipe",
        default="lora-balanced",
        choices=available_recipes(),
        help="Training recipe preset. Default: lora-balanced.",
    )
    package.add_argument(
        "--primary-asset",
        default="full-mix",
        choices=ASSET_NAMES,
        help="Asset used as the training audio. Default: full-mix.",
    )
    package.add_argument(
        "--max-hours",
        type=float,
        default=6.0,
        help="Max remote training runtime encoded into the job. Default: 6.",
    )
    package.add_argument(
        "--checkpoint-dir",
        default="~/.cache/anvil-audio/acestep/checkpoints",
        help="Remote ACE-Step checkpoint root.",
    )
    package.add_argument(
        "--training-lyrics",
        default="[Instrumental]",
        help=(
            "Lyrics marker passed to LoRA preprocessing. Default: [Instrumental]. "
            "Use an empty string or a short vocal marker for vocal-stem jobs."
        ),
    )
    package.add_argument(
        "--repo-url",
        default=None,
        help="Git URL used by bootstrap.sh. Default: current origin or Anvil repo.",
    )
    package.add_argument(
        "--repo-ref",
        default=None,
        help="Git branch/tag/SHA used by bootstrap.sh. Default: current branch.",
    )
    package.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing job directory.",
    )

    run_ssh = subparsers.add_parser(
        "run-ssh",
        help="Upload and run a packaged job on any existing SSH GPU host.",
    )
    run_ssh.add_argument("job_dir", help="Directory from `anvil cloud package`.")
    run_ssh.add_argument(
        "--host",
        required=True,
        help="SSH host, e.g. ubuntu@203.0.113.10.",
    )
    run_ssh.add_argument(
        "--remote-dir",
        default="~/anvil-cloud-jobs",
        help="Remote parent directory. Default: ~/anvil-cloud-jobs.",
    )
    run_ssh.add_argument("--port", type=int, default=22, help="SSH port.")
    run_ssh.add_argument(
        "--identity-file",
        default=None,
        help="Optional private key path for ssh/rsync.",
    )
    run_ssh.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ssh/rsync commands without running them.",
    )
    run_ssh.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Upload and train without running scripts/bootstrap.sh.",
    )
    run_ssh.add_argument(
        "--no-train",
        action="store_true",
        help="Upload/bootstrap only; do not run scripts/run_training.sh.",
    )
    run_ssh.add_argument(
        "--collect",
        action="store_true",
        help="Run collect.sh and rsync outputs/logs back after training.",
    )

    runpod = subparsers.add_parser(
        "runpod",
        help="Launch, inspect, or terminate RunPod GPU pods.",
    )
    runpod_subparsers = runpod.add_subparsers(dest="runpod_command", required=True)

    runpod_launch = runpod_subparsers.add_parser(
        "launch",
        help="Launch a RunPod pod for a packaged cloud job.",
    )
    runpod_launch.add_argument("job_dir", help="Directory from `anvil cloud package`.")
    runpod_launch.add_argument(
        "--gpu-type",
        required=True,
        help=(
            "RunPod gpuId from `runpodctl gpu list`, "
            'e.g. "NVIDIA H200" or "NVIDIA A100-SXM4-80GB".'
        ),
    )
    runpod_launch.add_argument("--name", default=None, help="Optional pod name.")
    runpod_launch.add_argument("--gpu-count", type=int, default=1)
    runpod_launch.add_argument(
        "--image",
        default=DEFAULT_RUNPOD_IMAGE,
        help=(
            f"RunPod image used when --template-id is empty. "
            f"Default: {DEFAULT_RUNPOD_IMAGE}."
        ),
    )
    runpod_launch.add_argument(
        "--template-id",
        default=DEFAULT_RUNPOD_TEMPLATE_ID,
        help=(
            "RunPod template id. Default: "
            f"{DEFAULT_RUNPOD_TEMPLATE_ID}. Pass an empty string to use --image."
        ),
    )
    runpod_launch.add_argument(
        "--cloud-type",
        default="ALL",
        choices=("ALL", "SECURE", "COMMUNITY"),
        help="RunPod cloud type. Default: ALL.",
    )
    runpod_launch.add_argument("--volume-gb", type=int, default=100)
    runpod_launch.add_argument("--container-disk-gb", type=int, default=80)
    runpod_launch.add_argument("--min-vcpu-count", type=int, default=8)
    runpod_launch.add_argument("--min-memory-gb", type=int, default=48)
    runpod_launch.add_argument(
        "--ports",
        default="22/tcp",
        help='RunPod exposed ports. Default: "22/tcp".',
    )
    runpod_launch.add_argument(
        "--volume-mount-path",
        default="/workspace",
        help="Remote mount path. Default: /workspace.",
    )
    runpod_launch.add_argument(
        "--docker-args",
        default="sleep infinity",
        help='Container command. Default: "sleep infinity".',
    )
    runpod_launch.add_argument(
        "--allowed-cuda-version",
        action="append",
        default=[],
        help="Allowed CUDA version. Repeat for multiple versions.",
    )
    runpod_launch.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable KEY=VALUE. Repeat as needed.",
    )
    runpod_launch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the GraphQL request without launching a pod.",
    )
    runpod_launch.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "Send a UI-like minimal RunPod request: GPU/count/template/SSH only. "
            "Useful when the allocator rejects extra disk or min-spec constraints."
        ),
    )

    runpod_status = runpod_subparsers.add_parser(
        "status",
        help="Fetch RunPod pod status and SSH runtime hints.",
    )
    runpod_status.add_argument("pod_id", help="RunPod pod id.")

    runpod_terminate = runpod_subparsers.add_parser(
        "terminate",
        help="Terminate a RunPod pod.",
    )
    runpod_terminate.add_argument("pod_id", help="RunPod pod id.")
    runpod_terminate.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the GraphQL request without terminating a pod.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        ok = _cmd_doctor()
        raise SystemExit(0 if ok else 1)

    if args.command == "search":
        try:
            offers = fetch_gpufindr_offers(
                GPUFindrSearch(
                    gpu=args.gpu,
                    source=args.source,
                    location=args.location,
                    max_price=args.max_price,
                    min_vram_gb=args.min_vram_gb,
                    min_gpus=args.min_gpus,
                    min_reliability=args.min_reliability,
                    sort=args.sort,
                    limit=args.limit,
                    max_pages=args.max_pages,
                )
            )
        except Exception as exc:
            print(f"cloud search failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_gpu_offers(offers)
        return

    if args.command == "package":
        try:
            result = create_cloud_job_package(
                CloudJobPackageConfig(
                    training_bundle=Path(args.training_bundle),
                    output_dir=Path(args.output_dir),
                    name=args.name,
                    base_model=args.base_model,
                    model_variant=args.model_variant,
                    recipe=args.recipe,
                    primary_asset=args.primary_asset,
                    max_hours=args.max_hours,
                    checkpoint_dir=args.checkpoint_dir,
                    training_lyrics=args.training_lyrics,
                    repo_url=args.repo_url,
                    repo_ref=args.repo_ref,
                    force=args.force,
                )
            )
        except Exception as exc:
            print(f"cloud package failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_package_result(result)
        return

    if args.command == "run-ssh":
        try:
            commands = run_ssh_job(
                SSHRunConfig(
                    job_dir=Path(args.job_dir),
                    host=args.host,
                    remote_dir=args.remote_dir,
                    port=args.port,
                    identity_file=(
                        Path(args.identity_file) if args.identity_file else None
                    ),
                    dry_run=args.dry_run,
                    skip_bootstrap=args.skip_bootstrap,
                    no_train=args.no_train,
                    collect=args.collect,
                )
            )
        except Exception as exc:
            print(f"cloud ssh run failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_ssh_commands(commands, dry_run=args.dry_run)
        return

    if args.command == "runpod":
        _cmd_runpod(args)
        return

    parser.error(f"unknown cloud command: {args.command}")


def _cmd_doctor() -> bool:
    checks = {
        "bash": "required for generated remote scripts",
        "ssh": "required for `anvil cloud run-ssh`",
        "rsync": "required for job upload and artifact collection",
        "git": "recommended so job packages can default to the current branch",
    }
    print("Cloud runner tools:")
    ok = True
    for name, description in checks.items():
        path = shutil.which(name)
        status = "OK" if path else "MISSING"
        print(f"- {name:<5} {status:<7} {path or description}")
        if name in {"bash", "ssh", "rsync"} and path is None:
            ok = False
    return ok


def _print_gpu_offers(offers) -> None:
    if not offers:
        print("No matching GPU offers found.")
        return
    print(
        f"{'GPU':<16} {'PROVIDER':<12} {'LOCATION':<18} "
        f"{'PRICE':>9} {'VRAM':>7} {'COUNT':>5} {'REL':>5} URL"
    )
    print("-" * 104)
    for offer in offers:
        print(
            f"{offer.name[:16]:<16} "
            f"{offer.source[:12]:<12} "
            f"{offer.location[:18]:<18} "
            f"${offer.total_cost_ph:>7.2f} "
            f"{offer.vram_gb:>6.0f}G "
            f"{offer.num_gpus:>5} "
            f"{offer.reliability:>5.2f} "
            f"{offer.url}"
        )


def _cmd_runpod(args) -> None:
    if args.runpod_command == "launch":
        try:
            result = launch_pod(
                RunPodLaunchConfig(
                    job_dir=Path(args.job_dir),
                    gpu_type=args.gpu_type,
                    name=args.name,
                    gpu_count=args.gpu_count,
                    image_name=args.image,
                    template_id=args.template_id or None,
                    cloud_type=args.cloud_type,
                    volume_gb=args.volume_gb,
                    container_disk_gb=args.container_disk_gb,
                    min_vcpu_count=args.min_vcpu_count,
                    min_memory_gb=args.min_memory_gb,
                    ports=args.ports,
                    volume_mount_path=args.volume_mount_path,
                    docker_args=args.docker_args,
                    allowed_cuda_versions=tuple(args.allowed_cuda_version),
                    env=_parse_env_pairs(args.env),
                    minimal=args.minimal,
                    dry_run=args.dry_run,
                )
            )
        except Exception as exc:
            print(f"runpod launch failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_runpod_result(result)
        return

    if args.runpod_command == "status":
        try:
            result = pod_status(args.pod_id)
        except Exception as exc:
            print(f"runpod status failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_runpod_result(result)
        return

    if args.runpod_command == "terminate":
        try:
            result = terminate_pod(args.pod_id, dry_run=args.dry_run)
        except Exception as exc:
            print(f"runpod terminate failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print_runpod_result(result)
        return

    raise SystemExit(f"unknown runpod command: {args.runpod_command}")


def _print_runpod_result(result) -> None:
    if isinstance(result, dict) and result.get("dry_run"):
        print(json.dumps(result["request"], indent=2, sort_keys=True))
        return
    print(json.dumps(result, indent=2, sort_keys=True))
    if isinstance(result, dict):
        ssh_target = ssh_target_from_pod(result)
        if ssh_target:
            print(f"\nSSH target: {ssh_target}")
            print("Use the printed host/port with `anvil cloud run-ssh --host ...`.")


def _parse_env_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--env must be KEY=VALUE, got: {value}")
        key, raw = value.split("=", 1)
        if not key.strip():
            raise SystemExit(f"--env key cannot be blank: {value}")
        env[key.strip()] = raw
    return env


def _print_package_result(result) -> None:
    print(f"Cloud job: {result.job_dir}")
    print(f"Manifest: {result.job_path}")
    print(f"Dataset: {result.dataset_dir}")
    print(f"Clips: {result.clip_count}")
    print(f"Assets: {result.asset_count}")
    if result.warnings:
        print(f"Warnings: {len(result.warnings)}")
        for warning in result.warnings[:10]:
            print(f"- {warning}")
        if len(result.warnings) > 10:
            print(f"- ... {len(result.warnings) - 10} more")


def _print_ssh_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    header = "Planned commands:" if dry_run else "Ran commands:"
    print(header)
    for command in commands:
        print(format_command(command))


if __name__ == "__main__":
    main()
