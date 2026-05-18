"""Cloud training CLI."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from anvil_audio.cloud import (
    CloudJobPackageConfig,
    SSHRunConfig,
    create_cloud_job_package,
    format_command,
    run_ssh_job,
)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        ok = _cmd_doctor()
        raise SystemExit(0 if ok else 1)

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
