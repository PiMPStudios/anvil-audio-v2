"""SSH transport for portable cloud job packages."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

REMOTE_RUNTIME_EXCLUDES = (".venv/", "work/", "outputs/", "logs/")


@dataclass(slots=True)
class SSHRunConfig:
    """Configuration for running a cloud job on an existing SSH host."""

    job_dir: Path
    host: str
    remote_dir: str = "~/anvil-cloud-jobs"
    port: int = 22
    identity_file: Path | None = None
    dry_run: bool = False
    skip_bootstrap: bool = False
    no_train: bool = False
    collect: bool = False


def plan_ssh_run(config: SSHRunConfig) -> list[list[str]]:
    """Build the local commands needed to run a job on an SSH host."""
    job_dir = config.job_dir.expanduser().resolve()
    _validate_job_dir(job_dir)
    remote_job_dir = _remote_job_dir(config.remote_dir, job_dir.name)

    commands: list[list[str]] = [
        _ssh_command(config, f"mkdir -p {_remote_shell_path(remote_job_dir)}"),
        _rsync_upload_command(config, job_dir, remote_job_dir),
    ]
    if not config.skip_bootstrap:
        commands.append(
            _ssh_command(
                config,
                f"bash {_remote_shell_path(remote_job_dir)}/scripts/bootstrap.sh",
            )
        )
    if not config.no_train:
        commands.append(
            _ssh_command(
                config,
                f"bash {_remote_shell_path(remote_job_dir)}/scripts/run_training.sh",
            )
        )
    if config.collect:
        commands.append(
            _ssh_command(
                config,
                f"bash {_remote_shell_path(remote_job_dir)}/scripts/collect.sh",
            )
        )
        commands.extend(_rsync_collect_commands(config, job_dir, remote_job_dir))
    return commands


def run_ssh_job(config: SSHRunConfig) -> list[list[str]]:
    """Run or print SSH commands for a portable cloud job."""
    commands = plan_ssh_run(config)
    if config.dry_run:
        return commands
    for command in commands:
        subprocess.run(command, check=True)
    return commands


def _validate_job_dir(job_dir: Path) -> None:
    job_path = job_dir / "job.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"Missing cloud job manifest: {job_path}")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "anvil_cloud_job_version" not in payload:
        raise ValueError(f"Invalid cloud job manifest: {job_path}")


def _remote_job_dir(remote_dir: str, job_name: str) -> str:
    return f"{remote_dir.rstrip('/')}/{job_name}"


def _remote_shell_path(path: str) -> str:
    if path.startswith("~/"):
        return "$HOME/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _ssh_base(config: SSHRunConfig) -> list[str]:
    command = ["ssh", "-p", str(config.port)]
    if config.identity_file is not None:
        command.extend(["-i", str(config.identity_file.expanduser())])
    command.append(config.host)
    return command


def _ssh_command(config: SSHRunConfig, remote_command: str) -> list[str]:
    return [*_ssh_base(config), remote_command]


def _rsync_upload_command(
    config: SSHRunConfig, job_dir: Path, remote_job_dir: str
) -> list[str]:
    command = [
        "rsync",
        "-az",
        "--delete",
    ]
    for excluded in REMOTE_RUNTIME_EXCLUDES:
        command.extend(["--exclude", excluded])
    command.extend(
        [
        "-e",
        _rsync_ssh_transport(config),
        f"{job_dir}/",
        f"{config.host}:{remote_job_dir}/",
        ]
    )
    return command


def _rsync_collect_commands(
    config: SSHRunConfig, job_dir: Path, remote_job_dir: str
) -> list[list[str]]:
    artifacts_dir = job_dir / "remote_artifacts"
    return [
        [
            "rsync",
            "-az",
            "-e",
            _rsync_ssh_transport(config),
            f"{config.host}:{remote_job_dir}/outputs/",
            f"{artifacts_dir}/outputs/",
        ],
        [
            "rsync",
            "-az",
            "-e",
            _rsync_ssh_transport(config),
            f"{config.host}:{remote_job_dir}/logs/",
            f"{artifacts_dir}/logs/",
        ],
    ]


def _rsync_ssh_transport(config: SSHRunConfig) -> str:
    parts = ["ssh", "-p", str(config.port)]
    if config.identity_file is not None:
        parts.extend(["-i", str(config.identity_file.expanduser())])
    return shlex.join(parts)


def format_command(command: list[str]) -> str:
    """Return a shell-friendly command preview."""
    return shlex.join(command)
