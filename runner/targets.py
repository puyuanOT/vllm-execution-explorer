"""Where a live vLLM engine runs.

The explorer itself runs on the operator's Mac or Linux machine. The engine
process is a CUDA Linux program. It can be this machine, when this machine is
that host, or one existing SSH host such as a Lambda GPU instance. The remote
runner listens on loopback; the laptop reaches it through an SSH tunnel.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlunparse

from vllm_030.constants import PINNED_VLLM_PREFIX

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class TargetError(ValueError):
    pass


@dataclass(frozen=True)
class LocalTarget:
    """Start the engine on this machine."""

    kind: str = "local"
    port: int = 8765


@dataclass(frozen=True)
class SshTarget:
    """Use one already-provisioned GPU host. Lambda instances are this shape."""

    user: str
    host: str
    port: int = 22
    identity_file: str | None = None
    remote_port: int = 8765
    local_port: int = 8765
    remote_workdir: str | None = None
    remote_python: str = "python"
    kind: str = "ssh"

    def __post_init__(self) -> None:
        _require_token("user", self.user)
        _require_hostname("host", self.host)
        _require_port("port", self.port)
        _require_port("remote_port", self.remote_port)
        _require_port("local_port", self.local_port)
        if self.identity_file is not None and not self.identity_file.strip():
            raise TargetError("identity_file is empty")
        if self.remote_workdir is not None:
            _require_remote_path(self.remote_workdir)


@dataclass(frozen=True)
class LaunchPlan:
    engine_location: str
    client_platform: str
    runner_endpoint: str
    remote_bind: str
    ssh_user: str | None = None
    ssh_host: str | None = None
    ssh_argv: tuple[str, ...] | None = None
    tunnel_argv: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TargetDecision:
    ok: bool
    message: str
    launch: LaunchPlan | None = None


def decide_target(
    target: LocalTarget | SshTarget,
    *,
    client_platform: str,
    cuda_available: bool,
    vllm_version: str | None,
) -> TargetDecision:
    """Choose a launch plan from the operator's machine and the selected host."""
    if isinstance(target, SshTarget):
        launch = LaunchPlan(
            engine_location="remote",
            client_platform=client_platform,
            runner_endpoint=_endpoint(target.local_port),
            remote_bind="127.0.0.1",
            ssh_user=target.user,
            ssh_host=target.host,
            ssh_argv=tuple(remote_start_argv(target)),
            tunnel_argv=tuple(tunnel_argv(target)),
        )
        return TargetDecision(
            ok=True,
            message=(
                f"The explorer on this {_platform_name(client_platform)} opens an SSH tunnel to "
                f"{target.user}@{target.host}. The vLLM engine runs on that GPU host."
            ),
            launch=launch,
        )

    if client_platform != "linux":
        return TargetDecision(
            ok=False,
            message=(
                f"This {_platform_name(client_platform)} runs the explorer, replay, and live status. "
                "A live vLLM 0.30.0 engine needs a CUDA Linux host, so choose an SSH GPU host."
            ),
        )
    if not cuda_available:
        return TargetDecision(
            ok=False,
            message=(
                "This Linux machine runs the explorer and replay. A live engine needs a CUDA GPU "
                "on this machine, or an SSH GPU host."
            ),
        )
    if vllm_version is None or not vllm_version.startswith(PINNED_VLLM_PREFIX):
        found = vllm_version or "no vLLM install"
        return TargetDecision(
            ok=False,
            message=f"Found {found}. The live engine is pinned to vLLM {PINNED_VLLM_PREFIX}x.",
        )
    return TargetDecision(
        ok=True,
        message="This Linux machine can host the vLLM 0.30.0 engine.",
        launch=LaunchPlan(
            engine_location="this-machine",
            client_platform=client_platform,
            runner_endpoint=_endpoint(target.port),
            remote_bind="127.0.0.1",
        ),
    )


def tunnel_argv(target: SshTarget) -> list[str]:
    """Forward the laptop's local port to the runner's loopback port."""
    forward = f"{target.local_port}:127.0.0.1:{target.remote_port}"
    return [*_ssh_base(target), "-N", "-L", forward]


def remote_start_argv(target: SshTarget) -> list[str]:
    """Start the remote runner bound to loopback, so it is not on the public NIC."""
    return ssh_argv(target, remote_start_shell(target))


def ssh_argv(target: SshTarget, remote_command: str) -> list[str]:
    if not remote_command.strip():
        raise TargetError("remote command is empty")
    return [*_ssh_base(target), remote_command]


def ssh_transport(target: SshTarget) -> list[str]:
    """SSH options for rsync. The user@host destination is left to rsync."""
    base = _ssh_base(target)
    if "@" not in base[-1]:
        raise TargetError("ssh destination is missing")
    return base[:-1]


def remote_start_shell(target: SshTarget) -> str:
    serve = (
        f"{shlex.quote(target.remote_python)} -m runner.serve "
        f"--bind 127.0.0.1 --port {target.remote_port}"
    )
    if target.remote_workdir:
        return f"cd {shlex.quote(target.remote_workdir)} && {serve}"
    return serve


def execution_target_record(decision: TargetDecision) -> dict[str, Any]:
    """Provenance stored in the run manifest. The SSH private key stays out of it."""
    if decision.launch is None:
        raise TargetError("a declined target has no execution record")
    launch = decision.launch
    record: dict[str, Any] = {
        "kind": "ssh" if launch.engine_location == "remote" else "local",
        "engine_location": launch.engine_location,
        "client_platform": launch.client_platform,
        "runner_endpoint": launch.runner_endpoint,
        "remote_bind": launch.remote_bind,
    }
    if launch.ssh_user and launch.ssh_host:
        record["ssh"] = {
            "user": launch.ssh_user,
            "host": launch.ssh_host,
            "remote_bind": launch.remote_bind,
        }
    return record


def parse_serve_bind(bind: str, port: int) -> tuple[str, int]:
    """Accept only a loopback listener. Remote access is the SSH tunnel."""
    _require_port("port", port)
    if bind not in _LOOPBACK:
        raise TargetError(
            "the runner listens on loopback. Reach a remote GPU host through an SSH tunnel."
        )
    return bind, port


def _ssh_base(target: SshTarget) -> list[str]:
    argv = [
        "ssh",
        "-p",
        str(target.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=30",
    ]
    if target.identity_file:
        argv.extend(["-o", "IdentitiesOnly=yes", "-i", target.identity_file])
    argv.append(f"{target.user}@{target.host}")
    return argv


def _endpoint(port: int) -> str:
    return urlunparse(("http", f"127.0.0.1:{port}", "", "", "", ""))


def _platform_name(platform_name: str) -> str:
    if platform_name == "darwin":
        return "Mac"
    if platform_name == "linux":
        return "Linux machine"
    return platform_name


def _require_port(label: str, port: int) -> None:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise TargetError(f"{label} must be a TCP port")


def _require_token(label: str, value: str) -> None:
    if (
        not value
        or any(char.isspace() for char in value)
        or any(char in value for char in "@/\"'`;&|$()<>")
    ):
        raise TargetError(f"{label} must be a single SSH user name")


def _require_hostname(label: str, value: str) -> None:
    if not value or any(char.isspace() for char in value):
        raise TargetError(f"{label} must be a hostname or address")
    if any(char in value for char in "\"'`;&|$()<>"):
        raise TargetError(f"{label} contains shell metacharacters")


def _require_remote_path(value: str) -> None:
    if not value.startswith("/") or any(char in value for char in "\"'`;&|$()<>"):
        raise TargetError("remote_workdir must be an absolute path without shell metacharacters")
