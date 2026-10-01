"""Copy this project to a fresh GPU host and open a loopback tunnel."""

from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from runner.targets import (
    SshTarget,
    TargetError,
    decide_target,
    execution_target_record,
    ssh_argv,
    ssh_transport,
)
from vllm_030.constants import PINNED_VLLM_PREFIX

KEY_LIMIT = 32_768
_EXCLUDES = (
    ".git",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    ".venv",
    "node_modules",
    "dist",
    "recordings",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "vllm-install.lock",
    "vllm-install.log",
    "runner.log",
    "runner.pid",
)


@dataclass
class ConnectResult:
    ok: bool
    message: str
    runner_endpoint: str
    gpu: str | None
    vllm_version: str | None
    execution_target: dict[str, Any]
    tunnel: Any = None

    def public(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "runner_endpoint": self.runner_endpoint,
            "gpu": self.gpu,
            "vllm_version": self.vllm_version,
            "execution_target": self.execution_target,
        }


def default_remote_workdir(user: str) -> str:
    home = "/root" if user == "root" else f"/home/{user}"
    return f"{home}/vllm-execution-explorer"


def write_private_key(directory: Path, payload: bytes) -> Path:
    """Store an uploaded key outside the project tree. The recording never sees it."""
    if not payload or len(payload) > KEY_LIMIT or b"PRIVATE KEY" not in payload:
        raise TargetError("Choose a private key file.")
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / f"identity-{uuid.uuid4().hex}.pem"
    path.write_bytes(payload)
    path.chmod(0o600)
    for old in directory.glob("identity-*.pem"):
        if old != path:
            old.unlink(missing_ok=True)
    return path


def redact(text: str, identity: str | None) -> str:
    if identity:
        text = text.replace(identity, "[identity file]")
    return text


def connect_gpu_host(
    target: SshTarget,
    *,
    project_root: Path,
    automatic_setup: bool = True,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    popen: Callable[..., Any] | None = None,
    ready: Callable[[str], None] | None = None,
    tunnel_grace_seconds: float = 0.5,
) -> ConnectResult:
    if target.remote_workdir is None:
        raise TargetError("remote_workdir is required")
    if target.identity_file is None:
        raise TargetError("Choose a private key file.")
    runner = run or subprocess.run
    launcher = popen or subprocess.Popen
    identity = target.identity_file
    try:
        if automatic_setup:
            _check(
                runner(
                    ssh_argv(target, f"mkdir -p {shlex.quote(target.remote_workdir)}"),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
                identity,
            )
            _check(
                runner(
                    _rsync_argv(target, project_root),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                ),
                identity,
            )
        probe = _check(
            runner(
                ssh_argv(target, bootstrap_shell(target, automatic_setup=automatic_setup)),
                check=False,
                capture_output=True,
                text=True,
                timeout=1800 if automatic_setup else 120,
            ),
            identity,
        )
    except subprocess.TimeoutExpired as exc:
        raise TargetError("The GPU host did not finish connecting in time.") from exc
    except FileNotFoundError as exc:
        raise TargetError("This computer needs ssh and rsync to reach a GPU host.") from exc
    gpu, vllm_version = _parse_probe(probe)
    if not gpu:
        raise TargetError("The execution host did not expose an NVIDIA GPU through nvidia-smi.")
    if not vllm_version or not vllm_version.startswith(PINNED_VLLM_PREFIX):
        found = vllm_version or "no vLLM install"
        raise TargetError(f"The execution host needs vLLM {PINNED_VLLM_PREFIX}x; found {found}.")
    decision = decide_target(
        target,
        client_platform=sys.platform,
        cuda_available=False,
        vllm_version=None,
    )
    if decision.launch is None or decision.launch.tunnel_argv is None:
        raise TargetError("SSH tunnel is unavailable")
    try:
        tunnel = launcher(
            list(decision.launch.tunnel_argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TargetError("This computer needs ssh and rsync to reach a GPU host.") from exc
    if tunnel_grace_seconds:
        time.sleep(tunnel_grace_seconds)
    if tunnel.poll() is not None:
        output = ""
        stream = getattr(tunnel, "stdout", None)
        if stream is not None:
            output = stream.read() or ""
        detail = redact(output.strip() or "SSH tunnel exited.", identity)
        raise TargetError(detail)
    stream = getattr(tunnel, "stdout", None)
    if stream is not None:
        threading.Thread(target=_drain, args=(stream,), daemon=True).start()
    endpoint = decision.launch.runner_endpoint
    try:
        (ready or _wait_for_local)(endpoint)
    except Exception:
        tunnel.terminate()
        raise
    return ConnectResult(
        ok=True,
        message=_message(target, gpu, vllm_version, endpoint),
        runner_endpoint=endpoint,
        gpu=gpu,
        vllm_version=vllm_version,
        execution_target=execution_target_record(decision),
        tunnel=tunnel,
    )


def bootstrap_shell(target: SshTarget, *, automatic_setup: bool = True) -> str:
    """Prepare or validate a remote runtime, then start it on loopback."""
    if target.remote_workdir is None:
        raise TargetError("remote_workdir is required")
    workdir = shlex.quote(target.remote_workdir)
    port = target.remote_port
    setup = """
if ! command -v python3 >/dev/null 2>&1; then
  echo 'python3 is required on the execution host' >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import fastapi,uvicorn,multipart' >/dev/null 2>&1; then
  .venv/bin/pip install -q 'fastapi>=0.115' 'uvicorn>=0.32' 'python-multipart>=0.0.9'
fi
version=$(.venv/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)
case "$version" in
  0.30.*) ;;
  *)
    lock=vllm-install.lock
    waited=0
    while [ -e "$lock" ]; do
      lock_owner=$(cat "$lock" 2>/dev/null || true)
      case "$lock_owner" in
        ''|*[!0-9]*)
          lock_owner=$(pgrep -f '[.]venv/bin/pip install.*vllm==0.30.0' 2>/dev/null | head -n 1 || true)
          ;;
      esac
      if [ -z "$lock_owner" ] || ! kill -0 "$lock_owner" 2>/dev/null; then
        rm -f "$lock"
        break
      fi
      if [ "$waited" -ge 900 ]; then
        echo 'timed out waiting for the existing vLLM installation' >&2
        exit 1
      fi
      sleep 2
      waited=$((waited+2))
    done
    printf '%s\n' "$$" > "$lock"
    trap 'rm -f vllm-install.lock' EXIT HUP INT TERM
    if ! .venv/bin/pip install -q --upgrade 'vllm==0.30.0' ninja > vllm-install.log 2>&1; then
      tail -n 40 vllm-install.log >&2 || true
      exit 1
    fi
    rm -f "$lock"
    trap - EXIT HUP INT TERM
    ;;
esac
""" if automatic_setup else """
if [ ! -x .venv/bin/python ]; then
  echo 'the prepared host has no .venv runtime; enable automatic setup' >&2
  exit 1
fi
if ! .venv/bin/python -c 'import fastapi,uvicorn,multipart' >/dev/null 2>&1; then
  echo 'the prepared host is missing controller dependencies; enable automatic setup' >&2
  exit 1
fi
"""
    return f"""set -eu
cd {workdir}
{setup}
version=$(.venv/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)
case "$version" in
  0.30.*) ;;
  *) echo "prepared host needs vLLM 0.30.x; found ${{version:-missing}}" >&2; exit 1 ;;
esac
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'nvidia-smi is unavailable on the execution host' >&2
  exit 1
fi
gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || true)
if [ -z "$gpu" ]; then
  echo 'no NVIDIA GPU is visible on the execution host' >&2
  exit 1
fi
if ! .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  echo 'PyTorch cannot access CUDA on the execution host' >&2
  exit 1
fi
if [ -f runner.pid ]; then
  old=$(cat runner.pid || true)
  case "$old" in
    ''|*[!0-9]*) ;;
    *) kill "$old" 2>/dev/null || true; sleep 1 ;;
  esac
fi
nohup env PYTHONUNBUFFERED=1 PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m runner.serve --bind 127.0.0.1 --port {port} > runner.log 2>&1 &
echo $! > runner.pid
i=0
ready=0
while [ "$i" -lt 40 ]; do
  if .venv/bin/python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:{port}/api/health", timeout=2)' >/dev/null 2>&1; then
    ready=1
    break
  fi
  i=$((i+1))
  sleep 0.5
done
if [ "$ready" -ne 1 ]; then
  echo 'remote runner did not start' >&2
  tail -n 40 runner.log >&2 || true
  exit 1
fi
echo "GPU:$gpu"
echo "VLLM:$version"
"""


def _rsync_argv(target: SshTarget, project_root: Path) -> list[str]:
    if target.remote_workdir is None:
        raise TargetError("remote_workdir is required")
    remote_shell = " ".join(shlex.quote(part) for part in ssh_transport(target))
    argv = ["rsync", "-az"]
    for name in _EXCLUDES:
        argv.extend(["--exclude", name])
    argv.extend(
        [
            "-e",
            remote_shell,
            f"{project_root.resolve()}/",
            f"{target.user}@{target.host}:{shlex.quote(target.remote_workdir + '/')}",
        ]
    )
    return argv


def _parse_probe(stdout: str) -> tuple[str | None, str | None]:
    gpu = None
    version = None
    for line in stdout.splitlines():
        if line.startswith("GPU:"):
            gpu = line[4:].strip() or None
        elif line.startswith("VLLM:"):
            value = line[5:].strip()
            version = None if value in {"", "missing"} else value
    if gpu == "unavailable":
        gpu = None
    return gpu, version


def _message(target: SshTarget, gpu: str | None, version: str | None, endpoint: str) -> str:
    host = f"{target.user}@{target.host}"
    gpu_text = gpu or "GPU name unavailable"
    engine = f"vLLM {version} is installed"
    return (
        f"Connected to {host}. {gpu_text}. {engine}. "
        f"Runs use {endpoint}. The private key stays on this computer."
    )


def _check(completed: subprocess.CompletedProcess[str], identity: str) -> str:
    if completed.returncode != 0:
        detail = f"{completed.stderr or ''}{completed.stdout or ''}".strip()
        detail = detail or f"command exited {completed.returncode}"
        raise TargetError(redact(detail[-2000:], identity))
    return completed.stdout or ""


def _wait_for_local(endpoint: str) -> None:
    import urllib.error
    import urllib.request

    url = endpoint.rstrip("/") + "/api/health"
    last = "no response"
    for _ in range(40):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last = str(exc)
        time.sleep(0.25)
    raise TargetError(f"The SSH tunnel did not reach the remote runner ({last}).")


def _drain(stream: Any) -> None:
    try:
        for _line in stream:
            pass
    except Exception:
        return
