"""Fresh-host connect stores the key locally and keeps it out of the response."""

import shlex
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from runner.remote_host import (
    ConnectResult,
    _rsync_argv,
    bootstrap_shell,
    connect_gpu_host,
    redact,
    write_private_key,
)
from runner.serve import create_app
from runner.targets import SshTarget, TargetError
from fastapi.testclient import TestClient

PEM = b"-----BEGIN OPENSSH PRIVATE KEY-----\nsecret-material\n-----END OPENSSH PRIVATE KEY-----\n"


class _Tunnel:
    def __init__(self) -> None:
        self.stdout = None
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True


def _target(key: Path) -> SshTarget:
    return SshTarget(
        user="ubuntu",
        host="203.0.113.10",
        identity_file=str(key),
        local_port=8766,
        remote_port=8765,
        remote_workdir="/home/ubuntu/vllm-execution-explorer",
        remote_python=".venv/bin/python",
    )


def test_private_key_is_stored_outside_the_recording(tmp_path: Path):
    path = write_private_key(tmp_path / "keys", PEM)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert "secret-material" not in path.name
    with pytest.raises(TargetError, match="private key"):
        write_private_key(tmp_path / "keys", b"not a key")
    assert path.exists()


def test_setup_sync_preserves_spaced_path_and_excludes_local_secrets(tmp_path: Path):
    if not shutil.which("rsync"):
        pytest.skip("rsync is not installed")
    project = tmp_path / "source"
    project.mkdir()
    for name in ("app.py", ".env", ".env.local", "example.pem", "identity.key"):
        (project / name).write_text("temporary test fixture")
    target = SshTarget(user="ubuntu", host="gpu.example", remote_workdir="/srv/explorer folder")
    argv = _rsync_argv(target, project)
    assert shlex.split(argv[-1].partition(":")[2]) == ["/srv/explorer folder/"]
    destination = tmp_path / "copied"
    filters = argv[2:argv.index("-e")]
    subprocess.run(["rsync", "-a", *filters, f"{project}/", str(destination)], check=True)
    assert sorted(path.name for path in destination.iterdir()) == ["app.py"]


def test_connect_syncs_the_host_and_redacts_the_key(tmp_path: Path):
    key = write_private_key(tmp_path / "keys", PEM)
    project = tmp_path / "project"
    project.mkdir()
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "rsync":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "mkdir" in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(
            argv,
            0,
            "GPU:NVIDIA A10, 23028 MiB\nVLLM:0.30.0\n",
            "",
        )

    result = connect_gpu_host(
        _target(key),
        project_root=project,
        run=run,
        popen=lambda argv, **kwargs: calls.append(argv) or _Tunnel(),
        ready=lambda _endpoint: None,
        tunnel_grace_seconds=0,
    )

    rendered = str(result.public())
    assert "secret-material" not in rendered
    assert str(key) not in rendered
    assert "test-key.pem" not in rendered
    assert result.runner_endpoint == "http://127.0.0.1:8766"
    assert result.gpu == "NVIDIA A10, 23028 MiB"
    assert result.vllm_version == "0.30.0"
    assert result.execution_target["ssh"]["host"] == "203.0.113.10"
    assert any(call[0] == "rsync" and "--exclude" in call and ".venv" in call for call in calls)
    assert any(call[0] == "ssh" and "8766:127.0.0.1:8765" in " ".join(call) for call in calls)
    shell = bootstrap_shell(_target(key))
    assert "vllm==0.30.0" in shell
    assert "nohup sh -c '.venv/bin/pip install" not in shell
    assert "0.30.*" in shell
    assert "kill -0" in shell
    assert "--bind 127.0.0.1" in shell
    assert "0.0.0.0" not in shell
    assert str(key) not in shell


def test_failed_ssh_redacts_the_identity_path(tmp_path: Path):
    key = write_private_key(tmp_path / "keys", PEM)

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "", f"Identity file {key} not accessible")

    with pytest.raises(TargetError, match=r"\[identity file\]") as caught:
        connect_gpu_host(
            _target(key),
            project_root=tmp_path,
            run=run,
            tunnel_grace_seconds=0,
        )
    assert str(key) not in str(caught.value)
    assert redact(f"see {key}", str(key)) == "see [identity file]"


def test_connect_api_accepts_a_key_file_and_hides_it(tmp_path: Path):
    seen: dict[str, object] = {}

    def connect_host(target: SshTarget, **_kwargs: object) -> ConnectResult:
        identity = Path(str(target.identity_file))
        seen["mode"] = identity.stat().st_mode & 0o777
        seen["body"] = identity.read_bytes()
        seen["host"] = target.host
        seen["workdir"] = target.remote_workdir
        return ConnectResult(
            ok=True,
            message="Connected to ubuntu@203.0.113.10. NVIDIA A10, 23028 MiB.",
            runner_endpoint="http://127.0.0.1:8766",
            gpu="NVIDIA A10, 23028 MiB",
            vllm_version="0.30.0",
            execution_target={"kind": "ssh", "ssh": {"user": "ubuntu", "host": target.host}},
            tunnel=_Tunnel(),
        )

    client = TestClient(
        create_app(tmp_path / "runs", connect_host=connect_host, keys_dir=tmp_path / "keys")
    )
    response = client.post(
        "/api/targets/connect",
        data={"user": "ubuntu", "host": "203.0.113.10", "port": "22"},
        files={"identity": ("test-key.pem", PEM, "application/octet-stream")},
    )
    assert response.status_code == 200
    assert "secret-material" not in response.text
    assert "test-key.pem" not in response.text
    assert "identity-" not in response.text
    assert response.json()["runner_endpoint"] == "http://127.0.0.1:8766"
    assert seen["mode"] == 0o600
    assert seen["body"] == PEM
    assert seen["host"] == "203.0.113.10"
    assert seen["workdir"] == "/home/ubuntu/vllm-execution-explorer"

    rejected = client.post(
        "/api/targets/connect",
        data={"user": "ubuntu", "host": "203.0.113.10", "port": "22"},
        files={"identity": ("notes.txt", b"this file is not a credential", "text/plain")},
    )
    assert rejected.status_code == 400
    assert "this file is not a credential" not in rejected.text


def test_prepared_host_mode_skips_copy_and_install(tmp_path: Path):
    key = write_private_key(tmp_path / "keys", PEM)
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            "GPU:NVIDIA A10, 23028 MiB\nVLLM:0.30.2\n",
            "",
        )

    result = connect_gpu_host(
        _target(key),
        project_root=tmp_path,
        automatic_setup=False,
        run=run,
        popen=lambda _argv, **_kwargs: _Tunnel(),
        ready=lambda _endpoint: None,
        tunnel_grace_seconds=0,
    )

    assert result.vllm_version == "0.30.2"
    assert len(calls) == 1
    assert calls[0][0] == "ssh"
    shell = calls[0][-1]
    assert "pip install" not in shell
    assert "python3 -m venv" not in shell
    assert "enable automatic setup" in shell


@pytest.mark.parametrize(
    ("probe", "message"),
    [
        ("GPU:\nVLLM:0.30.0\n", "NVIDIA GPU"),
        ("GPU:NVIDIA A10\nVLLM:missing\n", "needs vLLM 0.30"),
        ("GPU:NVIDIA A10\nVLLM:0.29.0\n", "found 0.29.0"),
    ],
)
def test_connect_rejects_missing_gpu_or_wrong_vllm(tmp_path: Path, probe: str, message: str):
    key = write_private_key(tmp_path / "keys", PEM)

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == "rsync" or "mkdir" in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, probe, "")

    with pytest.raises(TargetError, match=message):
        connect_gpu_host(
            _target(key),
            project_root=tmp_path,
            run=run,
            tunnel_grace_seconds=0,
        )


def test_connect_api_forwards_automatic_setup_choice(tmp_path: Path):
    seen: dict[str, object] = {}

    def connect_host(target: SshTarget, **kwargs: object) -> ConnectResult:
        seen["automatic_setup"] = kwargs.get("automatic_setup")
        return ConnectResult(
            ok=True,
            message="ready",
            runner_endpoint="http://127.0.0.1:8766",
            gpu="NVIDIA A10",
            vllm_version="0.30.0",
            execution_target={"kind": "ssh", "ssh": {"user": target.user, "host": target.host}},
            tunnel=_Tunnel(),
        )

    client = TestClient(create_app(tmp_path / "runs", connect_host=connect_host, keys_dir=tmp_path / "keys"))
    response = client.post(
        "/api/targets/connect",
        data={
            "user": "ubuntu",
            "host": "203.0.113.10",
            "port": "22",
            "automatic_setup": "false",
        },
        files={"identity": ("test-key.pem", PEM, "application/octet-stream")},
    )

    assert response.status_code == 200
    assert seen["automatic_setup"] is False


def test_failed_reconnect_clears_the_previous_artifact_target(tmp_path: Path):
    connected_tunnel = _Tunnel()

    def connect_host(target: SshTarget, **_kwargs: object) -> ConnectResult:
        if target.host == "failed.example":
            raise TargetError("Host is unreachable")
        return ConnectResult(True, "ready", "http://127.0.0.1:8766", "GPU", "0.30.0", {"ssh": {"host": target.host}}, connected_tunnel)

    client = TestClient(create_app(tmp_path / "runs", connect_host=connect_host, keys_dir=tmp_path / "keys"))
    def connect(host: str, user: str = "ubuntu"):
        return client.post("/api/targets/connect", data={"user": user, "host": host}, files={"identity": ("fixture.pem", PEM)})

    assert connect("first.example").status_code == 200
    original_key = next((tmp_path / "keys").glob("identity-*.pem"))
    assert connect("next.example", user="-invalid").status_code == 400
    assert original_key.exists(), "Invalid target input must not remove the working credential"
    assert not connected_tunnel.terminated
    assert connect("failed.example").status_code == 400
    assert connected_tunnel.terminated
    assert client.get("/api/targets/status").json()["execution_target"] is None
    model = tmp_path / "model.bin"
    model.write_bytes(b"fixture weights")
    prepared = client.post("/api/artifacts/operator-local/prepare", json={"source_path": str(model), "destination": "/srv/model"}).json()
    assert client.post(f"/api/artifacts/transfers/{prepared['transfer_id']}/start").status_code == 409


def test_concurrent_connections_do_not_replace_each_others_credentials(tmp_path: Path):
    connecting = threading.Event()
    release = threading.Event()

    def connect_host(target: SshTarget, **_kwargs: object) -> ConnectResult:
        connecting.set()
        assert release.wait(5)
        assert Path(target.identity_file).is_file()
        return ConnectResult(True, "ready", "http://127.0.0.1:8766", "GPU", "0.30.0", {"ssh": {"host": target.host}}, _Tunnel())

    client = TestClient(create_app(tmp_path / "runs", connect_host=connect_host, keys_dir=tmp_path / "keys"))
    def connect():
        return client.post("/api/targets/connect", data={"user": "ubuntu", "host": "gpu.example"}, files={"identity": ("fixture.pem", PEM)})

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(connect)
        assert connecting.wait(5)
        try:
            assert connect().status_code == 409
            assert client.post("/api/targets/ssh", json={"user": "ubuntu", "host": "other.example"}).status_code == 409
        finally:
            release.set()
        assert first.result().status_code == 200
