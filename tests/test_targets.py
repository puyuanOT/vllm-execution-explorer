"""Local Mac/Linux explorer versus a CUDA engine on this machine or over SSH."""

import pytest

from runner.targets import (
    LocalTarget,
    SshTarget,
    TargetError,
    decide_target,
    execution_target_record,
    parse_serve_bind,
    remote_start_shell,
    tunnel_argv,
)


def test_mac_runs_the_explorer_and_selects_an_ssh_engine_host():
    decision = decide_target(
        LocalTarget(),
        client_platform="darwin",
        cuda_available=False,
        vllm_version=None,
    )

    assert decision.ok is False
    assert decision.launch is None
    assert "Mac" in decision.message
    assert "SSH GPU host" in decision.message


def test_linux_with_cuda_and_pinned_vllm_hosts_the_engine_locally():
    decision = decide_target(
        LocalTarget(port=9000),
        client_platform="linux",
        cuda_available=True,
        vllm_version="0.30.0",
    )

    assert decision.ok is True
    assert decision.launch is not None
    assert decision.launch.engine_location == "this-machine"
    assert decision.launch.runner_endpoint == "http://127.0.0.1:9000"
    assert decision.launch.ssh_argv is None
    record = execution_target_record(decision)
    assert record["kind"] == "local"
    assert "ssh" not in record


def test_linux_without_cuda_stays_on_replay_until_an_ssh_host_is_chosen():
    decision = decide_target(
        LocalTarget(),
        client_platform="linux",
        cuda_available=False,
        vllm_version="0.30.0",
    )

    assert decision.ok is False
    assert "CUDA GPU" in decision.message


def test_local_engine_rejects_a_different_vllm_version():
    decision = decide_target(
        LocalTarget(),
        client_platform="linux",
        cuda_available=True,
        vllm_version="0.29.0",
    )

    assert decision.ok is False
    assert "0.30." in decision.message


def test_lambda_style_ssh_host_from_a_mac_does_not_need_local_cuda():
    target = SshTarget(
        user="ubuntu",
        host="192.0.2.10",
        identity_file="/keys/test-key.pem",
        local_port=8765,
        remote_port=8765,
        remote_workdir="/opt/vllm-execution-explorer",
    )

    decision = decide_target(
        target,
        client_platform="darwin",
        cuda_available=False,
        vllm_version=None,
    )

    assert decision.ok is True
    launch = decision.launch
    assert launch is not None
    assert launch.engine_location == "remote"
    assert launch.runner_endpoint == "http://127.0.0.1:8765"
    assert launch.tunnel_argv == (
        "ssh",
        "-p",
        "22",
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
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        "/keys/test-key.pem",
        "ubuntu@192.0.2.10",
        "-N",
        "-L",
        "8765:127.0.0.1:8765",
    )
    shell = remote_start_shell(target)
    assert shell == (
        "cd /opt/vllm-execution-explorer && "
        "python -m runner.serve --bind 127.0.0.1 --port 8765"
    )
    assert "0.0.0.0" not in shell
    record = execution_target_record(decision)
    assert record["ssh"] == {
        "user": "ubuntu",
        "host": "192.0.2.10",
        "remote_bind": "127.0.0.1",
    }
    assert "test-key.pem" not in str(record)


def test_ssh_agent_omits_the_identity_flag():
    argv = tunnel_argv(SshTarget(user="ubuntu", host="gpu.example"))
    assert "-i" not in argv


def test_runner_bind_stays_on_loopback():
    assert parse_serve_bind("127.0.0.1", 8765) == ("127.0.0.1", 8765)
    with pytest.raises(TargetError, match="loopback"):
        parse_serve_bind("0.0.0.0", 8765)


def test_ssh_target_rejects_shell_metacharacters():
    with pytest.raises(TargetError):
        SshTarget(user="ubuntu", host="gpu.example;rm")
    with pytest.raises(TargetError):
        SshTarget(user="ubuntu;id", host="gpu.example")


@pytest.mark.parametrize("user", ["-oProxyCommand=true", "name:host", "name\x00"])
def test_ssh_user_cannot_become_an_option_or_rsync_destination(user):
    with pytest.raises(TargetError):
        SshTarget(user=user, host="gpu.example")


def test_remote_workdir_rejects_control_characters():
    with pytest.raises(TargetError):
        SshTarget(user="ubuntu", host="gpu.example", remote_workdir="/srv/explorer\nnext")
