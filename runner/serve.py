"""Local controller API for one active experiment."""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from runner.artifacts import ArtifactTransferManager, HTTP_UPLOAD_LIMIT_BYTES, HTTP_UPLOAD_MAX_FILES
from runner.capabilities import capability_report
from runner.configuration import (
    ConfigurationError,
    apply_legacy_fields,
    copy_configuration,
    launch_blockers,
    parse_run_configuration,
)
from runner.controller import drive_run
from runner.fixture_engine import FixtureSession
from runner.isolation import ExperimentIsolation
from runner.manifest import LOCAL_FIXTURE_TARGET, manifest_skeleton
from runner.state import ACTIVE_STATUSES, RunState
from runner.trace_index import RunTraceIndex
from runner.remote_host import (
    ConnectResult,
    connect_gpu_host,
    default_remote_workdir,
    redact,
    write_private_key,
)
from runner.targets import (
    LocalTarget,
    SshTarget,
    TargetError,
    decide_target,
    execution_target_record,
    parse_serve_bind,
)
from runner.workload import (
    PROVISIONAL_LIMITS,
    WorkRequest,
    WorkloadError,
    append_request_to_plan,
    parse_request_item,
    parse_run_workload,
    require_engine_settings,
)

_UNCHANGED = object()
from vllm_030.collector import Recording


class RunRequest(BaseModel):
    engine: str = "vllm"
    requests: list[dict[str, Any]]
    workload_plan: dict[str, Any] | None = None
    max_num_seqs: int = 4
    prefix_caching: bool = True
    token_budget: int = 2048
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_source_kind: str = "huggingface"
    tokenizer: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    max_model_len: int | None = None
    chunked_prefill: bool = True
    gpu_memory_utilization: float = 0.9
    scheduling_policy: str = "fcfs"
    long_prefill_token_threshold: int = 0
    kv_cache_memory_bytes: int | None = None
    dtype: str = "bfloat16"
    quantization: str | None = None
    sampling: dict[str, Any] | None = None
    model_configuration: dict[str, Any] | None = None
    vllm_configuration: dict[str, Any] | None = None
    profiling_configuration: dict[str, Any] | None = None
    lora_adapters: list[dict[str, Any]] = Field(default_factory=list)
    huggingface_token: str | None = None


class AddedRequest(BaseModel):
    request_id: str
    prompt: str
    max_output_tokens: int
    arrival_offset_ms: int = 0
    input_mode: str = "prompt"
    target_input_tokens: int | None = None
    resolved_input_tokens: int | None = None
    adapter_id: str | None = None
    sampling: dict[str, Any] | None = None
    priority: int = 0


class ArtifactPrepareRequest(BaseModel):
    source_path: str
    destination: str


class SshRequest(BaseModel):
    user: str
    host: str
    port: int = 22
    identity_file: str | None = None
    remote_port: int = 8765
    local_port: int = 8765
    remote_workdir: str | None = None
    start_tunnel: bool = False


class ServeConfig(BaseModel):
    bind: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    recordings: str = "recordings/runs"


def parse_serve_args(argv: list[str]) -> ServeConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Loopback controller for the vLLM execution explorer")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--recordings", default="recordings/runs")
    args = parser.parse_args(argv)
    bind, port = parse_serve_bind(args.bind, args.port)
    return ServeConfig(bind=bind, port=port, recordings=args.recordings)


def create_app(
    recordings_dir: Path,
    *,
    popen: Any = None,
    connect_host: Any = None,
    keys_dir: Path | None = None,
    project_root: Path | None = None,
    session_factory: Any = None,
    recording_opener: Any = None,
    monotonic_clock: Any = None,
    monotonic_clock_ns: Any = None,
    sleeper: Any = None,
    artifact_runner: Any = None,
    artifacts_dir: Path | None = None,
) -> FastAPI:
    recordings_dir.mkdir(parents=True, exist_ok=True)
    state = RunState()
    isolation = ExperimentIsolation()
    launch_lock = threading.Lock()
    trace_indexes: dict[str, RunTraceIndex] = {}
    tunnel: dict[str, Any] = {"process": None, "target": None}
    target_lock = threading.Lock()
    connection_lock = threading.Lock()
    target_status: dict[str, Any] = {
        "status": "idle",
        "phase": "idle",
        "message": "No execution host selected.",
        "history": [],
        "execution_target": None,
    }
    clock = monotonic_clock or time.monotonic
    clock_ns = monotonic_clock_ns
    wait = sleeper or time.sleep
    open_recording = recording_opener or Recording.open
    key_store = keys_dir or (Path.home() / ".vllm-execution-explorer" / "keys")
    source_root = project_root or Path(__file__).resolve().parents[1]
    connector = connect_host or connect_gpu_host
    artifacts = ArtifactTransferManager(run=artifact_runner)
    artifact_store = artifacts_dir or (Path.home() / ".vllm-execution-explorer" / "artifacts")
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await run_in_threadpool(isolation.close)

    app = FastAPI(title="vLLM execution explorer", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/limits")
    def limits() -> dict[str, Any]:
        return PROVISIONAL_LIMITS

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return capability_report(
            platform=sys.platform,
            cuda_available=_cuda_available(),
            vllm_version=_vllm_version(),
            cuda_compute_capability=_cuda_compute_capability(),
            bitsandbytes_plugin_available=_bitsandbytes_plugin_available(),
        )

    @app.get("/api/preflight")
    def preflight() -> dict[str, Any]:
        decision = decide_target(
            LocalTarget(),
            client_platform=sys.platform,
            cuda_available=_cuda_available(),
            vllm_version=_vllm_version(),
        )
        return {"ok": decision.ok, "message": decision.message, "platform": sys.platform}

    @app.get("/api/targets/status")
    def current_target_status() -> dict[str, Any]:
        with target_lock:
            return {
                **target_status,
                "history": [dict(item) for item in target_status["history"]],
            }

    @app.post("/api/targets/ssh")
    def ssh_target(body: SshRequest) -> dict[str, Any]:
        if not connection_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An SSH host connection is already in progress.")
        try:
            return configure_ssh_target(body)
        finally:
            connection_lock.release()

    def configure_ssh_target(body: SshRequest) -> dict[str, Any]:
        try:
            target = SshTarget(
                user=body.user,
                host=body.host,
                port=body.port,
                identity_file=body.identity_file,
                remote_port=body.remote_port,
                local_port=body.local_port,
                remote_workdir=body.remote_workdir,
            )
        except TargetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        decision = decide_target(
            target,
            client_platform=sys.platform,
            cuda_available=False,
            vllm_version=None,
        )
        if body.start_tunnel and decision.launch and decision.launch.tunnel_argv:
            launcher = popen
            if launcher is None:
                import subprocess

                launcher = subprocess.Popen
            if tunnel["process"] is not None:
                tunnel["process"].terminate()
            tunnel["process"] = launcher(list(decision.launch.tunnel_argv))
            tunnel["target"] = target
        record = execution_target_record(decision)
        _set_target_status(
            target_lock,
            target_status,
            status="ready" if decision.ok else "blocked",
            phase="tunnel_ready" if body.start_tunnel and decision.ok else "target_validated",
            message=decision.message,
            execution_target=record,
        )
        return {
            "ok": decision.ok,
            "message": decision.message,
            "runner_endpoint": decision.launch.runner_endpoint if decision.launch else None,
            "execution_target": record,
            "identity_file_used": bool(body.identity_file),
        }

    @app.post("/api/targets/connect")
    async def connect_target(
        user: str = Form(...),
        host: str = Form(...),
        port: int = Form(22),
        automatic_setup: bool = Form(True),
        identity: UploadFile = File(...),
    ) -> dict[str, Any]:
        if not connection_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An SSH host connection is already in progress.")
        try:
            return await connect_target_once(user, host, port, automatic_setup, identity)
        finally:
            connection_lock.release()

    async def connect_target_once(
        user: str,
        host: str,
        port: int,
        automatic_setup: bool,
        identity: UploadFile,
    ) -> dict[str, Any]:
        payload = await identity.read(32_769)
        try:
            target = SshTarget(
                user=user.strip(),
                host=host.strip(),
                port=port,
                remote_port=8765,
                local_port=8766,
                remote_workdir=default_remote_workdir(user.strip()),
                remote_python=".venv/bin/python",
            )
            key_path = write_private_key(key_store, payload)
            target = replace(target, identity_file=str(key_path))
        except TargetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if tunnel["process"] is not None:
            tunnel["process"].terminate()
            tunnel["process"] = None
        tunnel["target"] = None
        _set_target_status(
            target_lock,
            target_status,
            status="connecting",
            phase="checking_ssh_and_preparing_runtime" if automatic_setup else "validating_prepared_runtime",
            message=(
                "Checking the SSH host and preparing the isolated runtime."
                if automatic_setup
                else "Checking the SSH host and validating its existing runtime."
            ),
            execution_target=None,
        )
        launcher = popen
        try:
            result = await asyncio.to_thread(
                connector,
                target,
                project_root=source_root,
                automatic_setup=automatic_setup,
                popen=launcher,
                progress=lambda phase, message: _set_target_status(
                    target_lock, target_status, status="connecting", phase=phase, message=message,
                ),
            )
        except TargetError as exc:
            _set_target_status(target_lock, target_status, status="failed", phase="connection_failed", message=redact(str(exc), str(key_path)))
            raise HTTPException(status_code=400, detail=redact(str(exc), str(key_path))) from exc
        except Exception as exc:
            _set_target_status(target_lock, target_status, status="failed", phase="connection_failed", message=redact(str(exc), str(key_path)))
            raise HTTPException(status_code=400, detail=redact(str(exc), str(key_path))) from exc
        if not isinstance(result, ConnectResult) or not result.ok:
            _set_target_status(
                target_lock,
                target_status,
                status="failed",
                phase="connection_failed",
                message="The GPU host did not connect.",
            )
            raise HTTPException(status_code=400, detail="The GPU host did not connect.")
        body = result.public()
        if str(key_path) in json.dumps(body):
            if result.tunnel is not None:
                result.tunnel.terminate()
            raise HTTPException(status_code=400, detail="The connection response was discarded.")
        tunnel["process"] = result.tunnel
        tunnel["target"] = target
        _set_target_status(
            target_lock,
            target_status,
            status="ready",
            phase="runtime_and_tunnel_ready",
            message=str(body.get("message") or "Execution host connected."),
            execution_target=body.get("execution_target"),
        )
        return body

    @app.post("/api/artifacts/operator-local/prepare")
    def prepare_artifact(body: ArtifactPrepareRequest) -> dict[str, Any]:
        try:
            return artifacts.prepare(body.source_path, body.destination)
        except TargetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/artifacts/operator-local/upload")
    async def upload_artifact(
        destination: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if not files or len(files) > HTTP_UPLOAD_MAX_FILES:
            raise HTTPException(status_code=400, detail=f"Choose from 1 to {HTTP_UPLOAD_MAX_FILES} model files.")
        upload_id = f"upload-{uuid.uuid4().hex[:12]}"
        directory = artifact_store / upload_id
        directory.mkdir(parents=True, exist_ok=False)
        total = 0
        names: set[str] = set()
        try:
            for upload in files:
                name = Path(upload.filename or "").name
                if not name or name in names or name in {".", ".."}:
                    raise TargetError("Uploaded model filenames must be unique and non-empty.")
                names.add(name)
                path = directory / name
                with path.open("wb") as handle:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > HTTP_UPLOAD_LIMIT_BYTES:
                            raise TargetError("The uploaded model artifact exceeds the HTTP staging limit.")
                        handle.write(chunk)
                await upload.close()
            result = artifacts.prepare(str(directory), destination)
            result["upload"] = {
                "status": "staged",
                "file_count": len(files),
                "total_bytes": total,
                "retained_for_retry": True,
            }
            return result
        except (OSError, TargetError) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/artifacts/transfers/{transfer_id}")
    def artifact_status(transfer_id: str) -> dict[str, Any]:
        try:
            return artifacts.snapshot(transfer_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact transfer not found") from exc

    @app.post("/api/artifacts/transfers/{transfer_id}/start")
    def start_artifact_transfer(transfer_id: str) -> dict[str, Any]:
        target = tunnel.get("target")
        if not isinstance(target, SshTarget):
            raise HTTPException(status_code=409, detail="Connect an SSH execution host before transferring artifacts.")
        try:
            return artifacts.start(transfer_id, target)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="artifact transfer not found") from exc

    @app.post("/api/artifacts/transfers/{transfer_id}/retry")
    def retry_artifact_transfer(transfer_id: str) -> dict[str, Any]:
        return start_artifact_transfer(transfer_id)

    @app.post("/api/runs")
    def start_run(body: RunRequest) -> dict[str, Any]:
        nonlocal state
        if body.engine not in {"vllm", "fixture"}:
            raise HTTPException(status_code=400, detail="engine must be vllm or fixture")
        if body.engine == "vllm":
            decision = decide_target(
                LocalTarget(),
                client_platform=sys.platform,
                cuda_available=_cuda_available(),
                vllm_version=_vllm_version(),
            )
            if not decision.ok:
                raise HTTPException(status_code=409, detail=decision.message)
        try:
            requests, plan = parse_run_workload({"requests": body.requests, "workload_plan": body.workload_plan})
            payload = body.model_dump()
            configuration = parse_run_configuration(payload, requests, plan)
            blockers = launch_blockers(configuration, requests, body.engine)
            if blockers:
                raise ConfigurationError(" ".join(blockers), field="configuration")
            apply_legacy_fields(body, configuration)
        except (WorkloadError, ConfigurationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if body.engine == "vllm":
            quantization = configuration["model"]["quantization"]["kind"]
            quantization_capability = capability_report(
                platform=sys.platform,
                cuda_available=_cuda_available(),
                vllm_version=_vllm_version(),
                cuda_compute_capability=_cuda_compute_capability(),
                bitsandbytes_plugin_available=_bitsandbytes_plugin_available(),
            )["model_options"]["quantization"][quantization]
            if not quantization_capability["selectable"]:
                raise HTTPException(status_code=409, detail=quantization_capability["message"])
        with launch_lock:
            production_process = body.engine == "vllm" and session_factory is None and recording_opener is None
            with state.lock:
                if state.status in ACTIVE_STATUSES or state.execution_process_alive:
                    raise HTTPException(status_code=409, detail="one experiment is already running")
            if production_process:
                state = isolation.share(state)
            with state.lock:
                if state.status in ACTIVE_STATUSES or state.execution_process_alive:
                    raise HTTPException(status_code=409, detail="one experiment is already running")
                run_id = f"run-{uuid.uuid4().hex[:12]}"
                state.run_id = run_id
                state.status = "preparing"
                state.intake_open = True
                state.cancelled = False
                state.text = {item.request_id: "" for item in requests}
                state.message = ""
                state.extra = []
                state.plan = plan
                state.origin = None
                state.arrivals = []
                state.finalized = False
                state.configuration = copy_configuration(configuration)
                state.phase = "configuration_accepted"
                state.phase_history = [_phase_record("configuration_accepted")]
                state.execution_process = None
            arguments = {
                    "recordings_dir": recordings_dir,
                    "run_id": run_id,
                    "requests": requests,
                    "body": body,
                    "session_factory": session_factory,
                    "recording_opener": open_recording,
                    "clock": clock,
                    "clock_ns": clock_ns,
                    "sleeper": wait,
                }
            if production_process:
                isolation.start(state, arguments)
            else:
                threading.Thread(target=_execute, kwargs={"state": state, **arguments}, daemon=True).start()
        return {
            "run_id": run_id,
            "engine": body.engine,
            "configuration_snapshot_sha256": configuration["snapshot_sha256"],
        }

    @app.get("/api/runs/current")
    def current() -> dict[str, Any]:
        return state.snapshot()

    @app.post("/api/runs/current/requests")
    def add_request(body: AddedRequest) -> dict[str, bool]:
        try:
            request = parse_request_item(body.model_dump())
        except WorkloadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with state.lock:
            if state.status not in ACTIVE_STATUSES or not state.intake_open:
                raise HTTPException(status_code=409, detail="intake is closed")
            configuration = state.configuration or {}
            if request.priority and (configuration.get("vllm") or {}).get("scheduling_policy", "fcfs") != "priority":
                raise HTTPException(status_code=400, detail="Nonzero request priorities require the priority scheduling policy.")
            model = configuration.get("model") or {}
            known_adapters = {"base", *(item.get("id") for item in model.get("lora_adapters") or [])}
            if (request.adapter_id or "base") not in known_adapters:
                raise HTTPException(status_code=400, detail="request selects an unattached adapter")
            context = int((configuration.get("vllm") or {}).get("max_model_len") or 0)
            if request.resolved_input_tokens is not None and context and (
                request.resolved_input_tokens + request.max_output_tokens > context
            ):
                raise HTTPException(status_code=400, detail="request exceeds the configured model context")
            known = set(state.text) | {item.request_id for item in state.extra}
            if request.request_id in known:
                raise HTTPException(status_code=400, detail="request id is already in this run")
            if len(known) >= int(PROVISIONAL_LIMITS["max_requests"]):
                raise HTTPException(status_code=400, detail="request limit reached")
            state.extra.append(request)
            return {"accepted": True}

    @app.get("/api/runs/{run_id}/trace")
    def run_trace(
        run_id: str,
        after_sequence: int = 0,
        after_content_sequence: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        _reject_run_id(run_id)
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise HTTPException(status_code=400, detail="after_sequence must be a nonnegative integer")
        if isinstance(after_content_sequence, bool) or not isinstance(after_content_sequence, int) or after_content_sequence < 0:
            raise HTTPException(status_code=400, detail="after_content_sequence must be a nonnegative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 2000:
            raise HTTPException(status_code=400, detail="limit must be from 1 to 2000")
        directory = recordings_dir / run_id
        if not directory.is_dir():
            raise HTTPException(status_code=404, detail="recording not found")
        index = trace_indexes.get(run_id)
        if index is None:
            index = RunTraceIndex(directory)
            trace_indexes[run_id] = index
        events, next_event, events_truncated = index.events.page(after_sequence, limit)
        content, next_content, content_truncated = index.content.page(after_content_sequence, limit)
        with state.lock:
            current = state.run_id == run_id
            status = state.status if current else "idle"
            plan = state.plan if current else None
            origin = state.origin if current else None
            arrivals = list(state.arrivals) if current else []
            message = state.message if current else ""
            capture = {
                "status": state.status,
                "phase": state.phase,
                "history": [dict(item) for item in state.phase_history],
                "finalized": state.finalized,
            } if current else None
            process_alive = current and state.execution_process_alive
            if process_alive and status not in ACTIVE_STATUSES:
                status = "finalizing"
                capture.update(status=status, phase="reaping_execution_process", finalized=False)
        manifest_path = directory / "manifest.json"
        manifest = None
        if manifest_path.is_file() and not process_alive:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = str(manifest.get("completion_status") or status)
            plan = manifest.get("workload_plan", plan)
            origin = manifest.get("controller_origin", origin)
            arrivals = list(manifest.get("arrivals") or arrivals)
            capture = manifest.get("capture", capture)
        return {
            "run_id": run_id,
            "status": status,
            "capture_state": status,
            "capture": capture,
            "finalized": manifest is not None,
            "provisional": manifest is None,
            "events": events,
            "content": content,
            "next_event_sequence": next_event,
            "next_content_sequence": next_content,
            "events_truncated": events_truncated,
            "content_truncated": content_truncated,
            "workload_plan": plan,
            "controller_origin": origin,
            "arrivals": arrivals,
            "message": message,
            "manifest": manifest,
        }

    @app.post("/api/runs/current/close-intake")
    def close_intake() -> dict[str, bool]:
        with state.lock:
            state.intake_open = False
        return {"intake_open": False}

    @app.post("/api/runs/current/cancel")
    def cancel() -> dict[str, str]:
        with state.lock:
            state.cancelled = True
            state.intake_open = False
        return {"status": "cancel_requested"}

    @app.get("/api/recordings/{run_id}/manifest")
    def recording_manifest(run_id: str) -> dict[str, Any]:
        return json.loads(_recording_file(recordings_dir, run_id, "manifest.json").read_text(encoding="utf-8"))

    @app.get("/api/recordings/{run_id}/events")
    def recording_events(run_id: str) -> PlainTextResponse:
        return PlainTextResponse(_recording_file(recordings_dir, run_id, "events.jsonl").read_text(encoding="utf-8"))

    @app.get("/api/recordings/{run_id}/content")
    def recording_content(run_id: str) -> PlainTextResponse:
        _reject_run_id(run_id)
        path = recordings_dir / run_id / "content.jsonl"
        if not path.is_file():
            return PlainTextResponse("")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @app.get("/api/recordings/{run_id}/profile")
    def recording_profile(run_id: str) -> PlainTextResponse:
        return PlainTextResponse(_recording_file(recordings_dir, run_id, "profiling/activities.json").read_text(encoding="utf-8"))

    return app


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _bitsandbytes_plugin_available() -> bool:
    if not _module_available("vllm_bnb_plugin"):
        return False
    try:
        discovered = importlib.metadata.entry_points()
        if hasattr(discovered, "select"):
            candidates = discovered.select(group="vllm.general_plugins", name="bitsandbytes")
        else:  # pragma: no cover - compatibility with older importlib.metadata APIs
            candidates = (
                item
                for item in discovered.get("vllm.general_plugins", [])
                if item.name == "bitsandbytes"
            )
        return any(
            item.value.partition(":")[0] == "vllm_bnb_plugin"
            for item in candidates
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _cuda_compute_capability() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return int(major), int(minor)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None


def _reject_run_id(run_id: str) -> None:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise HTTPException(status_code=404, detail="recording not found")


def _recording_file(recordings_dir: Path, run_id: str, name: str) -> Path:
    _reject_run_id(run_id)
    path = recordings_dir / run_id / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="recording not found")
    return path


def _execute(
    state: RunState,
    recordings_dir: Path,
    run_id: str,
    requests: list[WorkRequest],
    body: RunRequest,
    *,
    session_factory: Any = None,
    recording_opener: Any = Recording.open,
    clock: Any = time.monotonic,
    clock_ns: Any = None,
    sleeper: Any = time.sleep,
) -> None:
    recording: Any = None
    session: Any = None
    completion = "failed"
    tokens: dict[str, list[int]] = {}
    arrivals: list[dict[str, object]] = []
    origin: dict[str, object] = {}
    notes: dict[str, str] = {}
    prompted: set[str] = set()
    cleanup_error = ""
    manifest_error = ""
    try:
        _set_phase(state, "preparing", "initializing_recording")
        recording = recording_opener(recordings_dir / run_id, run_id=run_id)
        for request in requests:
            prompted.add(request.request_id)
            if request.input_mode == "synthetic":
                recording.content(
                    kind="synthetic_input",
                    request_id=request.request_id,
                    target_input_tokens=request.target_input_tokens,
                    resolved_input_tokens=request.resolved_input_tokens,
                )
            else:
                recording.content(kind="prompt_text", request_id=request.request_id, text=request.prompt)
        _set_phase(state, "preparing", "preparing_execution_runtime")
        if session_factory is not None:
            session = session_factory(recording, body)
        elif body.engine == "fixture":
            session = FixtureSession(
                recording,
                max_num_seqs=body.max_num_seqs,
                token_budget=body.token_budget,
                sleeper=sleeper,
            )
        else:
            from runner.http_session import VllmHttpSession

            session = VllmHttpSession.open(recording, body)
        if getattr(session, "resolved_configuration", None) is not None:
            with state.lock:
                state.configuration = copy_configuration(session.resolved_configuration)
        _set_phase(state, "capturing", "capture_ready")

        def publish_arrival(arrival: dict[str, object]) -> None:
            with state.lock:
                state.arrivals.append(dict(arrival))

        _set_phase(state, "capturing", "dispatching_workload")
        completion, tokens, arrivals = drive_run(
            session,
            recording,
            requests,
            clock=clock,
            clock_ns=clock_ns,
            sleeper=sleeper,
            intake_open=lambda: _flag(state, "intake_open"),
            cancelled=lambda: _flag(state, "cancelled"),
            pull_extra=lambda: _take_extra(state),
            finish_intake=lambda: _finish_intake(state),
            on_status=lambda text: _set_text(state, text),
            on_arrival=publish_arrival,
            origin_out=origin,
            notes=notes,
            prompted=prompted,
        )
        with state.lock:
            state.origin = dict(origin)
            state.arrivals = [dict(item) for item in arrivals]
            if notes.get("message"):
                state.message = notes["message"]
    except Exception as exc:  # noqa: BLE001 - partial evidence must still be saved
        completion = "failed"
        with state.lock:
            state.message = _recording_message(str(exc), body.huggingface_token)
            if origin:
                state.origin = dict(origin)
    finally:
        _set_phase(state, "finalizing", "finalizing_recording")
        with state.lock:
            state.intake_open = False
        if session is not None:
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must not block finalization
                cleanup_error = _recording_message(
                    f"Session cleanup failed: {exc}",
                    body.huggingface_token,
                )
        if recording is None:
            manifest_error = "Recording initialization failed."
        else:
            try:
                manifest = _manifest_for(
                    state,
                    run_id,
                    body,
                    session,
                    arrivals,
                    tokens,
                    origin,
                    completion,
                )
                recording.close(manifest)
            except Exception as exc:  # noqa: BLE001 - the run must still leave the active state
                manifest_error = _recording_message(
                    f"Manifest finalization failed: {exc}",
                    body.huggingface_token,
                )
        with state.lock:
            state.status = completion
            state.phase = completion
            state.phase_history.append(_phase_record(completion))
            state.intake_open = False
            state.finalized = recording is not None and not manifest_error
            if completion == "completed" and not cleanup_error and not manifest_error and not state.message:
                state.message = "Recording finalized."
            elif completion == "completed" and not cleanup_error and not manifest_error:
                state.message = state.message or "Recording finalized."
            extra_notes = " ".join(part for part in (cleanup_error, manifest_error) if part)
            if extra_notes:
                state.message = f"{state.message} {extra_notes}".strip()


def _manifest_for(
    state: RunState,
    run_id: str,
    body: RunRequest,
    session: Any,
    arrivals: list[dict[str, object]],
    tokens: dict[str, list[int]],
    origin: dict[str, object],
    completion: str,
) -> dict[str, Any]:
    kind = "fixture" if body.engine == "fixture" else "gpu"
    target = (
        LOCAL_FIXTURE_TARGET
        if kind == "fixture"
        else {
            "kind": "local",
            "engine_location": "this-machine",
            "client_platform": sys.platform,
            "runner_endpoint": "http://127.0.0.1:8765",
            "remote_bind": "127.0.0.1",
        }
    )
    geometry = None
    if session is not None:
        kind = getattr(session, "execution_kind", kind)
        target = getattr(session, "execution_target", None) or target
        geometry = getattr(session, "cache_geometry", None)
    manifest = manifest_skeleton(
        run_id,
        execution_kind=kind,
        execution_target=target,
        arrivals=arrivals,
        completion_status=completion,
    )
    manifest["engine_output_token_ids"] = tokens
    from runner.profiling import profile_record

    manifest["profiling"] = profile_record(getattr(body, "profiling_configuration", None))
    if manifest["profiling"]["enabled"]:
        manifest["profiling"].update(status="failed", reason="The requested diagnostic profiler did not return initialized capture metadata.")
    if geometry is not None:
        manifest["cache_geometry"] = geometry
    if session is not None:
        for source_name, manifest_key in (
            ("engine_record", "engine"),
            ("model_record", "model"),
            ("tokenizer_record", "tokenizer"),
            ("hardware_record", "hardware"),
            ("sampling_record", "sampling"),
            ("initial_cache_state", "initial_cache_state"),
            ("architecture_record", "architecture"),
            ("revision_resolution", "revision_resolution"),
            ("observability_record", "observability"),
            ("profiling_record", "profiling"),
        ):
            value = getattr(session, source_name, None)
            if value is not None:
                manifest[manifest_key] = value
    recorded_settings = getattr(session, "engine_settings", None) if session is not None else None
    if recorded_settings is not None:
        manifest["engine_settings"] = recorded_settings
    elif body.engine == "fixture":
        manifest["engine_settings"] = {
            "async_scheduling": False,
            "enable_prefix_caching": False,
            "prefix_caching_applied": False,
            "max_num_seqs": body.max_num_seqs,
            "max_num_scheduled_tokens": body.token_budget,
            "demo_fixture": True,
            "settings_note": "Demo fixture did not finish applying settings. Prefix caching was not simulated.",
        }
    else:
        manifest["engine_settings"] = {
            "async_scheduling": False,
            "enable_prefix_caching": body.prefix_caching,
            "max_num_seqs": body.max_num_seqs,
            "max_num_scheduled_tokens": body.token_budget,
        }
    with state.lock:
        if state.plan is not None:
            manifest["workload_plan"] = state.plan
        if state.configuration is not None:
            manifest["run_configuration"] = copy_configuration(state.configuration)
        phase_history = [dict(item) for item in state.phase_history]
        run_message = state.message.strip()
    if not phase_history or phase_history[-1].get("phase") != completion:
        phase_history.append(_phase_record(completion))
    transport = str(getattr(session, "dispatch_transport", "fixture_direct")) if session is not None else "unknown"
    http_transport = transport == "instrumented_loopback_http"
    http_observed = http_transport and any(bool(item.get("http_receipt_observed")) for item in arrivals)
    capture_transport = (
        "instrumented_http"
        if http_transport
        else "fixture_direct"
        if kind == "fixture"
        else "in_process_direct_engine"
    )
    manifest["capture"] = {
        "contract_version": 1,
        "transport": capture_transport,
        "implementation": transport,
        "http_boundary_observed": http_observed,
        "engine_admission_boundary": "scheduler.add_request",
        "phase_history": phase_history,
        "status": completion,
        "preparation_observations": getattr(session, "preparation_observations", None),
    }
    if run_message:
        manifest["capture"]["message"] = _recording_message(run_message, body.huggingface_token)
    hardware = manifest.get("hardware") or {}
    observed_gpu = kind != "fixture" and session is not None and bool(hardware.get("gpu"))
    if kind == "fixture":
        provenance = {
            "evidence_kind": "fixture",
            "real_gpu": False,
            "http_serving": False,
            "gpu_acceptance": "not_applicable",
            "claim": "Deterministic demo evidence; not a vLLM or GPU observation.",
        }
    elif observed_gpu:
        provenance = {
            "evidence_kind": "instrumented_gpu_run",
            "real_gpu": True,
            "http_serving": http_observed,
            "gpu_acceptance": "passed",
            "claim": (
                "Instrumented vLLM execution observed on GPU through the explorer loopback HTTP boundary."
                if http_observed
                else "Instrumented vLLM execution observed on GPU; explorer HTTP request and response boundaries were not observed."
            ),
        }
    else:
        provenance = {
            "evidence_kind": "instrumented_gpu_run",
            "real_gpu": False,
            "http_serving": False,
            "gpu_acceptance": "required",
            "claim": (
                "GPU capture did not begin; model or runtime preparation failed before observed execution."
                if completion == "failed"
                else "GPU capture ended before any GPU execution was observed."
            ),
        }
    manifest["provenance"] = provenance
    if state.execution_process is not None:
        manifest["execution_process"] = state.execution_process
    if origin.get("controller_started_monotonic_ns") is not None:
        manifest["controller_origin"] = origin
    if body.engine == "fixture":
        manifest["demo_fixture"] = True
        manifest["lesson"] = {
            "title": "Demo fixture",
            "summary": (
                "Authored arrivals ran on the local demo fixture. "
                "Teaching pace is synthetic and is not a GPU measurement."
            ),
            "evidence": "fixture",
        }
    return manifest


def _recording_message(message: str, huggingface_token: str | None) -> str:
    if huggingface_token:
        message = message.replace(huggingface_token, "[Hugging Face token]")
    return message[:4000]


def _flag(state: RunState, name: str) -> bool:
    with state.lock:
        return bool(getattr(state, name))


def _take_extra(state: RunState) -> list[WorkRequest]:
    with state.lock:
        extra = list(state.extra)
        state.extra[:] = []
        for item in extra:
            state.text.setdefault(item.request_id, "")
            if state.plan is not None:
                state.plan = append_request_to_plan(state.plan, item)
        return extra


def _finish_intake(state: RunState) -> bool:
    with state.lock:
        if state.extra:
            return False
        state.intake_open = False
        return True


def _set_text(state: RunState, text: dict[str, str]) -> None:
    with state.lock:
        state.text.update(text)


def _phase_record(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "observed_monotonic_ns": time.monotonic_ns(),
        "clock_domain": "controller:time.monotonic_ns",
    }


def _set_phase(state: RunState, status: str, phase: str) -> None:
    with state.lock:
        state.status = status
        state.phase = phase
        if not state.phase_history or state.phase_history[-1].get("phase") != phase:
            state.phase_history.append(_phase_record(phase))


def _set_target_status(
    lock: threading.Lock,
    target_status: dict[str, Any],
    *,
    status: str,
    phase: str,
    message: str,
    execution_target: Any = _UNCHANGED,
) -> None:
    observation = {
        "phase": phase,
        "status": status,
        "observed_monotonic_ns": time.monotonic_ns(),
        "clock_domain": "controller:time.monotonic_ns",
    }
    with lock:
        target_status["status"] = status
        target_status["phase"] = phase
        target_status["message"] = message
        if execution_target is not _UNCHANGED:
            target_status["execution_target"] = execution_target
        target_status["history"].append(observation)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _vllm_version() -> str | None:
    try:
        import vllm
    except ImportError:
        return None
    return str(getattr(vllm, "__version__", ""))


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    config = parse_serve_args(sys.argv[1:] if argv is None else argv)
    app = create_app(Path(config.recordings))
    uvicorn.run(app, host=config.bind, port=config.port)


if __name__ == "__main__":
    main()
