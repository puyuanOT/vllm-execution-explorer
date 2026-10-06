"""Loopback HTTP execution around the instrumented in-process vLLM session.

HTTP handlers never touch vLLM. They enqueue accepted requests for one owner
thread, which constructs, admits, steps, cancels, and closes the engine. Equal
offset requests are submitted by concurrent client workers. The endpoint is an
explorer transport, not a claim that OpenAI-compatible vLLM server internals
have been instrumented.
"""

from __future__ import annotations

import http.client
import json
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from runner.fixture_engine import ObservedChunk
from runner.vllm_session import VllmSession
from runner.workload import PROVISIONAL_LIMITS, WorkRequest, WorkloadError, parse_request_item

HTTP_PATH = "/v1/explorer/generate"


@dataclass
class _Ticket:
    request: WorkRequest
    dispatch_started_ns: int | None = None
    dispatch_sent_ns: int | None = None
    http_received_ns: int | None = None
    engine_admitted_ns: int | None = None
    http_responded_ns: int | None = None
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    resolved_input_tokens: int | None = None
    adapter_id: str = "base"
    lora_int_id: int | None = None
    dispatch_event_sequence: int | None = None
    dispatch_barrier: threading.Barrier | None = None
    sent: threading.Event = field(default_factory=threading.Event)
    received: threading.Event = field(default_factory=threading.Event)
    admitted: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)
    response_recorded: threading.Event = field(default_factory=threading.Event)


class VllmHttpSession:
    execution_kind = "gpu"
    dispatch_transport = "instrumented_loopback_http"

    def __init__(self, recording: Any, body: Any) -> None:
        self.recording = recording
        self.body = body
        self._admissions: queue.Queue[_Ticket] = queue.Queue()
        self._chunks: queue.Queue[ObservedChunk] = queue.Queue()
        self._tickets: dict[str, _Ticket] = {}
        self._tickets_lock = threading.Lock()
        self._owner_ready = threading.Event()
        self._owner_error: BaseException | None = None
        self._base: VllmSession | None = None
        self._trace: Any = None
        self._experiment_status = "completed"
        self.observability_record: dict[str, Any] = {"provider": "langsmith", "status": "not_started", "verification": "No experiment trace has been inspected."}
        self._stop = threading.Event()
        self._cancelled = threading.Event()
        self._clients = ThreadPoolExecutor(max_workers=int(PROVISIONAL_LIMITS["max_requests"]))
        self._futures: list[Future[Any]] = []
        self._owner = threading.Thread(target=self._own_engine, name="vllm-engine-owner", daemon=True)
        self._owner.start()
        if not self._owner_ready.wait(timeout=900):
            raise RuntimeError("the vLLM engine did not finish preparing in time")
        if self._owner_error is not None:
            self._clients.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(str(self._owner_error)) from self._owner_error

        session = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
                session._handle_post(self)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        host, port = self._server.server_address
        self.endpoint = f"http://{host}:{port}{HTTP_PATH}"
        self._http = threading.Thread(target=self._server.serve_forever, name="vllm-http-boundary", daemon=True)
        self._http.start()
        self._copy_records()

    @classmethod
    def open(cls, recording: Any, body: Any) -> VllmHttpSession:
        return cls(recording, body)

    def add_work_requests(self, requests: list[WorkRequest]) -> dict[str, dict[str, Any]]:
        self._raise_owner_error()
        tickets: list[_Ticket] = []
        barrier = threading.Barrier(len(requests)) if len(requests) > 1 else None
        for request in requests:
            ticket = _Ticket(request=request, dispatch_barrier=barrier)
            with self._tickets_lock:
                if request.request_id in self._tickets:
                    raise RuntimeError(f"duplicate HTTP request id {request.request_id}")
                self._tickets[request.request_id] = ticket
            tickets.append(ticket)
            self._futures.append(self._clients.submit(self._post_request, ticket))
        observations: dict[str, dict[str, Any]] = {}
        for ticket in tickets:
            if not ticket.sent.wait(timeout=30):
                raise RuntimeError(f"HTTP dispatch was not observed for {ticket.request.request_id}")
            if not ticket.received.wait(timeout=30):
                raise RuntimeError(f"HTTP receipt was not observed for {ticket.request.request_id}")
            if not ticket.admitted.wait(timeout=120):
                raise RuntimeError(f"engine admission was not observed for {ticket.request.request_id}")
            if ticket.error:
                raise RuntimeError(ticket.error)
            self._raise_owner_error()
            observations[ticket.request.request_id] = {
                "dispatch_started_monotonic_ns": ticket.dispatch_started_ns,
                "dispatch_completed_monotonic_ns": ticket.dispatch_sent_ns,
                "dispatch_clock_domain": "workload_driver:time.monotonic_ns",
                "http_receipt_observed": True,
                "http_received_monotonic_ns": ticket.http_received_ns,
                "http_receipt_clock_domain": "http_server:time.monotonic_ns",
                "engine_admission_observed": True,
                "engine_admitted_monotonic_ns": ticket.engine_admitted_ns,
                "engine_admission_clock_domain": "engine_owner:time.monotonic_ns",
                "resolved_input_tokens": ticket.resolved_input_tokens,
                "adapter_id": ticket.adapter_id,
                "lora_int_id": ticket.lora_int_id,
                "dispatch_event_recorded": True,
            }
        return observations

    def add_work_request(self, request: WorkRequest) -> dict[str, Any]:
        return self.add_work_requests([request])[request.request_id]

    def step(self) -> list[ObservedChunk]:
        self._raise_owner_error()
        chunks: list[ObservedChunk] = []
        try:
            chunks.append(self._chunks.get(timeout=0.02))
        except queue.Empty:
            self._raise_owner_error()
            return chunks
        while True:
            try:
                chunks.append(self._chunks.get_nowait())
            except queue.Empty:
                self._raise_owner_error()
                return chunks

    def has_unfinished(self) -> bool:
        self._raise_owner_error()
        with self._tickets_lock:
            active = any(not ticket.completed.is_set() for ticket in self._tickets.values())
        return active or not self._admissions.empty() or not self._chunks.empty()

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        with self._tickets_lock:
            tickets = list(self._tickets.values())
        response_error = ""
        for ticket in tickets:
            if ticket.received.is_set() and not ticket.response_recorded.wait(timeout=10):
                response_error = f"HTTP response observation did not finalize for {ticket.request.request_id}"
                break
        self._server.shutdown()
        self._server.server_close()
        self._http.join(timeout=2)
        self._stop.set()
        self._owner.join(timeout=30)
        self._clients.shutdown(wait=True, cancel_futures=True)
        if self._owner.is_alive():
            raise RuntimeError("the vLLM engine owner did not stop")
        self._raise_owner_error()
        if response_error:
            raise RuntimeError(response_error)

    def _own_engine(self) -> None:
        try:
            self._base = VllmSession.open(self.recording, self.body)
            self._copy_records()
            from runner.observability import ExperimentTrace

            self._trace = ExperimentTrace(self.recording.directory, str(getattr(self.body, "model", "unreported")), getattr(self._base, "resolved_configuration", None) or {})
            self.observability_record = self._trace.metadata()
        except BaseException as exc:  # noqa: BLE001 - returned to the preparing caller
            self._owner_error = exc
            self._owner_ready.set()
            return
        self._owner_ready.set()
        try:
            while not self._stop.is_set():
                if self._cancelled.is_set():
                    self._experiment_status = "cancelled"
                    self._base.cancel()
                    self._finish_open_tickets("cancelled")
                    self._cancelled.clear()
                admitted = False
                while True:
                    try:
                        ticket = self._admissions.get_nowait()
                    except queue.Empty:
                        break
                    admitted = True
                    try:
                        resolution = self._base.add_work_request(ticket.request)
                        self._trace.start_request(ticket.request.request_id, ticket.request.prompt,
                                                  resolution.get("input_token_ids") if isinstance(resolution, dict) else None,
                                                  request_configuration=resolution.get("request_configuration") if isinstance(resolution, dict) else None)
                        if isinstance(resolution, dict):
                            value = resolution.get("resolved_input_tokens")
                            ticket.resolved_input_tokens = None if value is None else int(value)
                            ticket.adapter_id = str(resolution.get("adapter_id") or "base")
                            lora_int_id = resolution.get("lora_int_id")
                            ticket.lora_int_id = None if lora_int_id is None else int(lora_int_id)
                        ticket.engine_admitted_ns = time.monotonic_ns()
                        self.recording.event(
                            "http_engine_admission_correlated",
                            request_id=ticket.request.request_id,
                            http_received_monotonic_ns=ticket.http_received_ns,
                            engine_admitted_monotonic_ns=ticket.engine_admitted_ns,
                            resolved_input_tokens=ticket.resolved_input_tokens,
                            input_token_count_provenance="execution_host_tokenizer",
                            adapter_id=ticket.adapter_id,
                            lora_int_id=ticket.lora_int_id,
                            clock_domain="host:time.monotonic_ns",
                        )
                    except Exception as exc:  # noqa: BLE001 - handler returns the exact failure
                        self._experiment_status = "failed"
                        ticket.error = str(exc)
                        self._trace.start_request(ticket.request.request_id, ticket.request.prompt)
                        self._trace.finish_request(ticket.request.request_id, "", None, self._trace_error(ticket.error))
                        ticket.completed.set()
                    finally:
                        ticket.admitted.set()
                if self._base.has_unfinished():
                    for chunk in self._base.step():
                        self._observe_chunk(chunk)
                    continue
                if not admitted:
                    self._stop.wait(0.005)
        except BaseException as exc:  # noqa: BLE001 - propagate engine-owner failures to the run
            self._owner_error = exc
            self._experiment_status = "failed"
            self._finish_open_tickets("failed", error=str(exc))
        finally:
            self._finish_open_tickets("cancelled")
            try:
                self._base.close()
                self._copy_records()
            except BaseException as exc:  # noqa: BLE001 - close failures are part of capture status
                if self._owner_error is None:
                    self._owner_error = exc
                    self._experiment_status = "failed"
            if self._trace is not None:
                self._trace.finish(self._experiment_status, self._trace_error(str(self._owner_error)) if self._owner_error else None)
                self.observability_record = self._trace.metadata()

    def _trace_error(self, value: str) -> str:
        token = getattr(self.body, "huggingface_token", None)
        return value.replace(str(token), "[credential omitted]") if token else value

    def _observe_chunk(self, chunk: ObservedChunk) -> None:
        with self._tickets_lock:
            ticket = self._tickets.get(chunk.request_id)
        if ticket is not None:
            ticket.token_ids.extend(chunk.new_token_ids)
            ticket.text += chunk.text_delta
            if chunk.finish_reason is not None:
                ticket.finish_reason = chunk.finish_reason
                ticket.completed.set()
                if self._trace is not None:
                    self._trace.finish_request(chunk.request_id, ticket.text, ticket.token_ids)
        self._chunks.put(chunk)

    def _finish_open_tickets(self, reason: str, *, error: str | None = None) -> None:
        with self._tickets_lock:
            tickets = list(self._tickets.values())
        for ticket in tickets:
            if not ticket.completed.is_set():
                ticket.finish_reason = reason
                if error:
                    ticket.error = error
                ticket.completed.set()
                if self._trace is not None:
                    self._trace.finish_request(ticket.request.request_id, ticket.text, ticket.token_ids,
                                               self._trace_error(error or reason))
            ticket.sent.set()
            ticket.admitted.set()
            ticket.received.set()

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != HTTP_PATH:
            handler.send_error(404)
            return
        try:
            size = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            handler.send_error(400, "invalid content length")
            return
        if size < 2 or size > int(PROVISIONAL_LIMITS["max_prompt_bytes"]) + 65_536:
            handler.send_error(413, "request payload is too large")
            return
        try:
            payload = json.loads(handler.rfile.read(size))
            received_request = parse_request_item(payload, field_prefix="http_request")
            request_id = received_request.request_id
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, WorkloadError):
            handler.send_error(400, "invalid request payload")
            return
        with self._tickets_lock:
            ticket = self._tickets.get(request_id)
        if ticket is None:
            handler.send_error(409, "request was not registered by the workload driver")
            return
        ticket.request = received_request
        ticket.http_received_ns = time.monotonic_ns()
        self.recording.event(
            "http_request_received",
            request_id=request_id,
            http_received_monotonic_ns=ticket.http_received_ns,
            clock_domain="http_server:time.monotonic_ns",
            method="POST",
            path=HTTP_PATH,
        )
        ticket.received.set()
        self._admissions.put(ticket)
        if not ticket.completed.wait(timeout=float(PROVISIONAL_LIMITS["max_run_seconds"]) + 30):
            ticket.error = "generation did not finish before the HTTP timeout"
        status = 500 if ticket.error else 200
        body = json.dumps(
            {
                "request_id": request_id,
                "text": ticket.text,
                "token_ids": ticket.token_ids,
                "finish_reason": ticket.finish_reason,
                "error": ticket.error,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        try:
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        ticket.http_responded_ns = time.monotonic_ns()
        try:
            self.recording.event(
                "http_response_completed",
                request_id=request_id,
                status_code=status,
                http_responded_monotonic_ns=ticket.http_responded_ns,
                finish_reason=ticket.finish_reason,
                clock_domain="http_server:time.monotonic_ns",
            )
        finally:
            ticket.response_recorded.set()

    def _post_request(self, ticket: _Ticket) -> None:
        endpoint = urlsplit(self.endpoint)
        payload = json.dumps(
            {
                "request_id": ticket.request.request_id,
                "prompt": ticket.request.prompt,
                "max_output_tokens": ticket.request.max_output_tokens,
                "arrival_offset_ms": ticket.request.arrival_offset_ms,
                "input_mode": ticket.request.input_mode,
                "target_input_tokens": ticket.request.target_input_tokens,
                "resolved_input_tokens": ticket.request.resolved_input_tokens,
                "adapter_id": ticket.request.adapter_id or "base",
                "sampling": ticket.request.sampling,
                "priority": ticket.request.priority,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=float(PROVISIONAL_LIMITS["max_run_seconds"]) + 60)
        try:
            if ticket.dispatch_barrier is not None:
                ticket.dispatch_barrier.wait(timeout=10)
            ticket.dispatch_started_ns = time.monotonic_ns()
            ticket.dispatch_event_sequence = self.recording.event(
                "request_dispatched",
                request_id=ticket.request.request_id,
                planned_dispatch_offset_ns=ticket.request.arrival_offset_ms * 1_000_000,
                dispatch_started_monotonic_ns=ticket.dispatch_started_ns,
                dispatch_clock_domain="workload_driver:time.monotonic_ns",
                dispatch_transport=self.dispatch_transport,
                phase="started",
            )
            connection.request(
                "POST",
                endpoint.path,
                body=payload,
                headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
            )
            ticket.dispatch_sent_ns = time.monotonic_ns()
            self.recording.event(
                "request_dispatch_completed",
                request_id=ticket.request.request_id,
                dispatch_completed_monotonic_ns=ticket.dispatch_sent_ns,
                dispatch_clock_domain="workload_driver:time.monotonic_ns",
                causal_event_sequence=ticket.dispatch_event_sequence,
            )
            ticket.sent.set()
            response = connection.getresponse()
            response.read()
            if response.status != 200 and ticket.error is None:
                ticket.error = f"HTTP generation failed with status {response.status}"
        except Exception as exc:  # noqa: BLE001 - surfaced through the capture state
            ticket.error = str(exc)
            if ticket.dispatch_started_ns is None:
                ticket.dispatch_started_ns = time.monotonic_ns()
            if ticket.dispatch_sent_ns is None:
                ticket.dispatch_sent_ns = time.monotonic_ns()
            ticket.sent.set()
            ticket.received.set()
            ticket.admitted.set()
            ticket.completed.set()
            ticket.response_recorded.set()
        finally:
            connection.close()

    def _raise_owner_error(self) -> None:
        if self._owner_error is not None:
            raise RuntimeError(f"vLLM engine owner failed: {self._owner_error}") from self._owner_error

    def _copy_records(self) -> None:
        base = self._base
        if base is None:
            return
        for name in (
            "execution_target",
            "cache_geometry",
            "engine_settings",
            "engine_record",
            "model_record",
            "tokenizer_record",
            "hardware_record",
            "sampling_record",
            "initial_cache_state",
            "preparation_observations",
            "architecture_record",
            "revision_resolution",
            "resolved_configuration",
            "profiling_record",
            "kv_cache_record",
        ):
            setattr(self, name, getattr(base, name, None))
