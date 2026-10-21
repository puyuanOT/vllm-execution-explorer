"""Measured memory accounting; device, process and allocator views overlap.

Sampling does not synchronize CUDA. Device maxima are sampled maxima, while
PyTorch peaks are allocator counters since initialization or its last reset.
Pointers are used only to deduplicate local storages and never leave the process.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import functools
import io
import os
import subprocess
import sys
import threading
import time
import uuid as uuid_library
from typing import Any

CLOCK_DOMAIN = "time.monotonic_ns"
SAMPLER_SHUTDOWN_TIMEOUT_SECONDS = 5


class NvmlMemorySource:
    name = "nvml"

    def __init__(self) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()

    def sample(self, pid: int) -> list[dict[str, Any]]:
        nvml, devices = self.nvml, []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = nvml.nvmlDeviceGetUUID(handle)
            name = nvml.nvmlDeviceGetName(handle)
            row = {"gpu_uuid": _text(uuid), "physical_index": index, "name": _text(name),
                   "total_bytes": int(memory.total), "used_bytes": int(memory.used), "free_bytes": int(memory.free),
                   "process_status": "unavailable", "process_used_bytes": None}
            try:
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                matching = [item for item in processes if item.pid == pid]
                values = [item.usedGpuMemory for item in matching]
                if any(value is None or value == getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", -1) for value in values):
                    row["process_status"] = "unavailable"
                else:
                    row.update(process_status="observed", process_used_bytes=sum(int(value) for value in values))
            except Exception as error:  # An unsupported per-process query does not erase device counters.
                row["process_reason"] = type(error).__name__
            devices.append(row)
        return devices


class NvidiaSmiMemorySource:
    name = "nvidia-smi"

    def sample(self, pid: int) -> list[dict[str, Any]]:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True)
        rows = []
        for fields in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            index, uuid, name, total, used, free = fields
            rows.append({"gpu_uuid": uuid, "physical_index": int(index), "name": name,
                         "total_bytes": _mib(total), "used_bytes": _mib(used), "free_bytes": _mib(free),
                         "process_status": "unavailable", "process_used_bytes": None})
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=2, check=True)
            process_rows = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
            for row in rows:
                values = [fields[2] for fields in process_rows if fields[0] == row["gpu_uuid"] and int(fields[1]) == pid]
                if all(value not in {"N/A", "[N/A]", "[Not Supported]"} for value in values):
                    row.update(process_status="observed", process_used_bytes=sum(_mib(value) for value in values))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return rows


def _mib(value: str) -> int:
    return round(float(value) * 1024 * 1024)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _torch_uuid(value: Any, devices: list[dict[str, Any]]) -> str:
    raw = str(uuid_library.UUID(bytes=value)) if isinstance(value, bytes) and len(value) == 16 else _text(value)
    # PyTorch exposes a bare UUID on some builds, NVML exposes GPU-<uuid>.
    # Resolve against observed management identities rather than assigning a
    # physical GPU namespace to an unobserved MIG instance.
    try:
        canonical = str(uuid_library.UUID(raw))
    except ValueError:
        return raw
    matches = [row["gpu_uuid"] for row in devices if row["gpu_uuid"] in {f"GPU-{canonical}", f"MIG-{canonical}"}]
    return matches[0] if len(matches) == 1 else raw


def _default_source() -> Any:
    try:
        return NvmlMemorySource()
    except Exception:
        return NvidiaSmiMemorySource()


def framework_memory(worker: Any, devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Read only the current process's already initialized CUDA allocator."""
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_initialized", None)) or not cuda.is_initialized():
        return None
    device = getattr(worker, "device", None)
    if device is None:
        return None
    props = cuda.get_device_properties(device)
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        uuid = _torch_uuid(uuid, devices)
    index = getattr(device, "index", None)
    if not isinstance(index, int) and str(device).startswith("cuda:"):
        index = int(str(device).split(":")[1])
    resolution = "torch_device_uuid" if uuid is not None else "unavailable"
    if uuid is None and isinstance(index, int):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = (visible.split(",")[index].strip() if visible and index < len(visible.split(",")) else str(index) if not visible else None)
        match = next((row for row in devices if physical == row["gpu_uuid"] or physical == str(row["physical_index"])), None)
        if match is not None:
            uuid, resolution = match["gpu_uuid"], "CUDA_VISIBLE_DEVICES" if visible else "default_cuda_order"
    return {"device": str(device), "gpu_uuid": None if uuid is None else _text(uuid), "uuid_resolution": resolution,
            "worker_pid": os.getpid(), "rank": getattr(worker, "rank", None), "local_rank": getattr(worker, "local_rank", None),
            "allocated_bytes": int(cuda.memory_allocated(device)), "reserved_bytes": int(cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
            "peak_scope": "allocator_since_initialization_or_last_external_reset", "source": "torch.cuda.memory"}


def loaded_storages(worker: Any) -> dict[str, Any]:
    """Count backing storage once, retaining shared ownership explicitly."""
    runner = getattr(worker, "model_runner", None)
    model = getattr(runner, "model", None)
    if model is None:
        return {"status": "unavailable", "reason": "Loaded worker model is not exposed.", "storages": []}
    allocations: dict[tuple[str, int], dict[str, Any]] = {}

    def add(tensor: Any, category: str, owner: str) -> None:
        if not callable(getattr(tensor, "untyped_storage", None)) or not callable(getattr(tensor, "numel", None)) or tensor.numel() == 0:
            return
        storage, device = tensor.untyped_storage(), str(tensor.device)
        identity = device, storage.data_ptr()
        row = allocations.setdefault(identity, {"storage_id": f"allocation-{len(allocations)}", "device": device,
                                               "byte_count": int(storage.nbytes()), "categories": [], "owners": [], "dtypes": []})
        if category not in row["categories"]:
            row["categories"].append(category)
        if owner not in row["owners"] and len(row["owners"]) < 8:
            row["owners"].append(owner)
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in row["dtypes"]:
            row["dtypes"].append(dtype)

    def add_container(value: Any, category: str, owner: str) -> None:
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                add_container(item, category, f"{owner}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                add_container(item, category, f"{owner}.{key}")
        else:
            add(value, category, owner)

    for method in ("named_parameters", "named_buffers"):
        named = getattr(model, method, None)
        if callable(named):
            try:
                tensors = named(remove_duplicate=False)
            except TypeError:
                tensors = named()
            for name, tensor in tensors:
                if "kv_cache" not in name:
                    add(tensor, "lora" if "lora_" in name.lower() else "weights", name)
    for name, module in model.named_modules():
        recurrent = "gateddeltanet" in type(module).__name__.lower() or "mamba" in type(module).__name__.lower()
        add_container(getattr(module, "kv_cache", None), "recurrent_state" if recurrent else "attention_kv", f"{name}.kv_cache")
        for attribute in ("lora_a_stacked", "lora_b_stacked", "lora_embedding_a", "lora_embedding_b"):
            add_container(getattr(module, attribute, None), "lora", f"{name}.{attribute}")
    for name, value in vars(runner).items():
        # Direct runner tensors are named persistent runtime buffers. Nested model,
        # graph and allocator objects are deliberately not guessed or traversed.
        add(value, "runtime_buffers", f"model_runner.{name}")
    storages = list(allocations.values())
    return {"status": "observed", "source": "loaded_tensor_backing_storages", "storages": storages,
            "unique_storage_bytes": sum(row["byte_count"] for row in storages), "worker_pid": os.getpid(),
            "rank": getattr(worker, "rank", None), "local_rank": getattr(worker, "local_rank", None),
            "coverage": "Registered model parameters/buffers, exposed attention/recurrent caches, LoRA stacks and direct runner tensor buffers; opaque workspaces and CUDA graph pools remain unattributed."}


class GpuMemoryRecorder:
    def __init__(self, recording: Any, *, source: Any = None, interval_ms: int = 100) -> None:
        self.recording, self.source = recording, source or _default_source()
        self.interval_ms, self.phase = interval_ms, "preparation"
        self.worker: Any = None
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._publication_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closing = self._closed = False
        self._final_sample_status: str | None = None
        self._thread: threading.Thread | None = None
        self._devices: list[dict[str, Any]] = []
        self.record = {"contract_version": 1, "status": "recording", "clock_domain": CLOCK_DOMAIN,
                       "sample_interval_ms": interval_ms, "source": self.source.name, "process_pid": os.getpid(),
                       "accounting": "Device, engine-process and framework counters are overlapping views; storage attribution counts unique backing allocations.",
                       "device_peak_kind": "highest_successful_sample", "allocator_peak_kind": "allocator_counter_since_initialization_or_last_external_reset"}

    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._loop, name="gpu-memory-recorder", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000):
            self.sample()
        self._final_sample_status = self._sample(final=True)

    def set_phase(self, phase: str, boundary: str) -> None:
        self.phase = phase
        self.recording.event("gpu_memory_phase", phase=phase, observation_boundary=boundary, memory_clock_domain=CLOCK_DOMAIN)

    def sample(self) -> None:
        self._sample()

    def _sample(self, *, final: bool = False) -> str | None:
        with self._publication_lock:
            if self._closed or (self._closing and not final):
                return None
        # Do not queue slow management queries behind one another during inference.
        if not self._sample_lock.acquire(blocking=False):
            return None
        started = time.monotonic_ns()
        try:
            try:
                self._devices = self.source.sample(os.getpid())
                framework = None
                try:
                    framework = framework_memory(self.worker, self._devices) if self.worker is not None else None
                except Exception:
                    pass
                fields = {"status": "observed", "devices": self._devices, "framework": framework}
            except Exception as error:
                fields = {"status": "unavailable", "reason": type(error).__name__}
            with self._publication_lock:
                if self._closed:
                    return None
                self.recording.event("gpu_memory_sample", source=self.source.name, phase=self.phase,
                                     sample_started_monotonic_ns=started, sample_completed_monotonic_ns=time.monotonic_ns(),
                                     memory_clock_domain=CLOCK_DOMAIN, engine_pid=os.getpid(), **fields)
                return fields["status"]
        finally:
            self._sample_lock.release()

    def inventory(self, worker: Any, boundary: str) -> dict[str, Any]:
        self.worker = worker
        try:
            inventory = loaded_storages(worker)
        except Exception as error:
            inventory = {"status": "unavailable", "reason": type(error).__name__, "storages": []}
        self.recording.event("gpu_memory_inventory", phase=self.phase, observation_boundary=boundary,
                             memory_clock_domain=CLOCK_DOMAIN, **inventory)
        self.sample()
        return inventory

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._publication_lock:
                self._closing = True
            self._stop.set()
            self.worker = None
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="gpu-memory-recorder", daemon=True)
                self._thread.start()
            self._thread.join(timeout=SAMPLER_SHUTDOWN_TIMEOUT_SECONDS)
            with self._publication_lock:
                # A timed-out native query may return later. Retire publication
                # before the caller is allowed to close the recording collector.
                self._closed = True
                timed_out = self._thread.is_alive()
                self.record.update(status="partial" if timed_out or self._final_sample_status is None else "completed",
                                   sampler_status="shutdown_timeout" if timed_out else "stopped",
                                   final_sample_status=self._final_sample_status or "not_recorded")
                if timed_out:
                    self.record.update(shutdown_timeout_seconds=SAMPLER_SHUTDOWN_TIMEOUT_SECONDS,
                                       reason="The GPU management query did not finish before the sampler shutdown deadline; late results are discarded and final memory accounting is unavailable.")
                elif self._final_sample_status is None:
                    self.record["reason"] = "The final GPU memory sample could not be recorded; late results are discarded."

    @contextmanager
    def worker_boundaries(self):
        """Instrument exact native initialization boundaries for this leased engine."""
        try:
            from vllm.v1.worker.gpu_worker import Worker
        except ImportError:
            self.record["initialization_boundaries"] = "unavailable"
            yield
            return
        originals = {}
        for name, phase in (("load_model", "model_load"), ("determine_available_memory", "memory_profiling"),
                            ("initialize_from_config", "cache_initialization"), ("compile_or_warm_up_model", "warmup")):
            original = getattr(Worker, name, None)
            if not callable(original):
                continue
            originals[name] = original

            def wrap(method: Any, native_name: str, native_phase: str):
                @functools.wraps(method)
                def wrapped(worker: Any, *args: Any, **kwargs: Any):
                    self.set_phase(native_phase, f"Worker.{native_name}.entry")
                    result = method(worker, *args, **kwargs)
                    self.inventory(worker, f"Worker.{native_name}.returned")
                    return result
                return wrapped

            setattr(Worker, name, wrap(original, name, phase))
        self.record["initialization_boundaries"] = list(originals)
        try:
            yield
        finally:
            for name, original in originals.items():
                setattr(Worker, name, original)


def observed_weight_storage(model_config: Any, inventory: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve format from native configuration and actual loaded storages."""
    compute = str(getattr(model_config, "dtype", "unknown")).removeprefix("torch.")
    quantization = getattr(model_config, "quantization", None)
    checkpoint_quantization = getattr(getattr(model_config, "hf_config", None), "quantization_config", None)
    if checkpoint_quantization is not None and not isinstance(checkpoint_quantization, (dict, list, str, int, float, bool)):
        convert = getattr(checkpoint_quantization, "to_dict", None) or getattr(checkpoint_quantization, "model_dump", None)
        checkpoint_quantization = convert() if callable(convert) else {"status": "unavailable", "type": type(checkpoint_quantization).__name__}
    storages = [row for row in (inventory or {}).get("storages", []) if "weights" in row.get("categories", [])]
    dtypes = sorted({dtype for row in storages for dtype in row["dtypes"]})
    format = None
    if quantization == "fp8" and any(dtype.startswith("float8") for dtype in dtypes):
        format = "fp8"
    elif not quantization and compute == "bfloat16" and "bfloat16" in dtypes:
        format = "bf16"
    elif not quantization and compute == "float16" and "float16" in dtypes:
        format = "fp16"
    elif not quantization and compute == "float32" and "float32" in dtypes:
        format = "fp32"
    return {"status": "observed" if format is not None else "unresolved", "format": format,
            "compute_dtype": compute, "native_quantization": quantization, "checkpoint_quantization": checkpoint_quantization,
            "tensor_dtypes": dtypes, "source": "native_model_configuration_and_loaded_tensor_storages",
            "reason": None if format is not None else "Loaded tensor types and native configuration do not qualify a specific weight storage format; packed bytes alone do not identify FP4."}
