"""Authored cache shapes exercise capture contracts; these are not GPU evidence."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from runner.configuration import ConfigurationError, parse_run_configuration
from runner.profile_trace import normalize_trace
from runner.profiling import DiagnosticProfile, profile_record
from runner.workload import parse_run_workload
from vllm_030.cache_tensors import CacheWriteCapture, cache_quantization, loaded_cache_tensors


class Values:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def clone(self):
        return Values(self.values)

    def reshape(self, *_shape):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def __getitem__(self, index):
        return Values(self.values[index]) if isinstance(index, slice) else Values([self.values[index]])

    def __floordiv__(self, divisor):
        return Values([value // divisor for value in self.values])

    def __mod__(self, divisor):
        return Values([value % divisor for value in self.values])

    def clamp(self, *, min, max):
        return Values([min if value < min else max if value > max else value for value in self.values])


class Cache:
    shape = (4, 2, 2, 8)
    dtype = "torch.float16"
    device = "cuda:0"

    def __init__(self, pointer=10):
        self.pointer = pointer
        self.data = [[[[float(block * 100 + head * 20 + slot * 8 + dimension) for dimension in range(8)]
                       for slot in range(2)] for head in range(2)] for block in range(4)]

    def numel(self):
        return 128

    def element_size(self):
        return 2

    def stride(self):
        return (32, 16, 8, 1)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.pointer, nbytes=lambda: 256)

    def __getitem__(self, index):
        block, head, slot, dimensions = index
        return Values(self.data[block.values[0]][head][slot.values[0]][dimensions])


def test_loaded_metadata_counts_backing_storage_once_and_keeps_layer_views(monkeypatch):
    FlashAttentionImpl = type("FlashAttentionImpl", (), {"head_size": 4, "do_kv_cache_update": lambda *_args: None})
    cache = Cache()
    modules = [(f"model.layers.{index}.self_attn.attn", SimpleNamespace(layer_name=f"model.layers.{index}.self_attn.attn", kv_cache=cache, impl=FlashAttentionImpl())) for index in range(2)]
    runner = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: modules), _kernel_block_sizes=[2])
    inventory = {"groups": [{"cache_group": 0, "pool": "pool-0", "block_size": 4, "token_slot_mapping": "qualified_full_attention", "layer_names": [name for name, _module in modules]}]}
    monkeypatch.setattr("vllm_030.cache_tensors.inspect.getsource", lambda *_args: "kv_cache.transpose(1, 2).split(self.head_size, dim=-1) reshape_and_cache_flash( slot_mapping,")
    metadata, records = loaded_cache_tensors(runner, inventory)
    assert metadata["unique_storage_bytes"] == 256 and len(metadata["storages"]) == 1
    assert [layer["storage_id"] for layer in metadata["layers"]] == ["storage-0", "storage-0"]
    assert all(layer["logical_bytes"] == 256 and layer["kernel_blocks_per_manager_block"] == 2 for layer in metadata["layers"])
    assert metadata["layers"][0]["device"] == "cuda:0" and len(records) == 2
    cache.dtype = "torch.uint8"
    metadata, _records = loaded_cache_tensors(runner, inventory)
    assert all(layer["layout_status"] == "unavailable" for layer in metadata["layers"])
    assert "BF16/FP16" in metadata["layers"][0]["layout_reason"]


def test_quantization_records_effective_mode_and_scale_presence_without_reading_values():
    scale = SimpleNamespace(shape=(8,), dtype="torch.float32", device="cuda:0")
    module = SimpleNamespace(_k_scale=scale)
    metadata = cache_quantization(module, SimpleNamespace(kv_cache_dtype="fp8_e4m3"), SimpleNamespace(kv_quant_mode=SimpleNamespace(name="PER_TENSOR")))
    assert metadata["cache_dtype"] == "fp8_e4m3" and metadata["mode"] == "PER_TENSOR"
    assert metadata["scales"]["K"] == {"status": "observed", "shape": [8], "dtype": "float32", "device": "cuda:0"}
    assert metadata["scales"]["V"]["status"] == "unavailable"


def capture_fixture(max_bytes=64, slot_address=6, request_id=None):
    cache = Cache()
    record = {"layer_index": 0, "pool": "pool-0", "cache_group": 0, "device": "cuda:0", "dtype": "float16", "layout_status": "qualified",
              "kernel_block_size": 2, "manager_block_size": 4, "kernel_blocks_per_manager_block": 2, "num_kernel_blocks": 4, "num_kv_heads": 2, "head_dim": 4}
    contexts = []
    @contextmanager
    def record_function(name):
        contexts.append(name)
        yield
    class Table:
        def __getitem__(self, index):
            return 3
    impl = SimpleNamespace(do_kv_cache_update=lambda *_args: None)
    hooks = SimpleNamespace(current_step="step-3", _current_mapping={"mapping_status": "observed", "packed_rows": [{"row": 0, "request_id": "a", "token_position": 2, "is_padding": False}]},
                            record_function=record_function, runner=SimpleNamespace(input_batch=SimpleNamespace(req_id_to_index={"a": 0}, block_table=[SimpleNamespace(block_table=SimpleNamespace(np=Table()))])),
                            _patch=lambda target, name, replacement: setattr(target, name, replacement), source_fingerprints={})
    records = {"model.layers.0.attn": {"record": record, "module": SimpleNamespace(impl=impl)}}
    capture = CacheWriteCapture(hooks, records, {"enabled": True, "request_id": request_id, "head_indices": [0], "dimension_start": 1, "dimension_count": 2, "max_bytes": max_bytes}, lambda _pool, _block: 5)
    capture.install()
    impl.do_kv_cache_update(None, None, None, cache, Values([slot_address]))
    return capture, cache, contexts


def test_snapshot_owns_contents_before_reuse_and_maps_manager_vs_kernel_address():
    capture, cache, contexts = capture_fixture()
    cache.data[3][0][0] = [-1.0] * 8
    capture.finalize()
    assert contexts == ["explorer.cache_write/0/cache-write-0"]
    participant = capture.writes[0]["participants"][0]
    assert (participant["block_id"], participant["slot"], participant["kernel_block_id"], participant["kernel_slot"], participant["allocation_generation"]) == (1, 2, 3, 0, 5)
    assert [item["values"] for item in capture.snapshots] == [[301.0, 302.0], [305.0, 306.0]]
    assert all(item["block_id"] == 1 and item["slot"] == 2 and item["shape"] == [2] for item in capture.snapshots)
    assert capture.capture_record()["retained_bytes"] == 8


def test_snapshot_byte_cap_and_unobserved_selection_remain_explicit():
    capture, _cache, _contexts = capture_fixture(max_bytes=4)
    capture.finalize()
    assert len(capture.snapshots) == 1
    assert capture.capture_record()["status"] == "partial" and "byte limit" in capture.capture_record()["reason"]
    capture, _cache, _contexts = capture_fixture(request_id="not-in-window")
    capture.finalize()
    assert capture.capture_record()["status"] == "unavailable" and "targeted rerun" in capture.capture_record()["reason"]


@pytest.mark.parametrize("address", [-1, 99])
def test_skipped_or_invalid_gpu_slots_cannot_produce_snapshots_or_write_participation(address):
    capture, _cache, _contexts = capture_fixture(slot_address=address)
    capture.finalize()
    assert not capture.snapshots and not capture.writes[0]["participants"]
    assert capture.capture_record()["retained_bytes"] == 0 and capture.capture_record()["status"] == "unavailable"


def test_cache_write_native_correlation_does_not_include_snapshot_kernels():
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "explorer.step/step-3", "ts": 0, "dur": 40, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "user_annotation", "name": "explorer.cache_write/0/cache-write-0", "ts": 5, "dur": 10, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "cuda_runtime", "name": "launch", "ts": 8, "dur": 1, "pid": 1, "tid": 1, "args": {"correlation": 10}},
        {"ph": "X", "cat": "cuda_runtime", "name": "snapshot launch", "ts": 20, "dur": 1, "pid": 1, "tid": 1, "args": {"correlation": 20}},
        {"ph": "X", "cat": "kernel", "name": "cache native", "ts": 80, "dur": 5, "args": {"correlation": 10}},
        {"ph": "X", "cat": "kernel", "name": "snapshot native", "ts": 90, "dur": 5, "args": {"correlation": 20}},
    ]
    activities = normalize_trace({"traceEvents": events}, run_id="authored", steps=[])["activities"]
    assert activities[-2]["scope"]["operation_id"] == "cache-write-0"
    assert "operation_id" not in activities[-1]["scope"]


def test_later_window_requires_both_start_step_and_request_arrival():
    profile = DiagnosticProfile.__new__(DiagnosticProfile)
    profile.record = profile_record({"mode": "eager_diagnostic", "start_step": 3, "trigger_request_id": "later"})
    profile._step_count, profile._started, profile._finished, profile._arrived_requests = 0, False, False, set()
    starts = []
    def start():
        starts.append(profile._step_count)
        profile._started = True
    profile._start_native = start
    profile.start()
    profile.before_step()
    profile.before_step()
    profile.before_step()
    assert not starts
    profile.note_request_arrival("later")
    profile.before_step()
    profile.before_step()
    assert starts == [4]


@pytest.mark.parametrize("change,field", [
    ({"start_step": 0}, "start_step"), ({"trigger_request_id": "missing"}, "trigger_request_id"),
    ({"tensor_capture": {"enabled": True, "max_bytes": 1}}, "max_bytes"),
    ({"tensor_capture": {"enabled": True, "head_indices": [False]}}, "head_indices"),
    ({"mode": "off", "tensor_capture": {"enabled": True}}, "enabled"),
])
def test_diagnostic_controls_reject_invalid_windows_and_tensor_bounds(change, field):
    payload = {"requests": [{"request_id": "a", "prompt": "hello", "arrival_offset_ms": 0, "max_output_tokens": 1}],
               "profiling_configuration": {"mode": "eager_diagnostic", **change}}
    requests, plan = parse_run_workload(payload)
    with pytest.raises(ConfigurationError) as error:
        parse_run_configuration(payload, requests, plan)
    assert field in error.value.field
