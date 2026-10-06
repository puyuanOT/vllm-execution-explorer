import assert from "node:assert/strict"
import { test } from "node:test"
import { captureIndices } from "./diagnostics.ts"
import { defaultSettings, staggeredExample } from "./example.ts"
import { toRunBody } from "./serialize.ts"
import { validateScenario, validateSettings } from "./validate.ts"
import { tensorRerunSettings } from "./tensorRerun.ts"
import type { RecordingManifest } from "../../../replay/src/reduce.ts"

test("diagnostic capture window and tensor preset retain request identity", () => {
  const settings = { ...defaultSettings(), engine: "vllm" as const, profilingMode: "eager_diagnostic" as const,
    profilingStartStep: "12", profilingTriggerRequestId: "sample-3", tensorCapture: true, tensorRequestId: "sample-3" }
  assert.deepEqual(validateScenario(staggeredExample(), null, settings), [])
  const config = toRunBody(staggeredExample(), settings).profiling_configuration
  assert.equal(config.start_step, 12)
  assert.equal(config.trigger_request_id, "sample-3")
  assert.deepEqual(config.tensor_capture, { enabled: true, preset: "small", request_id: "sample-3", token_positions: [],
    layer_indices: [], head_indices: [0], dimension_start: 0, dimension_count: 16, max_bytes: 1048576 })
})

test("custom numerical targets serialize sorted unique zero-based indices", () => {
  const settings = { ...defaultSettings(), engine: "vllm" as const, profilingMode: "eager_diagnostic" as const,
    tensorCapture: true, tensorPreset: "custom" as const, tensorTokenPositions: "8, 2 8", tensorLayerIndices: "35,0",
    tensorHeadIndices: "2,0", tensorDimensionStart: "16", tensorDimensionCount: "8", tensorMaxBytes: "2048" }
  assert.deepEqual(validateSettings(settings), [])
  assert.deepEqual(toRunBody(staggeredExample(), settings).profiling_configuration.tensor_capture, {
    enabled: true, preset: "custom", request_id: null, token_positions: [2, 8], layer_indices: [0, 35], head_indices: [0, 2],
    dimension_start: 16, dimension_count: 8, max_bytes: 2048,
  })
  for (const value of ["-1", "1.5", "1e2", "2;3", "2000001"]) assert.equal(captureIndices(value, 64), null, value)
  assert.throws(() => toRunBody(staggeredExample(), { ...settings, tensorLayerIndices: "-1" }), /zero-based/)
})

test("targeted rerun restores retained inference configuration while keeping the operator connection", () => {
  const current = { ...defaultSettings(), hostKind: "ssh" as const, sshHost: "gpu.example.com", sshPort: "2222", runnerUrl: "http://127.0.0.1:8766" }
  const manifest = { run_configuration: {
    model: { source: { kind: "huggingface", locator: "Qwen/model", revision: "commit123" },
      tokenizer: { locator: "Qwen/tokenizer", revision: "token123" }, dtype: "bfloat16", quantization: { kind: "none" }, lora_adapters: [] },
    vllm: { max_model_len: 1024, max_num_seqs: 2, max_num_scheduled_tokens: 64, enable_prefix_caching: true,
      enable_chunked_prefill: true, gpu_memory_utilization: 0.8, kv_cache_memory_bytes: 268435456, scheduling_policy: "fcfs" },
    sampling: { seed: 42, temperature: 0, top_p: 1, stop: ["END"] },
  } } as RecordingManifest
  const settings = tensorRerunSettings(current, manifest, { requestId: "sample-3", tokenPosition: 7, layerIndex: 18, kvHead: 1, startStep: 12 })
  assert.equal(settings.sshHost, current.sshHost)
  assert.equal(settings.sshPort, current.sshPort)
  assert.equal(settings.runnerUrl, current.runnerUrl)
  assert.equal(settings.model, "Qwen/model")
  assert.equal(settings.modelRevision, "commit123")
  assert.equal(settings.tokenizer, "Qwen/tokenizer")
  assert.equal(settings.tokenizerRevision, "token123")
  assert.equal(settings.dtype, "bfloat16")
  assert.equal(settings.prefixCaching, true)
  assert.equal(settings.tokenBudget, "64")
  assert.deepEqual(settings.stopStrings, ["END"])
  const body = toRunBody(staggeredExample(), settings)
  assert.equal(body.profiling_configuration.start_step, 12)
  assert.deepEqual(body.profiling_configuration.tensor_capture?.token_positions, [7])
  assert.deepEqual(body.profiling_configuration.tensor_capture?.layer_indices, [18])
  assert.deepEqual(body.profiling_configuration.tensor_capture?.head_indices, [1])
  assert.equal(body.profiling_configuration.trigger_request_id, "sample-3")
  const automaticTokenizer = { ...manifest, run_configuration: {
    ...(manifest.run_configuration as Record<string, unknown>),
    model: { source: { kind: "huggingface", locator: "Qwen/model", revision: "commit123" }, tokenizer: { locator: "Qwen/model", revision: null } },
  } }
  const restored = tensorRerunSettings({ ...current, tokenizer: "Other/tokenizer", tokenizerRevision: "unrelated-old-commit" }, automaticTokenizer,
    { requestId: "sample-3", tokenPosition: 7, layerIndex: 18, kvHead: 1 })
  assert.equal(restored.tokenizer, "Qwen/model")
  assert.equal(restored.tokenizerRevision, "")
  assert.equal(toRunBody(staggeredExample(), restored).model_configuration.tokenizer.revision, null)
})

test("deleted request selections and invalid capture bounds fail before submission", () => {
  const settings = { ...defaultSettings(), engine: "vllm" as const, profilingMode: "eager_diagnostic" as const, tensorCapture: true,
    profilingTriggerRequestId: "deleted", tensorRequestId: "deleted", profilingStartStep: "0", tensorMaxBytes: "63" }
  const fields = validateScenario(staggeredExample(), null, settings).map((issue) => issue.fieldId)
  for (const field of ["setting-profiling-trigger", "setting-profiling-tensor-request", "setting-profiling-start", "setting-profiling-tensor-bytes"]) assert.ok(fields.includes(field), field)
  assert.deepEqual(toRunBody(staggeredExample(), defaultSettings()).profiling_configuration, { mode: "off", max_steps: 8 })
  const disabled = { ...settings, profilingStartStep: "1", profilingTriggerRequestId: "", tensorCapture: false,
    tensorPreset: "custom" as const, tensorDimensionStart: "invalid", tensorDimensionCount: "0", tensorLayerIndices: "-1" }
  assert.deepEqual(validateScenario(staggeredExample(), null, disabled), [])
  assert.deepEqual(toRunBody(staggeredExample(), disabled).profiling_configuration.tensor_capture,
    { enabled: false, preset: "small", request_id: null, token_positions: [], layer_indices: [], head_indices: [0], dimension_start: 0, dimension_count: 16, max_bytes: 1048576 })
})
