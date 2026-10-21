import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

import type { CatalogEntry } from "../api.ts"
import { DEFAULT_DEMO_SELECTION, demoAdapterScenario, demoAvailability, demoCaptureSettings, matchesDemo, recordingConfiguration } from "../shell/demoCatalog.ts"
import { defaultSettings, staggeredExample } from "./example.ts"
import { toRunBody } from "./serialize.ts"
import { validateScenario, validateSettings } from "./validate.ts"
import { tensorRerunSettings } from "./tensorRerun.ts"

const original = JSON.parse(readFileSync(new URL("../../public/recordings/gpu-kv-layer-diagnostics/manifest.json", import.meta.url), "utf8"))
const entry: CatalogEntry = { id: "gpu-kv-layer-diagnostics", title: "Recorded GPU lesson", summary: "", evidence: "gpu", configuration: recordingConfiguration(original) }

test("saved GPU manifests distinguish requested compute from observed storage and preserve legacy unknowns", () => {
  assert.equal(entry.configuration?.requestedComputeDtype, "bfloat16")
  assert.equal(entry.configuration?.weightFormat, "unknown")
  assert.equal(entry.configuration?.observedComputeDtype, null)
  assert.equal(entry.configuration?.kvFormat, "bf16")
  assert.equal(entry.configuration?.loraMode, "none")
  assert.equal(entry.configuration?.runtimeVersion, "0.30.0")
  assert.equal(matchesDemo(entry, DEFAULT_DEMO_SELECTION), true)
  assert.equal(matchesDemo(entry, { ...DEFAULT_DEMO_SELECTION, weightFormat: "bf16" }), false)
  assert.equal(matchesDemo(entry, { ...DEFAULT_DEMO_SELECTION, kvFormat: "fp8_e4m3" }), false)
  assert.equal(matchesDemo({ ...entry, evidence: "synthetic" }, DEFAULT_DEMO_SELECTION), false)
})

test("observed formats match independently without relabeling an artifact", () => {
  const manifest = structuredClone(original)
  manifest.model.weight_storage = { status: "observed", format: "fp8", compute_dtype: "bfloat16", source: "native_model_configuration_and_loaded_tensor_storages" }
  const observed = { ...entry, configuration: recordingConfiguration(manifest) }
  assert.equal(observed.configuration.weightFormat, "fp8")
  assert.equal(observed.configuration.observedComputeDtype, "bfloat16")
  assert.equal(matchesDemo(observed, { ...DEFAULT_DEMO_SELECTION, weightFormat: "fp8", kvFormat: "bf16" }), true)
  assert.equal(matchesDemo(observed, { ...DEFAULT_DEMO_SELECTION, weightFormat: "bf16", kvFormat: "bf16" }), false)
  assert.equal(entry.configuration?.weightFormat, "unknown")
})

test("FP8 KV scales are matched by captured encoding rather than an ambiguous raw uint8 dtype", () => {
  const manifest = structuredClone(original)
  for (const layer of manifest.kv_cache.layers) {
    layer.dtype = "uint8"
    layer.quantization.cache_dtype = "fp8_e5m2"
  }
  assert.equal(recordingConfiguration(manifest).kvFormat, "fp8_e5m2")
  manifest.kv_cache.layers[0].quantization.cache_dtype = "fp8_e4m3"
  assert.equal(recordingConfiguration(manifest).kvFormat, "unknown")
  delete manifest.kv_cache
  manifest.engine_settings.kv_cache_dtype = "fp8_e4m3"
  assert.equal(recordingConfiguration(manifest).kvFormat, "unknown")
})

test("native fp8 alias uses qualified recorded encoding while ambiguous bytes remain unknown", () => {
  const manifest = JSON.parse(readFileSync(new URL("../../public/recordings/gpu-h100-fp8-kv/manifest.json", import.meta.url), "utf8"))
  for (const layer of manifest.kv_cache.layers) layer.quantization.cache_dtype = "fp8"
  const alias = { ...entry, configuration: recordingConfiguration(manifest) }
  assert.equal(alias.configuration.kvFormat, "fp8_e4m3")
  assert.equal(matchesDemo(alias, { ...DEFAULT_DEMO_SELECTION, kvFormat: "fp8_e4m3" }), true)
  for (const layer of manifest.kv_cache.layers) layer.layout_status = "unavailable"
  assert.equal(recordingConfiguration(manifest).kvFormat, "unknown")
  for (const layer of manifest.kv_cache.layers) {
    layer.layout_status = "qualified"
    layer.numerical_encoding = null
  }
  assert.equal(recordingConfiguration(manifest).kvFormat, "unknown")
})

test("LoRA modes require admitted routing evidence and a single actual adapter identity", () => {
  const manifest = structuredClone(original)
  manifest.engine_settings.enable_lora = true
  manifest.arrivals[0].adapter_id = "real-adapter"
  assert.equal(recordingConfiguration(manifest).loraMode, "mixed")
  for (const arrival of manifest.arrivals) arrival.adapter_id = "real-adapter"
  assert.equal(recordingConfiguration(manifest).loraMode, "all")
  manifest.arrivals[0].adapter_id = "another-adapter"
  assert.equal(recordingConfiguration(manifest).loraMode, "unknown")
  manifest.arrivals[0].adapter_id = "real-adapter"
  manifest.arrivals[0].engine_admission_observed = false
  assert.equal(recordingConfiguration(manifest).loraMode, "unknown")
})

test("availability exposes matching captures and concrete unsupported precision instead of fabricated variants", () => {
  assert.equal(demoAvailability([entry], DEFAULT_DEMO_SELECTION).status, "recorded")
  const pending = demoAvailability([entry], { ...DEFAULT_DEMO_SELECTION, weightFormat: "fp8" })
  assert.equal(pending.status, "capture_required")
  assert.equal(pending.recordings.length, 0)
  const unsupported = demoAvailability([entry], { ...DEFAULT_DEMO_SELECTION, weightFormat: "nvfp4" })
  assert.equal(unsupported.status, "unsupported")
  assert.match(unsupported.reason, /qualified NVFP4 checkpoint/)
  const hybrid = demoAvailability([entry], { ...DEFAULT_DEMO_SELECTION, modelId: "Qwen/Qwen3.8-27B" })
  assert.equal(hybrid.status, "capture_required")
  assert.match(hybrid.reason, /hybrid-layer diagnostics/)
  const flash = demoAvailability([entry], { ...DEFAULT_DEMO_SELECTION, modelId: "Qwen/Qwen3.8-Flash-Next" })
  assert.equal(flash.status, "unsupported")
  assert.match(flash.reason, /one H100 80GB.*dedicated runtime image.*4× H100.*172.78 GiB/)
})

test("capture settings preserve the connection and serialize independent weight and KV requests", () => {
  const current = { ...defaultSettings(), hostKind: "ssh" as const, sshHost: "gpu.example", runnerUrl: "http://127.0.0.1:8766", modelRevision: "old-model-commit" }
  const selected = demoCaptureSettings(current, { ...DEFAULT_DEMO_SELECTION, weightFormat: "fp8", kvFormat: "fp8_e5m2" })
  assert.equal(selected.sshHost, current.sshHost)
  assert.equal(selected.runnerUrl, current.runnerUrl)
  assert.equal(selected.quantization, "fp8")
  assert.equal(selected.dtype, "auto")
  assert.equal(selected.kvCacheDtype, "fp8_e5m2")
  assert.equal(selected.modelRevision, "")
  assert.equal(selected.demoLoraMode, "none")
  const body = toRunBody(staggeredExample(), selected)
  assert.equal(body.kv_cache_dtype, "fp8_e5m2")
  assert.equal(body.vllm_configuration.kv_cache_dtype, "fp8_e5m2")
  assert.equal(body.model_configuration.quantization.kind, "fp8")
  assert.equal(current.modelRevision, "old-model-commit")
  assert.deepEqual(validateSettings(selected), [])
})

test("targeted numerical reruns restore KV dtype and extended profile windows retain bounded validation", () => {
  const current = { ...defaultSettings(), kvCacheDtype: "fp8_e5m2" as const }
  const target = { requestId: "sample-1", tokenPosition: 0, layerIndex: 0, kvHead: 0 }
  assert.equal(tensorRerunSettings(current, { engine_settings: { kv_cache_dtype: "fp8_e4m3" } }, target).kvCacheDtype, "fp8_e4m3")
  assert.equal(tensorRerunSettings(current, {}, target).kvCacheDtype, "auto")
  assert.deepEqual(validateSettings({ ...defaultSettings(), profilingMode: "eager_diagnostic", profilingMaxSteps: "64" }), [])
  assert.ok(validateSettings({ ...defaultSettings(), profilingMode: "eager_diagnostic", profilingMaxSteps: "1025" }).some((issue) => issue.fieldId === "setting-profiling-steps"))
})

test("demo LoRA assignment uses a configured artifact and preserves stable request identities", () => {
  const scenario = staggeredExample()
  assert.throws(() => demoAdapterScenario(scenario, defaultSettings(), "mixed"), /adapter repository or path/)
  const settings = { ...defaultSettings(), loraAdapters: [{ id: "pinned-adapter", name: "Actual adapter", source: "host" as const, location: "/models/adapter", revision: "commit" }] }
  const assigned = demoAdapterScenario(scenario, settings, "mixed")
  assert.deepEqual(toRunBody(assigned, settings).requests.map((request) => request.adapter_id), [null, "pinned-adapter", null, "pinned-adapter", null, "pinned-adapter"])
  assert.deepEqual(toRunBody(demoAdapterScenario(scenario, settings, "all"), settings).requests.map((request) => request.adapter_id), Array(6).fill("pinned-adapter"))
  assert.deepEqual(toRunBody(demoAdapterScenario(assigned, settings, "none"), settings).requests.map((request) => request.adapter_id), Array(6).fill(null))
  assert.deepEqual(toRunBody(assigned, settings).requests.map((request) => request.request_id), toRunBody(scenario, settings).requests.map((request) => request.request_id))
  assert.ok(scenario.batches.every((batch) => batch.samples.every((sample) => sample.adapterId === null)))
})

test("pending demo LoRA intent cannot silently capture base requests and remains editable", () => {
  const scenario = staggeredExample()
  const selection = { ...DEFAULT_DEMO_SELECTION, loraMode: "mixed" as const }
  const pending = demoCaptureSettings(defaultSettings(), selection)
  assert.equal(pending.demoLoraMode, "mixed")
  assert.ok(validateScenario(scenario, null, pending).some((issue) => issue.fieldId === "adapter-demo-mode"))
  const configured = { ...pending, loraAdapters: [{ id: "actual-a", name: "Compatible adapter", source: "host" as const, location: "/adapter/a", revision: "commit" }] }
  assert.ok(validateScenario(scenario, null, configured).some((issue) => issue.fieldId === "adapter-demo-mode"))
  const mixed = demoAdapterScenario(scenario, configured, "mixed")
  assert.deepEqual(validateScenario(mixed, null, configured), [])
  const all = { ...configured, demoLoraMode: "all" as const }
  assert.ok(validateScenario(mixed, null, all).some((issue) => issue.fieldId === "adapter-demo-mode"))
  assert.deepEqual(validateScenario(demoAdapterScenario(scenario, all, "all"), null, all), [])
  assert.deepEqual(validateScenario(scenario, null, { ...configured, demoLoraMode: null }), [])
  const base = demoCaptureSettings(configured, { ...selection, loraMode: "none" })
  assert.equal(base.loraAdapters.length, 0)
  assert.ok(validateScenario(mixed, null, base).some((issue) => issue.fieldId === "adapter-demo-mode"))
  assert.deepEqual(validateScenario(demoAdapterScenario(mixed, base, "none"), null, base), [])
  const requestBody = toRunBody(mixed, configured) as unknown as Record<string, unknown>
  assert.equal("demoLoraMode" in requestBody, false)
  assert.equal(tensorRerunSettings(configured, {}, { requestId: "sample-1", tokenPosition: 0, layerIndex: 0, kvHead: 0 }).demoLoraMode, null)
})
