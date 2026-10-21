import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { dirname, join } from "node:path"
import { test } from "node:test"
import { fileURLToPath } from "node:url"

import { explainRequest } from "./explain.ts"
import { projectArchitecture, projectKvCache, projectRequestTokens } from "./learning.ts"
import { projectKvConfiguration, projectKvTensorPreview } from "./kv.ts"
import { projectMemory, storageBreakdown } from "./memory.ts"
import { parseProfileAttachment, readProfileArtifact, type ProfileAttachment } from "./profiling.ts"
import { recordingConfiguration } from "../../web/src/shell/demoCatalog.ts"
import { readRecordingFixture, readRecordingFixtureBytes } from "../../web/test/recordingFixture.ts"
import {
  attachContent,
  createPlayer,
  parseJsonl,
  type ContentRecord,
  type RecordingManifest,
  type ReplayState,
  type TraceEvent,
} from "./reduce.ts"

const catalog = join(dirname(fileURLToPath(import.meta.url)), "../../web/public/recordings")
const legacyGpuIds = new Set(["gpu-kv-layer-diagnostics", "gpu-model-batching", "gpu-prefix-reuse"])
const expandedGpuFormats = new Map([
  ["gpu-h100-bf16", { weights: "bf16", kv: "bf16", lora: "none" }],
  ["gpu-h100-fp8-weights", { weights: "fp8", kv: "bf16", lora: "none" }],
  ["gpu-h100-fp8-kv", { weights: "bf16", kv: "fp8_e4m3", lora: "none" }],
  ["gpu-h100-lora-all", { weights: "bf16", kv: "bf16", lora: "all" }],
  ["gpu-h100-lora-mixed", { weights: "bf16", kv: "bf16", lora: "mixed" }],
  ["gpu-h100-qwen38-27b", { weights: "bf16", kv: "bf16", lora: "none" }],
])
const realisticGpuFormats = new Map([...expandedGpuFormats].map(([id, format]) => [id.replace("gpu-h100-", "gpu-h100-realistic-"), format]))


test("demo catalog contains real GPU lessons; retained regression fixtures preserve their provenance", () => {
  const index = JSON.parse(readRecordingFixture(join(catalog, "index.json"))) as {
    recordings: Array<{ id: string; evidence: string }>
  }
  const ids = new Set(index.recordings.map((item) => item.id))
  assert.equal(index.recordings[0].id, "gpu-h100-realistic-bf16")
  assert.ok(index.recordings.every((item) => item.evidence === "gpu"))
  for (const required of [...realisticGpuFormats.keys(), "gpu-h100-bf16", ...legacyGpuIds]) {
    assert.ok(ids.has(required), required)
  }
  const fixtures = [
    "model-kv-walkthrough",
    "continuous-batching",
    "single-request",
    "chunked-prefill",
    "prefix-reuse",
    "prefix-miss",
    "preemption",
    "cancellation",
    "incomplete-capture",
  ]
  for (const item of [...index.recordings, ...fixtures.map((id) => ({ id, evidence: "fixture" }))]) {
    const manifest = JSON.parse(readRecordingFixture(join(catalog, item.id, "manifest.json"))) as RecordingManifest
    const events = parseJsonl<TraceEvent>(readRecordingFixture(join(catalog, item.id, "events.jsonl")))
    const content = parseJsonl<ContentRecord>(readRecordingFixture(join(catalog, item.id, "content.jsonl")))
    assert.equal(manifest.execution_kind, item.evidence)
    const player = createPlayer(events, manifest)
    const state = attachContent(player.at(player.maxSequence), content)
    if (item.id === "incomplete-capture") {
      assert.equal(state.trust, "unknown")
      assert.equal(state.incomplete, true)
    } else {
      assert.equal(state.trust, "known")
    }
    if (item.evidence === "gpu") {
      assert.equal((manifest.provenance as { real_gpu?: boolean }).real_gpu, true)
      assert.equal((manifest.engine as { version?: string }).version, "0.30.0")
      assert.equal(projectKvCache(state, manifest, content).evidenceKind, "gpu")
      const configuration = projectKvConfiguration(state, manifest)
      if (legacyGpuIds.has(item.id)) assert.equal(configuration.pools[0].layerIndices.length, 36)
      else assert.ok(expandedGpuFormats.has(item.id) || realisticGpuFormats.has(item.id), `Add explicit expectations for the new GPU lesson ${item.id}.`)
      const artifact = readProfileArtifact(manifest)
      if (expandedGpuFormats.has(item.id) || realisticGpuFormats.has(item.id)) assert.ok(artifact, `${item.id} must retain its native profile.`)
      if (artifact) {
        const bytes = readRecordingFixtureBytes(join(catalog, item.id, artifact.path))
        assert.equal(bytes.length, artifact.sizeBytes, item.id)
        assert.equal(createHash("sha256").update(bytes).digest("hex"), artifact.sha256, item.id)
        assert.ok(bytes.length <= 512 * 1024 * 1024, `${item.id} exceeds the browser attachment bound.`)
        const attachment = parseProfileAttachment(bytes.toString("utf8"), manifest)
        if (item.id === "gpu-kv-layer-diagnostics") {
          assert.equal(configuration.status, "observed")
          assert.ok(configuration.layers.every((layer) => layer.layoutStatus === "qualified" && layer.device === "cuda:0" && layer.dtype === "bfloat16"))
          assert.equal(configuration.uniqueStorageBytes, configuration.storages.reduce((total, storage) => total + storage.byteCount, 0))
          assert.ok(attachment.cacheWrites.length >= 36)
          assert.ok(attachment.cacheWrites.every((write) => write.status === "observed" && write.gpuActivityIds.length > 0))
          const tensors = projectKvTensorPreview(manifest, attachment)
          assert.equal(tensors.status, "observed")
          assert.equal(tensors.snapshots.length, 24)
          assert.deepEqual([...new Set(tensors.snapshots.map((slice) => slice.layerIndex))].sort((a, b) => a - b), [0, 18, 35])
          assert.ok(tensors.snapshots.every((slice) => slice.requestId === "prefix-prime" && slice.kvHead === 0 && slice.values.length === 16 && slice.values.every(Number.isFinite)))
          assert.ok(player.steps.some((step) => step.requests.some((request) => request.promptTokens > 0) && step.requests.some((request) => request.generationTokens > 0)))
          assert.ok((state.requests["prefix-reader"].matchedPrefixTokens ?? 0) > 0)
          for (const requestId of Object.keys(state.requests)) {
            const finalSample = projectRequestTokens(state, requestId, content).at(-1)
            assert.equal(finalSample?.kvState, "not_computed", requestId)
          }
        } else if (item.id === "gpu-model-batching" || item.id === "gpu-prefix-reuse") {
          // These archives predate per-layer cache scopes and numerical snapshots.
          assert.equal(configuration.status, "unavailable")
          assert.equal(attachment.cacheWrites.length, 0)
          assert.equal(projectKvTensorPreview(manifest, attachment).status, "unavailable")
        } else {
          checkExpandedGpuLesson(item.id, manifest, events, content, player, state, attachment)
        }
      }
      if (item.id === "gpu-model-batching") assert.ok(player.steps.some((step) => step.requests.some((request) => request.promptTokens > 0) && step.requests.some((request) => request.generationTokens > 0)))
      if (item.id === "gpu-prefix-reuse") assert.ok(Object.values(state.requests).some((request) => (request.matchedPrefixTokens ?? 0) > 0))
    }
    if (item.id === "model-kv-walkthrough") {
      const architecture = projectArchitecture(manifest)
      assert.equal(architecture.supported, true)
      assert.equal(architecture.modelType, "qwen3")
      assert.equal(architecture.source, "pinned_model_config_reference")
      assert.equal(architecture.layerCount, 36)
      assert.equal(architecture.queryHeads, 32)
      assert.equal(architecture.kvHeads, 8)
      const cache = projectKvCache(state, manifest, content)
      assert.equal(cache.status, "observed")
      assert.equal(cache.pools.length, 1)
      assert.equal(cache.pools[0].capacity, 8)
      assert.deepEqual(cache.pools[0].blocks.map((block) => block.blockId), [0, 1, 2, 3, 4, 5, 6, 7])
      assert.equal(cache.groups.length, 1)
      assert.equal(cache.groups[0].tokenSlotMapping, "authored_fixture")
      assert.equal(Object.keys(state.requests).length, 6)
      for (const requestId of Object.keys(state.requests)) {
        const tokens = projectRequestTokens(state, requestId, content)
        const finalSample = tokens.at(-1)
        assert.ok(finalSample, requestId)
        assert.equal(finalSample.kind, "output", requestId)
        assert.equal(finalSample.kvState, "not_computed", requestId)
        assert.equal(finalSample.computedEver, false, requestId)
      }
    }
    if (item.id === "continuous-batching") {
      assert.ok(player.steps.some((step) => step.requests.length >= 2))
    }
    if (item.id === "prefix-reuse") {
      assert.equal(state.requests.second.matchedPrefixTokens, 4)
      const explanation = explainRequest(state, "second")
      assert.match(explanation.observed.join("\n"), /matched 4 tokens/)
      assert.match(explanation.explanatory.join("\n"), /does not skip ordinary decoding/)
    }
    if (item.id === "preemption") {
      const preempt = events.find((entry) => entry.event_type === "preemption")
      assert.ok(preempt)
      const mid = player.at(preempt.sequence)
      assert.equal(mid.requests.crowded.recomputing, true)
      assert.equal(mid.requests.crowded.preemptionReason, null)
      assert.ok(Object.values(state.blocks).some((block) => block.blockId === 3 && block.generation === 2))
    }
    if (item.id === "single-request") {
      assert.match(state.requests["request-a"].text, /●/)
    }
    if (item.id === "chunked-prefill") {
      assert.match(state.requests.long.text, /Done/)
      assert.ok(player.steps.some((step) => step.requests.length === 2))
    }
  }
})

function checkExpandedGpuLesson(id: string, manifest: RecordingManifest, events: TraceEvent[], content: ContentRecord[],
  player: ReturnType<typeof createPlayer>, state: ReplayState, attachment: ProfileAttachment) {
  const realistic = realisticGpuFormats.has(id)
  const baseId = realistic ? id.replace("gpu-h100-realistic-", "gpu-h100-") : id
  const expected = expandedGpuFormats.get(baseId)!
  const configuration = recordingConfiguration(manifest)
  assert.equal(configuration.weightFormat, expected.weights, id)
  assert.equal(configuration.kvFormat, expected.kv, id)
  assert.equal(configuration.loraMode, expected.lora, id)
  assert.equal(manifest.completion_status, "completed", id)
  assert.match((manifest.hardware as { gpu: string }).gpu, /H100/, id)
  assert.equal((manifest.initial_cache_state as { reset_verified: boolean }).reset_verified, true, id)
  assert.equal((manifest.capture as { status: string }).status, "completed", id)

  const workload = manifest.workload_plan as { batches: Array<{ id: string; arrival_offset_ms: number; samples: Array<{ id: string; max_output_tokens: number }> }> }
  assert.equal(workload.batches.length, 3, id)
  assert.deepEqual(workload.batches.map((batch) => batch.samples.length), [1, 2, 3], id)
  assert.equal(new Set(workload.batches.map((batch) => batch.arrival_offset_ms)).size, 3, id)
  const requests = workload.batches.flatMap((batch) => batch.samples)
  const requestIds = requests.map((request) => request.id)
  assert.deepEqual(requestIds, Array.from({ length: 6 }, (_, index) => `request-${index + 1}`), id)
  assert.deepEqual(Object.keys(state.requests).sort(), requestIds, id)
  const outputs = manifest.engine_output_token_ids as Record<string, number[]>
  for (const request of requests) {
    assert.equal(request.max_output_tokens, realistic ? [64, 96, 128, 160, 192, 256][requests.indexOf(request)] : 6, `${id}/${request.id}`)
    assert.ok(outputs[request.id].length >= (realistic ? 64 : 1) && outputs[request.id].length <= request.max_output_tokens, `${id}/${request.id}`)
    if (realistic) assert.match(state.requests[request.id].promptText!, /Meridian/)
    assert.deepEqual(state.requests[request.id].acceptedTokenIds, outputs[request.id], `${id}/${request.id}`)
    assert.ok((state.requests[request.id].promptTokenCount ?? 0) > 0, `${id}/${request.id}`)
    const finalSample = projectRequestTokens(state, request.id, content).at(-1)
    assert.equal(finalSample?.kind, "output", `${id}/${request.id}`)
    assert.equal(finalSample?.kvState, baseId === "gpu-h100-qwen38-27b" ? "unknown" : "not_computed", `${id}/${request.id}`)
  }
  if (baseId === "gpu-h100-bf16") {
    assert.equal(outputs["request-1"].length, realistic ? 64 : 6)
    assert.ok((state.requests["request-2"].matchedPrefixTokens ?? 0) > 0)
    assert.ok(player.steps.some((step) => step.requests.some((request) => request.promptTokens > 0)
      && step.requests.some((request) => request.generationTokens > 0)))
  }

  // Arrival groups are workload input; native packed rows describe actual scheduler iterations.
  assert.ok(player.steps.length > workload.batches.length, id)
  assert.deepEqual(attachment.steps.map((step) => step.stepId), player.steps.map((step) => step.id), id)
  assert.equal((manifest.profiling as { captured_steps: number }).captured_steps, player.steps.length, id)
  for (const step of player.steps) {
    assert.equal(step.status, "result_received", `${id}/${step.id}`)
    const native = attachment.steps.find((candidate) => candidate.stepId === step.id)!
    assert.equal(native.mappingStatus, "observed", `${id}/${step.id}`)
    assert.equal(native.paddingStatus, "observed", `${id}/${step.id}`)
    assert.equal(native.unpaddedTokens, step.totalScheduledTokens, `${id}/${step.id}`)
    const scheduled = step.requests.flatMap((request) => Array.from({ length: request.rangeLength }, (_, offset) =>
      `${request.requestId}:${request.rangeStart + offset}`)).sort()
    const packed = native.packedRows.filter((row) => !row.isPadding).map((row) => `${row.requestId}:${row.tokenPosition}`).sort()
    assert.deepEqual(packed, scheduled, `${id}/${step.id}`)
    assert.equal(native.packedRows.filter((row) => row.isPadding).length, native.paddedTokens! - native.unpaddedTokens!, `${id}/${step.id}`)
  }

  const normalized = attachment.normalization as { status: string; gpu_activity_count: number; correlated_gpu_activity_count: number;
    attachment_retention?: { status: string; byte_limit: number; original_activity_count: number; retained_activity_count: number;
      omitted_cpu_activity_count: number; omitted_gpu_activity_count: number; gpu_activities_complete: boolean } }
  assert.deepEqual(normalized, (manifest.profiling as { normalization: object }).normalization, id)
  const gpu = attachment.activities.filter((activity) => activity.kind.startsWith("cuda_"))
  assert.ok(gpu.length > 0, id)
  assert.equal(gpu.length, normalized.gpu_activity_count, id)
  assert.equal(gpu.filter((activity) => activity.scope.correlation === "runtime_flow").length, normalized.correlated_gpu_activity_count, id)
  if (normalized.attachment_retention) {
    const retention = normalized.attachment_retention
    assert.equal(normalized.status, "partial", id)
    assert.equal(retention.status, "cpu_detail_bounded", id)
    assert.equal(retention.byte_limit, (realistic ? 512 : 64) * 1024 * 1024, id)
    assert.equal(retention.omitted_gpu_activity_count, 0, id)
    assert.equal(retention.gpu_activities_complete, true, id)
    assert.ok(retention.omitted_cpu_activity_count > 0, id)
    assert.equal(retention.retained_activity_count, attachment.activities.length, id)
    assert.equal(retention.original_activity_count, retention.retained_activity_count + retention.omitted_cpu_activity_count, id)
    assert.equal((manifest.profiling as { status: string }).status, "partial", id)
    assert.match((manifest.profiling as { reason: string }).reason, /Unrelated CPU detail.*All normalized CUDA activities/, id)
  } else {
    assert.equal(normalized.status, "completed", id)
    assert.equal((manifest.profiling as { status: string }).status, "completed", id)
  }

  const architecture = projectArchitecture(manifest)
  assert.equal(architecture.supported, true, id)
  assert.equal(architecture.source, "loaded_model_config", id)
  assert.equal(architecture.layerTypes.length, architecture.layerCount, id)
  const attentionLayers = architecture.layerTypes.flatMap((type, layer) => type === "full_attention" ? [layer] : [])
  const recurrentLayers = architecture.layerTypes.flatMap((type, layer) => type === "linear_attention" ? [layer] : [])
  const kv = projectKvConfiguration(state, manifest, content)
  assert.equal(kv.status, "observed", id)
  assert.deepEqual(kv.layers.map((layer) => layer.layerIndex).sort((a, b) => a - b), attentionLayers, id)
  assert.ok(kv.layers.every((layer) => layer.device === "cuda:0"), id)
  if (baseId === "gpu-h100-qwen38-27b") assert.ok(kv.layers.every((layer) => layer.layoutStatus === "qualified" || layer.layoutReason != null), id)
  else assert.ok(kv.layers.every((layer) => layer.layoutStatus === "qualified"), id)
  assert.equal(kv.uniqueStorageBytes, kv.storages.reduce((sum, storage) => sum + storage.byteCount, 0), id)
  assert.ok(attachment.cacheWrites.every((write) => write.status === "observed" && write.gpuActivityIds.length > 0), id)
  for (const step of attachment.steps) {
    assert.deepEqual(attachment.cacheWrites.filter((write) => write.stepId === step.stepId).map((write) => write.layerIndex).sort((a, b) => a - b), attentionLayers, `${id}/${step.stepId}`)
  }
  const tensors = projectKvTensorPreview(manifest, attachment)
  if (baseId === "gpu-h100-qwen38-27b") {
    assert.equal(tensors.status, "unavailable", id)
    assert.equal(tensors.capture?.enabled, false, id)
    assert.deepEqual(tensors.snapshots, [], id)
  } else {
    assert.equal(tensors.status, "observed", id)
    assert.ok(tensors.snapshots.length > 0, id)
  }
  assert.ok(tensors.snapshots.every((slice) => attentionLayers.includes(slice.layerIndex) && slice.values.length > 0 && slice.values.every(Number.isFinite)), id)
  if (expected.kv === "fp8_e4m3") {
    assert.ok(kv.layers.every((layer) => layer.quantization?.cacheDtype === "fp8_e4m3"), id)
    assert.ok(tensors.snapshots.every((slice) => slice.quantization?.format === "fp8_e4m3"
      && Number.isFinite(slice.quantization.scale.value) && slice.quantization.rawBits.length === slice.values.length), id)
  }
  if (baseId === "gpu-h100-bf16") {
    assert.equal(tensors.snapshots.length, 24)
    assert.deepEqual([...new Set(tensors.snapshots.map((slice) => slice.layerIndex))].sort((a, b) => a - b), [0, 18, 35])
    assert.deepEqual([...new Set(tensors.snapshots.map((slice) => slice.kvHead))].sort(), [0, 1])
  }

  assert.equal((manifest.gpu_memory as { status: string }).status, "completed", id)
  const phases = events.filter((event) => event.event_type === "gpu_memory_phase").map((event) => event.phase)
  for (const phase of ["model_load", "cache_initialization", "inference", "cleanup"]) assert.ok(phases.includes(phase), `${id}/${phase}`)
  const lastResult = player.steps.at(-1)!.resultSequence!
  const memory = projectMemory(events, manifest, lastResult)
  assert.equal(memory.current?.status, "observed", id)
  assert.ok(memory.current!.devices[0].used > 0 && memory.current!.devices[0].processUsed! > 0, id)
  assert.ok(memory.samples.some((sample) => sample.framework != null && sample.framework.allocated > 0), id)
  assert.equal(memory.inventory?.status, "observed", id)
  const storage = storageBreakdown(memory.inventory, "cuda:0")
  assert.ok(storage.some((row) => row.category === "weights" && row.bytes > 0), id)
  if (baseId === "gpu-h100-qwen38-27b") {
    assert.ok(memory.inventory!.storages.some((row) => row.categories.includes("attention_kv") && row.bytes > 0), id)
  } else assert.ok(storage.some((row) => row.category === "attention_kv" && row.bytes === kv.uniqueStorageBytes), id)
  assert.equal(new Set(memory.inventory!.storages.map((row) => row.id)).size, memory.inventory!.storages.length, id)
  assert.equal(projectMemory(events, manifest, player.maxSequence).phase, "cleanup", id)
  assert.equal(projectMemory(events, manifest, player.maxSequence).inventory, null, id)

  if (baseId === "gpu-h100-qwen38-27b") {
    assert.equal(architecture.modelType, "qwen3_5_text", id)
    assert.ok(recurrentLayers.length > 0 && attentionLayers.length > 0, id)
    assert.ok(architecture.recurrent != null, id)
    assert.ok(state.cacheGroups.some((group) => group.cacheType?.includes("MambaSpec") && group.tokenSlotMapping !== "qualified_full_attention"), id)
    assert.ok(memory.inventory!.storages.some((row) => row.categories.includes("recurrent_state") && row.bytes > 0), id)
    const recurrentScopes = attachment.activities.filter((activity) => activity.scope.component === "recurrent")
    assert.ok(recurrentScopes.some((activity) => activity.kind.startsWith("cuda_") && activity.scope.correlation === "runtime_flow"), id)
    assert.ok(recurrentScopes.every((activity) => recurrentLayers.includes(activity.scope.layerIndex!)), id)
    assert.ok(attachment.cacheWrites.every((write) => !recurrentLayers.includes(write.layerIndex)), id)
  } else {
    assert.equal(architecture.modelType, "qwen3", id)
    assert.deepEqual(recurrentLayers, [], id)
  }

  const adapters = (manifest.model as { lora_adapters: Array<{ id: string; lora_int_id: number; configuration_status: string; rank: number; artifact_sha256: Record<string, string> }> }).lora_adapters
  const arrivals = manifest.arrivals!
  assert.equal(arrivals.length, 6, id)
  assert.ok(arrivals.every((arrival) => arrival.engine_admission_observed === true && arrival.http_receipt_observed === true), id)
  const actualRoutes = new Set(arrivals.map((arrival) => arrival.adapter_id))
  if (expected.lora === "none") {
    assert.deepEqual(adapters, [], id)
    assert.deepEqual([...actualRoutes], ["base"], id)
  } else {
    assert.equal(adapters.length, 1, id)
    assert.equal(adapters[0].configuration_status, "observed", id)
    assert.ok(adapters[0].rank > 0 && adapters[0].lora_int_id > 0, id)
    assert.ok(Object.values(adapters[0].artifact_sha256).length >= 2 && Object.values(adapters[0].artifact_sha256).every((hash) => /^[a-f0-9]{64}$/.test(hash)), id)
    assert.deepEqual([...actualRoutes].sort(), expected.lora === "all" ? [adapters[0].id] : [adapters[0].id, "base"].sort(), id)
    for (const arrival of arrivals) {
      const request = state.requests[String(arrival.request_id)]
      assert.equal(request.adapterId, arrival.adapter_id, `${id}/${request.id}`)
      assert.equal(request.loraIntId, arrival.lora_int_id ?? null, `${id}/${request.id}`)
      assert.equal(request.modelRouteKind, arrival.adapter_id === "base" ? "base_model" : "lora_adapter", `${id}/${request.id}`)
    }
  }
}
