import { createHash } from "node:crypto"
import { readFileSync, writeFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

import { attachContent, createPlayer, parseJsonl } from "../../../replay/src/reduce.ts"
import { buildTimeline, projectSamples } from "../../../replay/src/timeline.ts"
import { projectKvConfiguration } from "../../../replay/src/kv.ts"
import { projectKvCache, projectRequestTokens } from "../../../replay/src/learning.ts"
import { mappedBlockSlot } from "../../../web/src/architecture/model.ts"

const recording = new URL("../../../web/public/recordings/gpu-kv-layer-diagnostics/", import.meta.url)
const files = ["manifest.json", "events.jsonl", "content.jsonl"]
const raw = Object.fromEntries(files.map((name) => [name, readFileSync(new URL(name, recording), "utf8")]))
const manifest = JSON.parse(raw["manifest.json"])
const events = parseJsonl(raw["events.jsonl"])
const content = parseJsonl(raw["content.jsonl"])
const player = createPlayer(events, manifest)
const timeline = buildTimeline(events, manifest)

const rows = [76, 93, 110, 127, 128, 138, 139, 140, 141, 150, 151, 157, 167, 178, 209, 399].map((sequence) => {
  const state = attachContent(player.at(sequence), content)
  const sample = projectSamples(timeline, state, sequence).find((item) => item.sample.id === "prefix-prime")
  const pool = projectKvConfiguration(state, manifest, content).pools[0]
  const block1 = projectKvCache(state, manifest, content).pools[0].blocks.find((block) => block.blockId === 1)
  const slot = mappedBlockSlot(block1, 16, 3, "prefix-prime")
  return {
    sequence,
    elapsed_ms: (events[sequence - 1].monotonic_ns - manifest.controller_origin.controller_started_monotonic_ns) / 1e6,
    event_type: events[sequence - 1].event_type,
    request_id: events[sequence - 1].request_id ?? null,
    prime: {
      status: state.requests["prefix-prime"].status,
      phase: sample.phase,
      accepted: sample.acceptedTokenCount,
      max_output_tokens: sample.sample.maxOutputTokens,
      output_text: sample.outputText,
      accepted_token_ids: state.requests["prefix-prime"].acceptedTokenIds,
      finish_reason: state.requests["prefix-prime"].finishReason,
      output_positions: projectRequestTokens(state, "prefix-prime", content).filter((token) => token.position >= 173).map((token) => ({
        position: token.position, token_id: token.tokenId, token_piece: token.tokenPiece, kv_state: token.kvState,
      })),
    },
    reader_status: state.requests["prefix-reader"]?.status ?? null,
    pool: { referenced: pool.referencedBlocks, cached: pool.cachedBlocks, uncached_free: pool.freeBlocks },
    block_1_slot_3: {
      current_holders: block1.bindings.map((binding) => binding.requestId),
      mapped_request: slot.requestId, original_writer: slot.sourceRequestId, token_piece: slot.tokenPiece,
    },
  }
})

const receipt = {
  run_id: manifest.run_id,
  source_files_sha256: Object.fromEntries(files.map((name) => [name, createHash("sha256").update(raw[name]).digest("hex")])),
  method: "Production createPlayer, attachContent, projectSamples, projectKvConfiguration, projectKvCache, projectRequestTokens and mappedBlockSlot; no UI rendering claimed.",
  rows,
}
const output = process.argv[2] ?? fileURLToPath(new URL("./demo-lifecycle-projection.json", import.meta.url))
writeFileSync(output, JSON.stringify(receipt, null, 2) + "\n")
console.log(JSON.stringify({ run_id: receipt.run_id, rows: rows.length, output, sha256: receipt.source_files_sha256 }))
