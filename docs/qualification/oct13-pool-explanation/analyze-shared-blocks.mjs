import assert from "node:assert/strict"
import fs from "node:fs"
import { fileURLToPath } from "node:url"
import { createPlayer, parseJsonl } from "../../../replay/src/reduce.ts"
import { projectKvCache } from "../../../replay/src/learning.ts"

const root = fileURLToPath(new URL("../../../", import.meta.url))
const recording = `${root}web/public/recordings/gpu-h100-bf16/`
const manifest = JSON.parse(fs.readFileSync(`${recording}manifest.json`, "utf8"))
const events = parseJsonl(fs.readFileSync(`${recording}events.jsonl`, "utf8"))
const content = parseJsonl(fs.readFileSync(`${recording}content.jsonl`, "utf8"))
const prompts = Object.fromEntries(content.filter(row => row.kind === "prompt").map(row => [row.request_id, row]))
const sequence = 1155
const state = createPlayer(events, manifest).at(sequence)
const blocks = projectKvCache(state, manifest, content).pools[0].blocks
const shared = blocks.filter(block => new Set(block.bindings.map(binding => binding.requestId)).size > 1)
const sharedIds = shared.map(block => block.blockId)
assert.deepEqual(sharedIds, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 59])
assert.equal(blocks[11].status, "cached")
assert.equal(blocks[11].refCount, 0)

const comparisons = [["request-1", "request-2"], ["request-1", "request-4"], ["request-2", "request-4"]].map(([a, b]) => {
  let common = 0
  while (common < Math.min(prompts[a].token_ids.length, prompts[b].token_ids.length)
    && prompts[a].token_ids[common] === prompts[b].token_ids[common]) common += 1
  return { requestIds: [a, b], commonPrefixTokens: common,
    whole16TokenBlocks: Math.floor(common / 16),
    firstDifference: { positionZeroBased: common, tokenIds: [prompts[a].token_ids[common], prompts[b].token_ids[common]],
      tokenPieces: [prompts[a].token_pieces[common], prompts[b].token_pieces[common]] } }
})
assert.deepEqual(comparisons.map(item => item.commonPrefixTokens), [175, 175, 177])

const audit = {
  recording: "gpu-h100-bf16", runId: manifest.run_id, candidateSequence: sequence,
  screenshotMatch: "The screenshot omits its playhead. This replay state matches its visible pool pattern; the exact event cannot be proven from the crop alone.",
  stepId: state.steps.at(-1).id,
  physicalLayout: "Ascending physical block IDs, sixteen columns per row, bottom row starts at zero. Adjacency is not token order.",
  activeRequestTables: Object.fromEntries(["request-2", "request-3", "request-4"].map(id => [id, state.requests[id].blockGroups])),
  sharedPhysicalIds: sharedIds,
  comparisons,
  tokenRange160Through175: Object.fromEntries(["request-1", "request-2", "request-4"].map(id => [id, {
    tokenIds: prompts[id].token_ids.slice(160, 176), tokenPieces: prompts[id].token_pieces.slice(160, 176) }])),
  selectedBlocks: blocks.filter(block => [0, 1, 10, 11, 12, 58, 59, 60, 61, 62].includes(block.blockId)).map(block => ({
    id: block.blockId, allocationGeneration: block.generation, status: block.status, isNull: block.isNull,
    refCount: block.refCount, validTokenCount: block.validTokenCount, currentBindings: block.bindings,
    recordedContents: block.contents.map(span => ({ writerRequestId: span.requestId, startInclusive: span.tokenStart,
      endExclusive: span.tokenEnd, stepId: span.stepId, sequence: span.sequence })) })),
  prefixLookups: events.filter(event => event.event_type === "prefix_lookup" && ["request-2", "request-4"].includes(event.request_id)),
  block59EventOrdering: events.filter(event => event.sequence <= sequence && event.block_id === 59),
  sameIterationCaveat: "Block 59 is hash-registered and reused during the same scheduling pass, before the host result observation of its write. This is not evidence of a completed previous model iteration.",
  evidence: { blockMarks: "web/src/architecture/BlockMarks.tsx:27", blockProjection: "replay/src/learning.ts:195",
    bindingReducer: "replay/src/reduce.ts:792", recordedStepTables: "web/public/recordings/gpu-h100-bf16/events.jsonl:1073",
    promptTokenSource: "web/public/recordings/gpu-h100-bf16/content.jsonl (kind=prompt)" }
}
const output = `${root}output/playwright/oct13-pool-explanation/shared-block-audit.json`
fs.writeFileSync(output, `${JSON.stringify(audit, null, 2)}\n`)
console.log(JSON.stringify({ output, sharedIds, prefixTokenCounts: comparisons.map(item => item.commonPrefixTokens) }))
