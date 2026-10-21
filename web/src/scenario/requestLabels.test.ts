import assert from "node:assert/strict"
import test from "node:test"

import { arrivalGroup, requestLabel } from "../explore/model.ts"

test("requests are named by workload label and fall back to the recorded id", () => {
  const labels = new Map([["sample-1", "input_1"], ["sample-2", "  "], ["sample-3", ""]])
  assert.equal(requestLabel(labels, "sample-1"), "input_1")
  assert.equal(requestLabel(labels, "sample-2"), "sample-2")
  assert.equal(requestLabel(labels, "sample-3"), "sample-3")
  assert.equal(requestLabel(labels, "sample-9"), "sample-9")
  assert.equal(requestLabel(undefined, "sample-1"), "sample-1")
})

test("requests without a recorded arrival group share one honestly named group", () => {
  const first = arrivalGroup({ batchId: "ungrouped:req-1", batchLabel: "Ungrouped" })
  const second = arrivalGroup({ batchId: "ungrouped:req-2", batchLabel: "Ungrouped" })
  assert.equal(first.key, second.key)
  assert.equal(first.label, "Arrival group not recorded")
  assert.equal(first.recorded, false)
  assert.deepEqual(arrivalGroup({ batchId: "batch-b", batchLabel: "B" }), { key: "batch-b", label: "Arrival batch B", recorded: true })
  assert.equal(arrivalGroup({ batchId: "batch-1", batchLabel: "Batch 1" }).label, "Batch 1")
  assert.equal(arrivalGroup({ batchId: "batch-1", batchLabel: "Arrival batch 1" }).label, "Arrival batch 1")
})
