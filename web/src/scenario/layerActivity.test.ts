import assert from "node:assert/strict"
import test from "node:test"

import { projectArchitecture } from "../architecture/model.ts"
import { measuredLayerActivity } from "../execution/layerActivity.ts"

const architecture = { ...projectArchitecture(), layerCount: 4, layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"] }

test("simultaneous native layer activity retains all mapped layers and distinguishes GDN and FFN", () => {
  const result = measuredLayerActivity({ stepId: "step-1", packedRows: [], activeLayers: [3, 0, 0, -1, 4, 1.5], activeModules: [
    { layerIndex: 0, module: "recurrent" }, { layerIndex: 0, module: "recurrent" },
    { layerIndex: 3, module: "attention" }, { layerIndex: 3, module: "ffn" }, { layerIndex: 2, module: "ffn" },
  ] }, architecture)
  assert.deepEqual(result, [{ layerIndex: 0, detail: "Gated DeltaNet" }, { layerIndex: 3, detail: "Self-attention + Feed-forward" }])
})

test("native gaps retain no layer activity even when an iteration has observed participants", () => {
  assert.deepEqual(measuredLayerActivity({ stepId: "step-1", activeLayers: [], packedRows: [
    { row: 0, requestId: "request-1", tokenPosition: 10, isPadding: false },
  ], activeModules: [{ layerIndex: 0, module: "recurrent" }] }, architecture), [])
  assert.deepEqual(measuredLayerActivity({ stepId: "step-1", activeLayers: [0], packedRows: [] }, architecture), [{ layerIndex: 0, detail: "Gated DeltaNet layer" }])
})
