import assert from "node:assert/strict"
import test from "node:test"
import * as THREE from "three"

import { kvHeadForQuery, mappedBlockSlot, observedProfileTokenPositions, poolCellPosition, poolGridLayout, scheduledTokenPositions, sceneClick, teachingLayer } from "../architecture/model.ts"
import { createPoolRaycast } from "../architecture/poolPicking.ts"

test("a real-sized pool has a unique, stable cell for every physical ID", () => {
  const layout = poolGridLayout(35000)
  const cells = new Set(Array.from({ length: layout.capacity }, (_, index) => poolCellPosition(index, layout).join(",")))
  assert.equal(cells.size, 35000)
  assert.equal(layout.capacity, 35000)
  assert.ok(layout.columns * layout.rows * layout.depths >= layout.capacity)
  assert.deepEqual(poolCellPosition(34999, layout), poolCellPosition(34999, poolGridLayout(35000)))
  assert.equal(poolGridLayout(-1).capacity, 0)
  assert.equal(poolGridLayout(Infinity).capacity, 0)
})

test("fast pool picking reaches the final physical block and skips empty grid padding", () => {
  const layout = poolGridLayout(35000)
  const mesh = poolMesh(layout)
  try {
    const last = new THREE.Vector3(...poolCellPosition(34999, layout))
    assert.equal(pick(mesh, last.clone().add(new THREE.Vector3(0, 0, -10)), new THREE.Vector3(0, 0, 1))[0]?.instanceId, 34999)
    const paddedIndex = layout.columns * layout.rows * layout.depths - 1
    const padded = new THREE.Vector3(...poolCellPosition(paddedIndex, layout))
    const expected = paddedIndex - layout.columns * layout.rows
    assert.ok(expected < layout.capacity)
    assert.equal(pick(mesh, padded.clone().add(new THREE.Vector3(0, 0, -10)), new THREE.Vector3(0, 0, 1))[0]?.instanceId, expected)
  } finally { dispose(mesh) }
})

test("pool picking respects gaps, near/far clipping and all six viewing directions", () => {
  const layout = poolGridLayout(8)
  const mesh = poolMesh(layout)
  try {
    assert.equal(pick(mesh, new THREE.Vector3(0, 1.75, 8), new THREE.Vector3(0, 0, -1)).length, 0)
    const front = new THREE.Vector3(...poolCellPosition(0, layout))
    const rayOrigin = front.clone().setZ(6)
    assert.equal(pick(mesh, rayOrigin, new THREE.Vector3(0, 0, -1))[0]?.instanceId, 0)
    assert.equal(pick(mesh, rayOrigin, new THREE.Vector3(0, 0, -1), 3.2)[0]?.instanceId, 4)
    assert.equal(pick(mesh, rayOrigin, new THREE.Vector3(0, 0, -1), 0, 2).length, 0)
    for (const axis of [0, 1, 2]) {
      for (const sign of [-1, 1]) {
        const id = axis === 0 ? sign > 0 ? 1 : 0 : axis === 1 ? sign > 0 ? 0 : 2 : sign > 0 ? 0 : 4
        const cell = new THREE.Vector3(...poolCellPosition(id, layout))
        const direction = new THREE.Vector3().setComponent(axis, -sign)
        const origin = cell.clone().setComponent(axis, sign * 8)
        assert.equal(pick(mesh, origin, direction)[0]?.instanceId, id, `axis ${axis}, sign ${sign}`)
      }
    }
    assert.equal(pick(mesh, front, new THREE.Vector3(0, 0, -1))[0]?.instanceId, 0)
  } finally { dispose(mesh) }
})

test("pool picking preserves block identity under translation, rotation and nonuniform scale", () => {
  const layout = poolGridLayout(8)
  const mesh = poolMesh(layout)
  try {
    mesh.position.set(5, -3, 2); mesh.rotation.set(0.3, 0.8, -0.2); mesh.scale.set(0.6, 1.3, 0.9)
    mesh.updateMatrixWorld(true)
    const local = new THREE.Vector3(...poolCellPosition(7, layout))
    const target = local.clone().applyMatrix4(mesh.matrixWorld)
    const origin = local.clone().add(new THREE.Vector3(0, 0, -10)).applyMatrix4(mesh.matrixWorld)
    const hit = pick(mesh, origin, target.clone().sub(origin).normalize())[0]
    assert.equal(hit?.instanceId, 7)
    assert.ok(hit && Math.abs(hit.distance - hit.point.distanceTo(origin)) < 1e-9)
  } finally { dispose(mesh) }
})

test("GQA maps all 32 query heads to the captured 8 KV heads without guessing incompatible geometry", () => {
  assert.deepEqual(Array.from({ length: 32 }, (_, index) => kvHeadForQuery(index, 32, 8)), Array.from({ length: 32 }, (_, index) => Math.floor(index / 4)))
  assert.equal(kvHeadForQuery(0, 7, 2), null)
  assert.equal(kvHeadForQuery(32, 32, 8), null)
  assert.equal(kvHeadForQuery(-1, 32, 8), null)
})

test("scheduled token geometry distinguishes prefill/decode and excludes the next sampled output", () => {
  const tokens = scheduledTokenPositions({ requests: [
    { requestId: "a", rangeStart: 2, rangeLength: 3, numPromptTokens: 4, promptTokens: 2, generationTokens: 1 },
    { requestId: "b", rangeStart: 8, rangeLength: 1, numPromptTokens: 8, promptTokens: 0, generationTokens: 1 },
  ] })
  assert.deepEqual(tokens, [
    { requestId: "a", position: 2, kind: "prefill" }, { requestId: "a", position: 3, kind: "prefill" },
    { requestId: "a", position: 4, kind: "decode" }, { requestId: "b", position: 8, kind: "decode" },
  ])
  assert.ok(!tokens.some((token) => token.requestId === "b" && token.position === 9))
})

test("logical teaching stages require an explicit teaching clock", () => {
  assert.equal(teachingLayer(undefined, 36), null)
  assert.equal(teachingLayer(NaN, 36), null)
  assert.equal(teachingLayer(0, 36), 0)
  assert.equal(teachingLayer(0.5, 36), 18)
  assert.equal(teachingLayer(1, 36), 35)
  assert.equal(teachingLayer(8, 36), 35)
})

test("measured row geometry follows observed native rows independently of host state and excludes padding", () => {
  const rows = observedProfileTokenPositions([
    { row: 2, requestId: null, tokenPosition: null, isPadding: true },
    { row: 1, requestId: "b", tokenPosition: 9, isPadding: false },
    { row: 0, requestId: "a", tokenPosition: 12, isPadding: false },
  ])
  assert.deepEqual(rows, [
    { requestId: "a", position: 12, row: 0, kind: "unclassified" },
    { requestId: "b", position: 9, row: 1, kind: "unclassified" },
  ])
})

test("reused prefix slots select the active request while retaining the original content source", () => {
  const contents = [{ requestId: "a", cacheGroup: 0, tokenStart: 0, tokenEnd: 2, slotStart: 0, slotEnd: 2,
    stepId: "first", sequence: 1, tokenIds: [100, 101], tokenPieces: ["Hello", " world"] }]
  const bindings = ["a", "b"].map((requestId) => ({ requestId, cacheGroup: 0, tokenStart: 0, tokenEnd: 4, stepId: "reuse", sequence: 2 }))
  const computed = mappedBlockSlot({ contents, bindings }, 16, 1, "b")
  assert.equal(computed.requestId, "b")
  assert.equal(computed.position, 1)
  assert.equal(computed.sourceRequestId, "a")
  assert.equal(computed.tokenId, 101)
  assert.equal(computed.tokenPiece, " world")
  const reserved = mappedBlockSlot({ contents, bindings }, 16, 2, "b")
  assert.equal(reserved.requestId, "b")
  assert.equal(reserved.position, 2)
  assert.equal(reserved.span, undefined)
  assert.equal(reserved.tokenId, null)
  assert.equal(mappedBlockSlot({ contents, bindings: [] }, 16, 1).requestId, "a")
  assert.equal(mappedBlockSlot({ contents: [], bindings }, 16, 1, "b").sourceRequestId, null)
})

test("rotate, Ctrl/Cmd pan and right-button pan never select objects on release", () => {
  const event = { delta: 0, nativeEvent: { button: 0, ctrlKey: false, metaKey: false, shiftKey: false } }
  assert.equal(sceneClick(event), true)
  assert.equal(sceneClick({ ...event, delta: 10 }), false)
  for (const modifier of ["ctrlKey", "metaKey", "shiftKey"] as const) {
    assert.equal(sceneClick({ ...event, nativeEvent: { ...event.nativeEvent, [modifier]: true } }), false)
  }
  assert.equal(sceneClick({ ...event, nativeEvent: { ...event.nativeEvent, button: 2 } }), false)
})

function poolMesh(layout: ReturnType<typeof poolGridLayout>): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial(), layout.capacity)
  mesh.raycast = createPoolRaycast(layout)
  mesh.updateMatrixWorld(true)
  return mesh
}
function pick(mesh: THREE.InstancedMesh, origin: THREE.Vector3, direction: THREE.Vector3, near = 0, far = Infinity) {
  const hits: THREE.Intersection[] = []
  mesh.raycast(new THREE.Raycaster(origin, direction, near, far), hits)
  return hits
}
function dispose(mesh: THREE.InstancedMesh) {
  mesh.geometry.dispose()
  if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose())
  else mesh.material.dispose()
  mesh.dispose()
}
