import type { ProfileTensorSnapshot } from "../../../replay/src/profiling.ts"

export interface NumericalFrame {
  key: string
  snapshots: ProfileTensorSnapshot[]
}

export interface NumericalCell {
  head: number
  dimension: number
  value: number
  snapshot: ProfileTensorSnapshot
}

export type NumericalValueMode = "stored" | "raw" | "reconstructed"

/** Presentation grouping only: heads can share a grid only at the same recorded operation and allocation. */
export function numericalFrames(snapshots: ProfileTensorSnapshot[]): NumericalFrame[] {
  const frames = new Map<string, NumericalFrame>()
  for (const snapshot of snapshots) {
    const key = JSON.stringify([
      snapshot.operationId, snapshot.stepId, snapshot.requestId, snapshot.tokenPosition,
      snapshot.layerIndex, snapshot.kind, snapshot.dtype, snapshot.device,
      snapshot.pool, snapshot.cacheGroup, snapshot.blockId, snapshot.allocationGeneration,
      snapshot.slot, snapshot.observationBoundary,
      snapshot.quantization?.format ?? "native",
    ])
    const frame = frames.get(key) ?? { key, snapshots: [] }
    frame.snapshots.push(snapshot)
    frames.set(key, frame)
  }
  return [...frames.values()]
}

/** Missing or conflicting coordinates stay absent; neither tensor geometry nor a neighboring head supplies a value. */
export function numericalCells(frame: NumericalFrame | undefined, heads: number, dimensions: number, mode: NumericalValueMode = "stored") {
  const cells = new Map<number, NumericalCell>()
  const conflicts = new Set<number>()
  for (const snapshot of frame?.snapshots ?? []) {
    if (!Number.isInteger(snapshot.kvHead) || snapshot.kvHead < 0 || snapshot.kvHead >= heads) continue
    const values = mode === "raw" ? snapshot.quantization?.rawBits ?? []
      : mode === "reconstructed" ? snapshot.quantization?.reconstructedValues ?? [] : snapshot.values
    values.forEach((value, offset) => {
      const dimension = snapshot.dimensionStart + offset
      if (!Number.isInteger(dimension) || dimension < 0 || dimension >= dimensions || !Number.isFinite(value)) return
      const key = snapshot.kvHead * dimensions + dimension
      const previous = cells.get(key)
      if (previous && previous.value !== value) conflicts.add(key)
      cells.set(key, { head: snapshot.kvHead, dimension, value, snapshot })
    })
  }
  for (const key of conflicts) cells.delete(key)
  return { cells, conflictCount: conflicts.size }
}
