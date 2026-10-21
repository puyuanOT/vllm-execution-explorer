import type { EngineSettings, RunRequestBody } from "./types.ts"

/** Indices are zero-based; an empty selection lets the capture preset resolve it. */
export function captureIndices(raw: string, maximumCount: number): number[] | null {
  if (!raw.trim()) return []
  const parts = raw.trim().split(/[\s,]+/)
  if (parts.some((part) => !/^\d+$/.test(part) || !Number.isSafeInteger(Number(part)) || Number(part) > 2_000_000)) return null
  const values = [...new Set(parts.map(Number))].sort((a, b) => a - b)
  return values.length <= maximumCount ? values : null
}

export function diagnosticConfiguration(settings: EngineSettings): RunRequestBody["profiling_configuration"] {
  const base = { mode: settings.profilingMode, max_steps: Number(settings.profilingMaxSteps) }
  if (settings.profilingMode === "off") return base
  const custom = settings.tensorCapture && settings.tensorPreset === "custom"
  const indices = (raw: string, limit: number) => {
    const values = captureIndices(raw, limit)
    if (values == null) throw new Error("Numerical capture indices must be valid zero-based integers within the selection limit.")
    return values
  }
  return {
    ...base, start_step: Number(settings.profilingStartStep), trigger_request_id: settings.profilingTriggerRequestId || null,
    tensor_capture: {
      enabled: settings.tensorCapture, preset: settings.tensorCapture ? settings.tensorPreset : "small", request_id: settings.tensorCapture ? settings.tensorRequestId || null : null,
      token_positions: custom && settings.tensorCapture ? indices(settings.tensorTokenPositions, 64) : [],
      layer_indices: custom && settings.tensorCapture ? indices(settings.tensorLayerIndices, 64) : [],
      head_indices: custom && settings.tensorCapture ? indices(settings.tensorHeadIndices, 32) : [0],
      dimension_start: custom ? Number(settings.tensorDimensionStart) : 0,
      dimension_count: custom ? Number(settings.tensorDimensionCount) : 16,
      max_bytes: settings.tensorCapture ? Number(settings.tensorMaxBytes) : 1048576,
    },
  }
}
