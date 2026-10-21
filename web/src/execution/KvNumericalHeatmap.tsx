import { useId, useMemo, useState, type MouseEvent } from "react"
import { cn } from "../lib/cn.ts"
import { Segmented } from "../ui/controls.tsx"
import { numericalCells, type NumericalFrame, type NumericalValueMode } from "./numericalHeatmap.ts"

const COLORS = ["#bc8247", "#876743", "#514c41", "#253442", "#175d6a", "#2398ad", "#6fdbef"]
const BYTE_COLORS = ["#0b1722", "#233d52", "#38637b", "#5093ae", "#6fdbef"]

/** This grid displays retained observations, never a read of current device memory. */
export function KvNumericalHeatmap({ frame, heads, dimensions, selectedHead, kind, onHeadChange }: {
  frame?: NumericalFrame; heads: number | null; dimensions: number | null; selectedHead: number
  kind: "K" | "V"; onHeadChange: (head: number) => void
}) {
  const id = useId()
  const [view, setView] = useState<"heatmap" | "table">("heatmap")
  const [requestedValueMode, setValueMode] = useState<NumericalValueMode>("reconstructed")
  const [dimensionInput, setDimensionInput] = useState("0")
  const [copyFeedback, setCopyFeedback] = useState<{ key: string; message: string; failed: boolean } | null>(null)
  const headCount = heads ?? 0
  const dimensionCount = dimensions ?? 0
  const hasFp8 = frame?.snapshots.some((snapshot) => snapshot.quantization != null) === true
  const valueMode = hasFp8 ? requestedValueMode : "stored"
  const { cells, conflictCount } = useMemo(() => numericalCells(frame, headCount, dimensionCount, valueMode), [frame, headCount, dimensionCount, valueMode])
  const captured = [...cells.values()].sort((left, right) => left.head - right.head || left.dimension - right.dimension)
  const vector = captured.filter((cell) => cell.head === selectedHead)
  const dimension = dimensionInput.trim() === "" ? NaN : Number(dimensionInput)
  const validDimension = Number.isInteger(dimension) && dimension >= 0 && dimension < dimensionCount
  const selectedCell = validDimension ? cells.get(selectedHead * dimensionCount + dimension) : undefined
  const maximumMagnitude = Math.max(0, ...captured.map((cell) => Math.abs(cell.value)))
  const feedbackKey = `${frame?.key ?? "uncaptured"}:${selectedHead}:${valueMode}`
  const feedback = copyFeedback?.key === feedbackKey ? copyFeedback : null
  const left = 38
  const top = 27
  const cellWidth = 4
  const cellHeight = 20
  const width = left + dimensionCount * cellWidth + 8
  const height = top + headCount * cellHeight + 26

  function pick(event: MouseEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect()
    const head = Math.floor(((event.clientY - bounds.top) / bounds.height * height - top) / cellHeight)
    const nextDimension = Math.floor(((event.clientX - bounds.left) / bounds.width * width - left) / cellWidth)
    if (head < 0 || head >= headCount || nextDimension < 0 || nextDimension >= dimensionCount) return
    onHeadChange(head)
    setDimensionInput(String(nextDimension))
  }

  async function copyHead() {
    if (!vector.length) return
    const snapshot = vector[0].snapshot
    const key = feedbackKey
    try {
      await navigator.clipboard.writeText(JSON.stringify({
        snapshot_ids: [...new Set(vector.map((cell) => cell.snapshot.snapshotId))],
        original_writer: snapshot.requestId, token_position: snapshot.tokenPosition,
        step_id: snapshot.stepId, operation_id: snapshot.operationId, layer_index: snapshot.layerIndex,
        kv_head: selectedHead, kind, dtype: snapshot.dtype,
        pool: snapshot.pool, cache_group: snapshot.cacheGroup, block_id: snapshot.blockId,
        allocation_generation: snapshot.allocationGeneration, slot: snapshot.slot,
        observation_boundary: snapshot.observationBoundary, logical_head_dimension: dimensions,
        dimensions: vector.map((cell) => cell.dimension), values: vector.map((cell) => cell.value),
        value_representation: hasFp8 ? valueMode === "raw" ? "raw_uint8_bits" : valueMode === "reconstructed" ? "decoded_fp8_times_recorded_scale" : "decoded_fp8_storage" : "native_tensor_values",
        quantization: hasFp8 ? {
          format: snapshot.quantization?.format,
          raw_bits: vector.map((cell) => cell.snapshot.quantization?.rawBits[cell.dimension - cell.snapshot.dimensionStart]),
          decoded_values: vector.map((cell) => cell.snapshot.values[cell.dimension - cell.snapshot.dimensionStart]),
          captured_scales: vector.map((cell) => cell.snapshot.quantization?.scale.value),
          reconstructed_values: vector.map((cell) => cell.snapshot.quantization?.reconstructedValues[cell.dimension - cell.snapshot.dimensionStart]),
        } : undefined,
      }, null, 2))
      setCopyFeedback({ key, failed: false, message: `Copied ${vector.length} captured ${kind} values from head ${selectedHead}.` })
    } catch {
      setCopyFeedback({ key, failed: true, message: "Clipboard unavailable. Select and copy the exact values from the table." })
      setView("table")
    }
  }

  if (!headCount || !dimensionCount) return <p className="m-0 text-pretty text-xs text-muted">KV-head and vector dimensions were not captured. A numerical grid cannot establish the missing shape.</p>

  return <section className="grid min-w-0 gap-3" aria-label="Recorded KV head and dimension values">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <p className="m-0 font-mono text-xs tabular-nums text-accent">{captured.length} / {headCount * dimensionCount} captured {kind} values</p>
      <Segmented<"heatmap" | "table"> label="Numerical KV display" value={view} onChange={setView} options={[{ value: "heatmap", label: "Heatmap" }, { value: "table", label: "Exact table" }]} />
    </div>
    <p className="m-0 text-pretty text-xs text-muted">Rows = {headCount} KV heads. Columns = {dimensionCount} components of one {kind} vector per head. These dimensions are not token positions or the model’s full hidden dimension.</p>
    {hasFp8 ? <>
      <Segmented<NumericalValueMode> label="FP8 numerical representation" value={valueMode} onChange={setValueMode} options={[{ value: "reconstructed", label: "Reconstructed" }, { value: "stored", label: "FP8 values" }, { value: "raw", label: "Raw bytes" }]} />
      <p className="m-0 text-pretty text-xs text-muted">{valueMode === "reconstructed" ? "Reconstructed KV = decoded FP8 value × scale captured on the same update stream. These retain quantization error and are not the original higher-precision inputs." : valueMode === "stored" ? "FP8 storage values decoded using the recorded E4M3FN or E5M2 encoding, before applying the captured scale." : "Unsigned byte bit patterns in physical cache storage. These integers are encodings, not KV magnitudes."}</p>
    </> : null}
    {view === "heatmap" ? <div className="border border-line bg-[#060a11] p-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="block h-auto w-full" role="img" aria-label={`${kind} heatmap: ${captured.length} retained values across ${headCount} KV heads and ${dimensionCount} vector dimensions. Hatched cells are not captured. Use the head and dimension controls for keyboard inspection.`} onClick={pick}>
        <defs><pattern id={`${id}-unknown`} width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#111923" /><path d="M-1 1L1-1M0 6L6 0M5 7L7 5" stroke="#52627a" strokeWidth="0.7" /></pattern></defs>
        <text x={left} y="14" fontSize="10" fill="#95a5bb" fontFamily="monospace">{kind} vector dimension (zero-based)</text>
        {Array.from({ length: Math.min(9, dimensionCount) }, (_, index) => Math.round(index * (dimensionCount - 1) / Math.max(1, Math.min(9, dimensionCount) - 1))).map((tick) => <text key={tick} x={left + tick * cellWidth} y={height - 6} textAnchor={tick === dimensionCount - 1 ? "end" : "start"} fontSize="10" fill="#95a5bb" fontFamily="monospace">{tick}</text>)}
        {Array.from({ length: headCount }, (_, head) => <text key={head} x={left - 7} y={top + head * cellHeight + 13} textAnchor="end" fontSize="10" fill={head === selectedHead ? "#6fdbef" : "#95a5bb"} fontFamily="monospace">H{head}</text>)}
        {Array.from({ length: headCount * dimensionCount }, (_, index) => {
          const cell = cells.get(index)
          const colors = valueMode === "raw" ? BYTE_COLORS : COLORS
          const level = cell ? valueMode === "raw" ? cell.value / 255 : maximumMagnitude ? (cell.value / maximumMagnitude + 1) / 2 : 0.5 : 0
          return <rect key={index} x={left + index % dimensionCount * cellWidth} y={top + Math.floor(index / dimensionCount) * cellHeight} width={cellWidth} height={cellHeight} fill={cell ? colors[Math.round(level * (colors.length - 1))] : `url(#${id}-unknown)`} stroke="#253442" strokeWidth="0.35" />
        })}
        {validDimension ? <rect x={left + dimension * cellWidth} y={top + selectedHead * cellHeight} width={cellWidth} height={cellHeight} fill="none" stroke="#f3f6fa" strokeWidth="1.2" pointerEvents="none" /> : null}
      </svg>
      <p className="mb-0 mt-2 text-pretty text-[11px] text-muted">Hatched = not captured, never zero. {captured.length ? valueMode === "raw" ? "Dark → cyan: byte 0 → byte 255. This fixed unsigned scale shows encoded bits, not KV magnitude." : `Amber → neutral → cyan: ${-maximumMagnitude} → 0 → ${maximumMagnitude}. Scale uses this snapshot’s recorded values.` : "No numerical values were retained at these coordinates."}</p>
    </div> : <div className="max-h-56 overflow-auto border border-line">
      <table className="w-full text-left font-mono text-xs tabular-nums"><caption className="px-2 py-2 text-left text-muted">Captured {kind} values only. Unlisted coordinates are not captured.</caption><thead><tr><th scope="col" className="px-2 py-1">Head / dimension</th><th scope="col" className="px-2 py-1">Exact value</th></tr></thead><tbody>{captured.map((cell) => <tr key={`${cell.head}:${cell.dimension}`} className={cn(cell.head === selectedHead && cell.dimension === dimension && "bg-accent/10")}><th scope="row" className="px-2 py-1 font-normal"><button type="button" className="min-h-6 border-0 bg-transparent p-0 text-xs text-accent" onClick={() => { onHeadChange(cell.head); setDimensionInput(String(cell.dimension)) }}>H{cell.head} · D{cell.dimension}</button></th><td className="select-text px-2 py-1 text-ink">{cell.value.toString()}</td></tr>)}</tbody></table>
      {!captured.length ? <p className="m-0 px-2 py-2 text-xs text-muted">No captured values at this token, layer and {kind} boundary.</p> : null}
    </div>}
    <label className="grid gap-1 text-xs text-muted" htmlFor={`${id}-dimension`}>{kind} vector dimension (0–{dimensionCount - 1})<input id={`${id}-dimension`} type="number" min={0} max={dimensionCount - 1} step={1} value={dimensionInput} onChange={(event) => setDimensionInput(event.target.value)} aria-invalid={!validDimension} aria-describedby={`${id}-exact`} className="min-w-0 font-mono text-xs tabular-nums" /></label>
    <div id={`${id}-exact`} aria-live="polite" aria-atomic="true" className="border border-line bg-[#060a11] px-2 py-2">
      <p className="m-0 break-words font-mono text-sm tabular-nums text-ink">{!validDimension ? `Choose a whole dimension from 0 to ${dimensionCount - 1}.` : selectedCell ? selectedCell.value.toString() : "Not captured"}</p>
      <p className="mb-0 mt-1 text-[11px] text-muted">Head {selectedHead} · dimension {validDimension ? dimension : "?"} · {kind}{selectedCell ? ` · ${selectedCell.snapshot.dtype} · ${selectedCell.snapshot.stepId}` : " · no retained numerical value"}</p>
      {selectedCell?.snapshot.quantization ? <p className="mb-0 mt-1 text-[11px] text-muted">{selectedCell.snapshot.quantization.format} · captured {selectedCell.snapshot.quantization.scale.mode.replace("_", " ")} scale {selectedCell.snapshot.quantization.scale.value.toString()} · raw byte {selectedCell.snapshot.quantization.rawBits[selectedCell.dimension - selectedCell.snapshot.dimensionStart]}</p> : null}
    </div>
    <div className="grid gap-1.5"><button type="button" className="justify-self-start text-xs" disabled={!vector.length} onClick={copyHead}>Copy captured head values{vector.length ? ` (${vector.length})` : ""}</button>{!vector.length ? <p className="m-0 text-xs text-muted">No captured {kind} values for head {selectedHead}; copying cannot supply an uncaptured vector.</p> : null}{feedback ? <p role={feedback.failed ? "alert" : "status"} className={cn("m-0 text-pretty text-xs", feedback.failed ? "text-[#e5687d]" : "text-accent")}>{feedback.message}</p> : null}</div>
    {conflictCount ? <p className="m-0 text-pretty text-xs text-muted">{conflictCount} coordinates have conflicting retained slices and are left unknown.</p> : null}
  </section>
}
