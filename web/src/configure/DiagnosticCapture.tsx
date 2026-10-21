import type { EngineSettings } from "../scenario/types.ts"
import type { FieldIssue } from "../scenario/validate.ts"
import { CheckboxField, Details, FieldError, FieldHint, inputClass } from "./ui.tsx"

export interface CaptureRequestChoice { id: string; label: string }

export function DiagnosticCapture({ settings, issues, requests = [], onSettings }: {
  settings: EngineSettings; issues: FieldIssue[]; requests?: CaptureRequestChoice[]
  onSettings: (settings: EngineSettings) => void
}) {
  const issueFor = (id: string) => issues.find((issue) => issue.fieldId === id)?.message ?? ""
  const numberField = (key: keyof EngineSettings, id: string, label: string, hint: string) => <label htmlFor={id} className="block text-xs font-medium text-slate-400">
    {label}
    <input id={id} className={`${inputClass} tabular-nums`} inputMode={["tensorTokenPositions", "tensorLayerIndices", "tensorHeadIndices"].includes(key) ? "text" : "numeric"} value={String(settings[key])}
      aria-invalid={Boolean(issueFor(id))} aria-describedby={`${id}-hint ${id}-error`}
      onChange={(event) => onSettings({ ...settings, [key]: event.target.value })} />
    <FieldHint id={`${id}-hint`}>{hint}</FieldHint><FieldError id={`${id}-error`} message={issueFor(id)} />
  </label>
  const requestField = (key: "profilingTriggerRequestId" | "tensorRequestId", id: string, label: string, automatic: string, hint: string) => <label htmlFor={id} className="block text-xs font-medium text-slate-400">
    {label}
    <select id={id} className={inputClass} value={settings[key]} aria-invalid={Boolean(issueFor(id))} aria-describedby={`${id}-hint ${id}-error`}
      onChange={(event) => onSettings({ ...settings, [key]: event.target.value })}>
      <option value="">{automatic}</option>
      {settings[key] && !requests.some((request) => request.id === settings[key]) ? <option value={settings[key]}>Request removed — choose another</option> : null}
      {requests.map((request) => <option key={request.id} value={request.id}>{request.label} ({request.id})</option>)}
    </select>
    <FieldHint id={`${id}-hint`}>{hint}</FieldHint><FieldError id={`${id}-error`} message={issueFor(id)} />
  </label>
  return <Details className="rounded-lg border border-slate-800 p-4" forceOpen={issues.some((issue) => issue.fieldId.startsWith("setting-profiling"))}>
    <summary className="cursor-pointer text-sm font-medium text-slate-200">Layer and KV diagnostics</summary>
    <div className="mt-3 grid items-start gap-4 md:grid-cols-2">
      <label htmlFor="setting-profiling-mode" className="block text-xs font-medium text-slate-400">Capture mode
        <select id="setting-profiling-mode" className={inputClass} value={settings.profilingMode} aria-invalid={Boolean(issueFor("setting-profiling-mode"))} aria-describedby="setting-profiling-hint setting-profiling-mode-error"
          onChange={(event) => onSettings({ ...settings, profilingMode: event.target.value as EngineSettings["profilingMode"] })}>
          <option value="off">Serving capture</option><option value="eager_diagnostic">Eager layer and KV diagnostic</option>
        </select>
        <FieldError id="setting-profiling-mode-error" message={issueFor("setting-profiling-mode")} />
      </label>
      {settings.profilingMode !== "off" ? numberField("profilingMaxSteps", "setting-profiling-steps", "Maximum captured scheduler steps", "1–16 steps; the default is eight.") : null}
    </div>
    <FieldHint id="setting-profiling-hint">Diagnostics use one GPU with dense Qwen3 and CUDA graphs disabled. Native GPU intervals describe shared batch operations, including supported layer cache writes. Recording and tensor readback add overhead.</FieldHint>
    {settings.profilingMode !== "off" ? <div className="mt-5 space-y-5 border-t border-slate-800 pt-5">
      <div className="grid items-start gap-4 md:grid-cols-2">
        {numberField("profilingStartStep", "setting-profiling-start", "Earliest scheduler step", "Step numbering starts at 1. Capture waits until this step and any selected arrival trigger.")}
        {requestField("profilingTriggerRequestId", "setting-profiling-trigger", "Request arrival trigger", "Start at the earliest step", "Wait for this request to be admitted before starting the diagnostic window.")}
      </div>
      <CheckboxField id="setting-profiling-tensors" checked={settings.tensorCapture} label="Capture selected numerical K/V slices"
        description="Choose targets before recording. Values omitted from a recording require a targeted rerun. Native BF16/FP16 slices are supported; other layouts report availability."
        onChange={(tensorCapture) => onSettings({ ...settings, tensorCapture })} />
      {settings.tensorCapture ? <div className="space-y-4">
        <div className="grid items-start gap-4 md:grid-cols-2">
          <label htmlFor="setting-profiling-tensor-preset" className="block text-xs font-medium text-slate-400">Tensor selection
            <select id="setting-profiling-tensor-preset" className={inputClass} value={settings.tensorPreset} aria-invalid={Boolean(issueFor("setting-profiling-tensor-preset"))} aria-describedby="setting-profiling-tensor-preset-hint setting-profiling-tensor-preset-error"
              onChange={(event) => onSettings({ ...settings, tensorPreset: event.target.value as EngineSettings["tensorPreset"] })}>
              <option value="small">Small learning preset</option><option value="custom">Custom selection</option>
            </select>
            <FieldHint id="setting-profiling-tensor-preset-hint">{settings.tensorPreset === "small" ? "Four input positions, first/middle/last layers, KV head 0, and the first 16 dimensions of K and V." : "Select zero-based positions, layers, heads and a bounded range of dimensions below."}</FieldHint>
            <FieldError id="setting-profiling-tensor-preset-error" message={issueFor("setting-profiling-tensor-preset")} />
          </label>
          {requestField("tensorRequestId", "setting-profiling-tensor-request", "Request to inspect", "First participating request", "Only selected request positions receive numerical snapshots.")}
        </div>
        {settings.tensorPreset === "custom" ? <div className="grid items-start gap-4 md:grid-cols-2">
          {numberField("tensorTokenPositions", "setting-profiling-tensor-positions", "Token positions", "Zero-based indices separated by commas or spaces; empty selects the first four participating positions.")}
          {numberField("tensorLayerIndices", "setting-profiling-tensor-layers", "Layer indices", "Zero-based indices; empty selects the first, middle, and last layer. Layer index 0 is displayed as Layer 1.")}
          {numberField("tensorHeadIndices", "setting-profiling-tensor-heads", "KV head indices", "Zero-based indices separated by commas or spaces.")}
          {numberField("tensorDimensionStart", "setting-profiling-tensor-dimension-start", "Starting K/V dimension", "Zero-based starting dimension within each head.")}
          {numberField("tensorDimensionCount", "setting-profiling-tensor-dimension-count", "Dimensions per K/V slice", "1–256 dimensions, limited by the observed head shape.")}
        </div> : null}
        {numberField("tensorMaxBytes", "setting-profiling-tensor-bytes", "Numerical capture byte limit", "64 bytes–16 MiB; default 1 MiB. Capture stops retaining numerical data when this limit is reached.")}
      </div> : null}
    </div> : null}
  </Details>
}
