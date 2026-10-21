import { Collapsible } from "@base-ui/react/collapsible"
import { ChevronRight } from "lucide-react"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import { modelIterationSummary, projectArchitecture, type MeasuredSceneProfile } from "../architecture/model.ts"
import { observedRequestIds, requestLabel } from "../explore/model.ts"
import { requestColor } from "../theme.ts"
import { measuredLayerActivity } from "./layerActivity.ts"
import { selectedLayerIndex } from "./iterationProfile.ts"
import type { Selection } from "./types.ts"

/** Batch evidence lives outside the canvas so camera movement never hides the model. */
export function ModelIterationBar({ state, manifest, profile, activeStepId, requestOrder, requestLabels, selection, onSelectRequest }: {
  state: ReplayState
  manifest: RecordingManifest | null
  profile?: MeasuredSceneProfile
  activeStepId?: string | null
  requestOrder: string[]
  requestLabels: ReadonlyMap<string, string>
  selection?: Selection | null
  onSelectRequest: (requestId: string) => void
}) {
  const summary = modelIterationSummary(state, profile, activeStepId)
  const observed = observedRequestIds(state, requestOrder)
  const architecture = projectArchitecture(manifest)
  const activity = profile ? measuredLayerActivity(profile, architecture) : []
  const inspectedLayer = selectedLayerIndex(selection ?? null)
  const detail = summary.clock === "profiler" ? "GPU activity · native packed rows · padding excluded" : null
  // Only an explicit disclosure toggle changes height; playback content never resizes the canvas.
  return <Collapsible.Root defaultOpen={false} render={<section />} className="shrink-0 border-b border-line bg-surface text-xs tabular-nums" aria-label="Intake requests and current model iteration">
    <div className="flex h-10 items-center gap-3 overflow-hidden px-3.5">
      <Collapsible.Trigger className="group flex shrink-0 items-center gap-1.5 border-0 bg-transparent px-0 text-muted hover:text-ink">
        <ChevronRight className="size-3.5 group-data-[panel-open]:rotate-90" aria-hidden="true" />Iteration details
      </Collapsible.Trigger>
      <span className="shrink-0 font-mono text-ink">{summary.label}</span>
      {observed.length ? <>
        <span className="shrink-0 text-muted">{profile ? "Intake · host replay" : "Intake"}</span>
        <ol className="m-0 flex min-w-0 list-none gap-2 overflow-x-auto p-0" aria-label="Requests observed in Intake">
          {observed.map((id) => <li key={id}>
            <button type="button" className="flex items-center gap-2 whitespace-nowrap border border-line bg-canvas px-2 py-1 text-ink" onClick={() => onSelectRequest(id)}>
              <i className="size-2.5 shrink-0" style={{ background: requestColor(requestOrder, id) }} aria-hidden="true" />
              {requestLabel(requestLabels, id)}
            </button>
          </li>)}
        </ol>
      </> : null}
    </div>
    <Collapsible.Panel className="h-28 overflow-y-auto border-t border-line px-3.5 py-2 [scrollbar-gutter:stable]" aria-label="Iteration details" tabIndex={0}>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
          <p className="m-0 font-mono font-semibold text-ink">{summary.label}</p>
          {summary.known ? <p className="m-0 text-muted">{summary.participants.length} participating request{summary.participants.length === 1 ? "" : "s"} · {summary.positions} input position{summary.positions === 1 ? "" : "s"}</p>
            : state.trust === "unknown" && !profile ? <p className="m-0 text-muted">Participation unknown after trace gap</p> : null}
        <p className="m-0 text-muted">{summary.clock === "profiler" ? "Captured packed inputs this iteration" : "Tokens scheduled this iteration"}</p>
        {detail ? <p className="m-0 text-[11px] text-faint">{detail}</p> : null}
        </div>
      </div>
      {profile ? <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-2" role="group" aria-label="Layer execution activity">
        <span className="font-mono text-[11px] text-muted">{profile.evidenceKind === "fixture" ? "Fixture activity" : "GPU activity"}</span>
        {activity.length ? <ul className="m-0 flex list-none flex-wrap gap-2 p-0" aria-label="Active mapped layers">
          {activity.map(({ layerIndex, detail }) => <li key={layerIndex} className="border border-amber-400/60 bg-canvas px-2 py-1 text-amber-400">
            <b className="font-mono">Layer {layerIndex + 1}</b><span className="ml-2">{detail}</span>
          </li>)}
        </ul> : <p className="m-0 font-medium text-ink">No mapped layer activity</p>}
        <span className="text-[11px] text-faint">Amber = active</span>
        {inspectedLayer != null ? <span className="border border-accent/50 px-2 py-1 text-accent">Inspecting layer {inspectedLayer + 1} · cyan rim</span> : null}
      </div> : null}
      {summary.participants.length ? <ol className="m-0 mt-2 flex list-none flex-wrap gap-2 p-0" aria-label="Requests in the model iteration">
        {summary.participants.map((request) => <li key={request.requestId} className="flex min-w-0 items-stretch border border-line bg-canvas">
          <i className="w-1 shrink-0" style={{ background: requestColor(requestOrder, request.requestId) }} aria-hidden="true" />
          <button type="button" className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1 px-2 py-1.5 text-left" onClick={() => onSelectRequest(request.requestId)}>
            <b className="font-medium text-ink">{requestLabel(requestLabels, request.requestId)}</b>
            <span className="text-muted">{request.prefill == null ? `${request.positions} input${request.positions === 1 ? "" : "s"}` : `Prefill: ${request.prefill} · Decode: ${request.decode}`}</span>
            <span className="max-w-48 truncate text-faint" title="Request token positions, zero-based">Positions {request.ranges.map((range) => range.start === range.end ? range.start : `${range.start}–${range.end}`).join(", ")}</span>
            {state.requests[request.requestId]?.modelRouteKind === "lora_adapter" ? <span className="text-faint">LoRA: {state.requests[request.requestId].adapterId ?? "recorded adapter"}</span> : null}
          </button>
        </li>)}
      </ol> : null}
    </Collapsible.Panel>
  </Collapsible.Root>
}
