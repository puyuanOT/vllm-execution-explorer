import type { MeasuredSceneProfile } from "../architecture/model.ts"
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, AudioLines, Box, ChevronRight, Maximize2, PanelRight, RotateCcw, RotateCw, Scan, Search, SquarePen, ZoomIn, ZoomOut } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { initialState, type RecordingManifest, type ReplayState, type TraceEvent } from "../../../replay/src/reduce.ts"
import { projectArchitecture, projectKvCache } from "../../../replay/src/learning.ts"
import { architectureFocusLabel, type ArchitectureFocus } from "../architecture/model.ts"
import { ArchitectureNavigation, PoolNavigation } from "./ArchitectureNavigation.tsx"
import { describeBlocks, type IndexedStep, type SampleProjection } from "../../../replay/src/timeline.ts"
import { workflowComponent, WORKFLOW_COMPONENTS } from "../explore/model.ts"
import type { WorkflowComponentId } from "../explore/model.ts"
import { cn } from "../lib/cn.ts"
import { ExecutionScene } from "../scene.tsx"
import type { CameraAction, CameraCommand } from "../scene.tsx"
import { Chip, Segmented, Sparkline } from "../ui/controls.tsx"
import { RequestSequencer } from "./RequestSequencer.tsx"
import type { TelemetryPoint } from "./telemetry.ts"
import type { Selection, WorkspaceView } from "./types.ts"

const EMPTY_REPLAY = initialState()

export type EvidenceKind = "synthetic" | "gpu" | "draft" | "recorded"

export function ExecutionWorkspace({
  samples,
  state,
  manifest,
  measuredProfile,
  teachingProgress,
  onTeachingProgress,
  trackedRequestId,
  trackedTokenPosition,
  requestOrder,
  colors,
  selection,
  highlightedIds,
  view,
  onView,
  animate,
  timeBasisLabel,
  evidence,
  evidenceKind,
  warning,
  submittedNote,
  currentEvent,
  telemetry,
  allSteps,
  sequence,
  onEditAsNew,
  onOpenInspector,
  onSelect,
}: {
  samples: SampleProjection[]
  state: ReplayState | null
  manifest: RecordingManifest | null
  measuredProfile?: MeasuredSceneProfile
  teachingProgress?: number
  onTeachingProgress: (value: number) => void
  trackedRequestId: string | null
  trackedTokenPosition: number | null
  requestOrder: string[]
  colors: Record<string, string>
  selection: Selection | null
  highlightedIds: Set<string>
  highlightedStepIds: Set<string>
  view: WorkspaceView
  onView: (view: WorkspaceView) => void
  animate: boolean
  timeBasisLabel: string
  evidence: string
  evidenceKind: EvidenceKind
  warning: string
  submittedNote: string
  currentEvent: TraceEvent | null
  telemetry: TelemetryPoint[]
  allSteps: IndexedStep[]
  sequence: number
  onEditAsNew?: () => void
  onOpenInspector?: () => void
  onSelect: (selection: Selection) => void
}) {
  const [focus, setFocus] = useState<WorkflowComponentId | null>(null)
  const [architectureFocus, setArchitectureFocus] = useState<ArchitectureFocus | null>(null)
  const [cameraCommand, setCameraCommand] = useState<CameraCommand | null>(null)
  const selectedComponent = selection?.kind === "component" ? selection.id : null
  const replay = state ?? EMPTY_REPLAY
  const architecture = useMemo(() => projectArchitecture(manifest), [manifest])
  const cache = useMemo(() => projectKvCache(replay, manifest), [manifest, replay])
  const labels = useMemo(() => new Map(samples.map((item) => [item.sample.id, item.sample.label])), [samples])

  useEffect(() => {
    setFocus(null)
    setArchitectureFocus(null)
  }, [requestOrder])

  const measuredMode = Boolean(measuredProfile)
  useEffect(() => {
    if (measuredMode) { setFocus("model-execution"); setArchitectureFocus({ kind: "model" }) }
  }, [measuredMode, requestOrder])

  function selectComponent(component: WorkflowComponentId) {
    onSelect({ kind: "component", id: component })
  }

  function exploreComponent(component: WorkflowComponentId) {
    selectComponent(component)
    setFocus(component)
    setArchitectureFocus(component === "model-execution" ? { kind: "model" } : null)
  }

  function exploreArchitecture(next: ArchitectureFocus) {
    setFocus("model-execution")
    setArchitectureFocus(next)
  }

  function backToOverview() { setFocus(null); setArchitectureFocus(null) }

  function moveCamera(action: CameraAction) {
    setCameraCommand((current) => ({ id: (current?.id ?? 0) + 1, action }))
  }

  const title = view === "spatial"
    ? architectureFocus ? architectureFocusLabel(architectureFocus) : focus ? workflowComponent(focus).label : "Model & serving architecture"
    : "Request lanes"
  const latestStep = replay.steps.at(-1) ?? null

  return (
    <section className="flex min-h-0 min-w-0 flex-col bg-canvas" aria-label="Execution explorer">
      <header className="flex min-h-[46px] flex-wrap items-center gap-x-3 gap-y-2 border-b border-line bg-[#060a11] px-3.5 py-1.5">
        <div className="flex min-w-fit flex-1 items-baseline gap-3">
          <h1 className="shrink-0 text-[15px] font-semibold text-ink">{title}</h1>
          <p className="m-0 hidden w-0 min-w-0 flex-1 truncate text-xs text-faint xl:block" title={timeBasisLabel}>{timeBasisLabel}</p>
        </div>
        <RequestFinder samples={samples} onSelect={onSelect} />
        <Segmented<WorkspaceView>
          label="Execution view"
          value={view}
          onChange={onView}
          options={[
            { value: "spatial", label: <><Box className="size-3.5" aria-hidden="true" />3D architecture</> },
            { value: "timeline", label: <><AudioLines className="size-3.5" aria-hidden="true" />Request lanes</> },
          ]}
        />
        {onEditAsNew ? (
          <button type="button" className="inline-flex min-h-8 items-center gap-2 border-line-strong bg-transparent px-3 font-mono text-[11px] uppercase tracking-[0.06em] text-ink" onClick={onEditAsNew}>
            <SquarePen className="size-3.5" aria-hidden="true" />Edit as new sequence
          </button>
        ) : null}
        {onOpenInspector ? (
          <button type="button" className="inline-flex min-h-8 items-center gap-2 border-line-strong bg-transparent px-3 font-mono text-[11px] uppercase tracking-[0.06em] text-ink" onClick={onOpenInspector}>
            <PanelRight className="size-3.5" aria-hidden="true" />Open inspector
          </button>
        ) : null}
      </header>

      {submittedNote ? <p className="m-0 border-b border-line bg-[#060a11] px-3.5 py-1 text-xs text-muted">{submittedNote}</p> : null}
      {warning ? (
        <p className="m-0 border-b border-amber-700/50 bg-amber-400/[0.06] px-3.5 py-1.5 text-pretty text-xs text-amber-200" role="status">
          {warning}
        </p>
      ) : null}

      {view === "timeline" ? (
        <div className="flex min-h-0 flex-1 flex-col gap-2.5 p-3">
          <div className="flex flex-wrap items-center gap-3">
            <EvidenceChip kind={evidenceKind} evidence={evidence} />
          </div>
          <RequestSequencer
            samples={samples}
            steps={replay.steps}
            allSteps={allSteps}
            colors={colors}
            selection={selection}
            highlightedIds={highlightedIds}
            sequence={sequence}
            onSelect={onSelect}
          />
        </div>
      ) : (
        <div className="relative flex min-h-0 flex-1 flex-col">
          <p className="sr-only">
            The visual overview contains request intake, scheduler, KV-cache manager, model execution, and output.
            Use the component controls below the canvas for equivalent keyboard access.
          </p>

          <div className="scene-viewport relative min-h-[62vh] flex-1 overflow-hidden sm:min-h-72">
            <div className="absolute inset-0">
              <ExecutionScene
                state={replay}
                manifest={manifest}
                measuredProfile={measuredProfile}
                architectureFocus={architectureFocus}
                teachingProgress={teachingProgress}
                trackedRequestId={trackedRequestId}
                trackedTokenPosition={trackedTokenPosition}
                kvCache={cache}
                requestOrder={requestOrder}
                selection={selection}
                animate={animate}
                focus={focus}
                cameraCommand={cameraCommand}
                onSelect={onSelect}
                onExplore={exploreComponent}
                onArchitectureExplore={exploreArchitecture}
              />
            </div>

            <div className="pointer-events-none absolute left-3 right-14 top-3 z-10 flex items-start gap-3">
              <div className="pointer-events-auto grid shrink-0 gap-1.5">
                {focus ? (
                  <nav className="cp-hud flex h-7 items-center gap-1.5 px-2 font-mono text-[11px] tracking-[0.04em]" aria-label="Component breadcrumb">
                    <Scan className="size-3.5 text-faint" aria-hidden="true" />
                    <button type="button" className="min-h-0 border-0 bg-transparent px-1 py-0 text-muted hover:text-ink" onClick={backToOverview}>Overview</button>
                    <ChevronRight className="size-3.5 text-[#3a4960]" aria-hidden="true" />
                    {architectureFocus && architectureFocus.kind !== "model" ? (
                      <>
                        <button type="button" className="min-h-0 border-0 bg-transparent px-1 py-0 text-muted hover:text-ink" onClick={() => setArchitectureFocus({ kind: "model" })}>{workflowComponent(focus).label}</button>
                        <ChevronRight className="size-3.5 text-[#3a4960]" aria-hidden="true" />
                        <span className="max-w-56 truncate text-accent" aria-current="page">{architectureFocusLabel(architectureFocus)}</span>
                      </>
                    ) : <span className="max-w-56 truncate text-accent" aria-current="page">{workflowComponent(focus).label}</span>}
                  </nav>
                ) : (
                  <p className="cp-hud m-0 flex h-7 items-center gap-2 px-2.5 font-mono text-[11px] text-ink"><Scan className="size-3.5 text-faint" aria-hidden="true" />Overview</p>
                )}
                <p className="m-0 hidden text-[11.5px] text-faint md:block"><b className="font-medium text-muted">Click</b> to inspect · <b className="font-medium text-muted">double-click</b> to drill in · <b className="font-medium text-muted">drag</b> to orbit</p>
              </div>
              <EventTicker event={currentEvent} labels={labels} />
            </div>

            <div className="absolute right-3 top-3 z-10 grid grid-cols-5 gap-px border border-line-strong bg-surface p-[3px] sm:grid-cols-2" role="group" aria-label="3D camera controls">
              {([
                ["orbit-left", "Orbit camera left", RotateCcw],
                ["orbit-right", "Orbit camera right", RotateCw],
                ["zoom-in", "Zoom camera in", ZoomIn],
                ["zoom-out", "Zoom camera out", ZoomOut],
                ["pan-left", "Pan camera left", ArrowLeft],
                ["pan-right", "Pan camera right", ArrowRight],
                ["pan-up", "Pan camera up", ArrowUp],
                ["pan-down", "Pan camera down", ArrowDown],
                ["reset", "Reset camera", Maximize2],
              ] as const).map((item) => {
                const [action, name, Icon] = item
                const hint = action.startsWith("pan") ? " · or right-drag / Ctrl-drag" : action.startsWith("orbit") ? " · or left-drag" : action.startsWith("zoom") ? " · or scroll" : ""
                return (
                  <button
                    key={action}
                    type="button"
                    className="grid size-7 min-h-0 place-items-center border-0 bg-transparent p-0 text-faint hover:text-ink"
                    aria-label={name}
                    title={`${name}${hint}`}
                    onClick={() => moveCamera(action)}
                  >
                    <Icon className="size-4" aria-hidden="true" />
                  </button>
                )
              })}
            </div>

            <div className="pointer-events-none absolute bottom-3 left-3 right-3 z-10 flex flex-wrap items-end justify-between gap-2">
              <SceneLegend focus={focus} state={replay} labels={labels} colors={colors} cache={cache} latestStep={latestStep} />
              <Watermark kind={evidenceKind} evidence={evidence} />
            </div>
          </div>

          {focus ? (
            <div className="max-h-56 overflow-auto border-t border-line bg-[#060a11] px-3.5 py-2.5">
              <details open className="group">
                <summary className="cursor-pointer font-mono text-[10.5px] uppercase tracking-[0.1em] text-muted outline-none focus-visible:text-ink">
                  {focus === "model-execution" ? "Model structure controls" : focus === "kv-cache" ? "Find a physical KV block" : "Recorded objects"}
                </summary>
                <div className="pt-2.5">
                  {focus === "model-execution" ? <ArchitectureNavigation architecture={architecture} focus={architectureFocus} onExplore={exploreArchitecture} onSelect={onSelect} progress={teachingProgress} onProgress={onTeachingProgress} /> : focus === "kv-cache" ? <PoolNavigation cache={cache} onSelect={onSelect} /> : <FocusedObjectBar component={focus} state={state} samples={samples} selection={selection} onSelect={onSelect} />}
                </div>
              </details>
            </div>
          ) : (
            <Telemetry telemetry={telemetry} selected={selectedComponent} state={state} onSelect={selectComponent} onExplore={exploreComponent} />
          )}
        </div>
      )}
    </section>
  )
}

function EventTicker({ event, labels }: { event: TraceEvent | null; labels: Map<string, string> }) {
  if (!event) return null
  const requestId = typeof event.request_id === "string" ? event.request_id : null
  const stepId = typeof event.step_id === "string" ? event.step_id : null
  const blockId = typeof event.block_id === "number" ? event.block_id : null
  return (
    <div className="mx-auto hidden min-w-0 max-w-[min(32rem,60%)] items-center gap-2.5 border border-line-strong bg-surface px-3 py-1 font-mono text-[11px] text-muted xl:flex" aria-hidden="true">
      <span className="shrink-0 font-semibold text-ink">EVT {event.sequence}</span>
      <span className="h-3 w-px shrink-0 bg-line-strong" />
      <span className="truncate">{event.event_type}</span>
      {requestId ? <span className="shrink-0 text-ink">{labels.get(requestId) ?? requestId}{blockId != null ? ` → block ${blockId}` : ""}</span> : null}
      {stepId ? <><span className="h-3 w-px shrink-0 bg-line-strong" /><span className="shrink-0">{stepId}</span></> : null}
    </div>
  )
}

function SceneLegend({ focus, state, labels, colors, cache, latestStep }: {
  focus: WorkflowComponentId | null
  state: ReplayState
  labels: Map<string, string>
  colors: Record<string, string>
  cache: ReturnType<typeof projectKvCache>
  latestStep: ReplayState["steps"][number] | null
}) {
  const chip = "pointer-events-auto flex min-h-7 max-w-full flex-wrap items-center gap-x-3 gap-y-1 border border-line-strong bg-surface/95 px-2.5 py-1 font-mono text-[10.5px] text-muted"
  if (focus === "kv-cache") {
    const blocks = cache.pools.flatMap((pool) => pool.blocks)
    if (!blocks.length) return <span />
    const referenced = blocks.filter((block) => !block.isNull && block.refCount > 0).length
    const cached = blocks.filter((block) => !block.isNull && block.refCount === 0 && block.hasHash).length
    const nulls = blocks.filter((block) => block.isNull).length
    const free = blocks.length - referenced - cached - nulls
    return (
      <div className={chip}>
        <span className="flex items-center gap-1.5"><i className="size-2 bg-[#6aa6e8]" />Referenced <b className="font-medium text-ink">{referenced.toLocaleString()}</b></span>
        <span className="flex items-center gap-1.5"><i className="size-2 border border-amber-400" />Cached, 0 refs <b className="font-medium text-ink">{cached.toLocaleString()}</b></span>
        <span className="flex items-center gap-1.5"><i className="size-2 border border-[#23403a] bg-[#122220]" />Free <b className="font-medium text-ink">{free.toLocaleString()}</b></span>
        <span className="flex items-center gap-1.5"><i className="cp-hatch size-2 border border-[#45556f]" />Null <b className="font-medium text-ink">{nulls.toLocaleString()}</b></span>
        {cache.status === "unknown" ? <span className="text-amber-300">Trace gap · current slot state unknown</span> : null}
      </div>
    )
  }
  if (focus === "model-execution") {
    return (
      <div className={chip}>
        <span className="flex items-center gap-1.5"><i className="size-2 bg-accent" />Selected</span>
        <span className="flex items-center gap-1.5"><i className="size-2 border border-dashed border-explain" />Walkthrough position</span>
        <span className="flex items-center gap-1.5"><i className="size-2 bg-amber-400" />Profiler activity</span>
        <span className="flex items-center gap-1.5"><i className="cp-hatch size-2 border border-[#45556f]" />Tensor values not captured</span>
      </div>
    )
  }
  if (!latestStep) return <span />
  const waiting = latestStep.queueAfter?.waiting.length ?? null
  return (
    <div className={chip}>
      <span className="text-faint">In {latestStep.id}</span>
      {latestStep.requests.slice(0, 6).map((request) => (
        <span key={request.requestId} className="flex items-center gap-1.5"><i className="size-2" style={{ background: colors[request.requestId] ?? "#7a8799" }} />{labels.get(request.requestId) ?? request.requestId}</span>
      ))}
      {latestStep.requests.length > 6 ? <span>+{latestStep.requests.length - 6} more</span> : null}
      {waiting != null ? <span className="text-faint">{waiting} waiting</span> : null}
      {state.trust === "unknown" ? <span className="text-amber-300">State after the trace gap is unknown</span> : null}
    </div>
  )
}

function Watermark({ kind, evidence }: { kind: EvidenceKind; evidence: string }) {
  const tone = kind === "synthetic" ? "text-synthetic" : kind === "gpu" ? "text-emerald-400" : kind === "draft" ? "text-amber-300" : "text-muted"
  return (
    <p className={cn("m-0 ml-auto flex max-w-[min(30rem,100%)] items-center gap-2.5 text-right font-mono text-[9.5px] uppercase tracking-[0.08em]", tone)}>
      {kind === "synthetic" ? <i className="h-1.5 w-7 shrink-0 bg-[repeating-linear-gradient(135deg,currentColor_0_2px,transparent_2px_5px)] opacity-70" aria-hidden="true" /> : <i className="h-px w-7 shrink-0 bg-current opacity-60" aria-hidden="true" />}
      <span className="text-pretty">{evidence}</span>
    </p>
  )
}

function EvidenceChip({ kind, evidence }: { kind: EvidenceKind; evidence: string }) {
  return <Chip tone={kind === "synthetic" ? "synthetic" : kind === "gpu" ? "gpu" : kind === "draft" ? "warning" : undefined}><span>{evidence}</span></Chip>
}

function RequestFinder({ samples, onSelect }: { samples: SampleProjection[]; onSelect: (selection: Selection) => void }) {
  const [query, setQuery] = useState("")
  const [open, setOpen] = useState(false)
  const normalized = query.trim().toLocaleLowerCase()
  const matches = (normalized ? samples.filter((sample) => (
    sample.sample.label.toLocaleLowerCase().includes(normalized)
    || sample.sample.id.toLocaleLowerCase().includes(normalized)
    || sample.phaseLabel.toLocaleLowerCase().includes(normalized)
  )) : samples).slice(0, 8)
  function choose(id: string) {
    onSelect({ kind: "request", id })
    setOpen(false)
  }
  return (
    <div className="relative">
      <label className="relative block">
        <span className="sr-only">Find a request</span>
        <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-faint" aria-hidden="true" />
        <input
          type="search"
          className="h-8 w-44 rounded border-line-strong bg-surface py-0 pl-7 pr-2 font-mono text-[11px] text-ink placeholder:text-faint"
          placeholder={`Find request (${samples.length})`}
          value={query}
          aria-expanded={open}
          aria-controls="request-finder-results"
          onFocus={() => setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          onChange={(event) => { setQuery(event.target.value); setOpen(true) }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && matches[0]) { event.preventDefault(); choose(matches[0].sample.id) }
            if (event.key === "Escape") { setQuery(""); setOpen(false) }
          }}
        />
      </label>
      {open ? (
        <ul id="request-finder-results" className="absolute right-0 top-full z-30 mt-1 grid max-h-72 w-72 list-none gap-px overflow-auto border border-line-strong bg-surface p-1 shadow-[0_18px_40px_-20px_rgba(0,0,0,0.9)]">
          {matches.length ? matches.map((sample) => (
            <li key={sample.sample.id}>
              <button
                type="button"
                className="flex w-full items-center justify-between gap-3 border-0 bg-transparent px-2 py-1.5 text-left text-sm text-muted hover:bg-raised hover:text-ink"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => choose(sample.sample.id)}
              >
                <span className="truncate font-mono text-xs text-ink">{sample.sample.label}</span>
                <span className="truncate text-[11px]">{sample.phaseLabel}</span>
              </button>
            </li>
          )) : <li className="px-2 py-1.5 text-xs text-muted">No request matches. Clear the search to see every request.</li>}
        </ul>
      ) : null}
    </div>
  )
}

function Telemetry({ telemetry, selected, state, onSelect, onExplore }: {
  telemetry: TelemetryPoint[]
  selected: WorkflowComponentId | null
  state: ReplayState | null
  onSelect: (component: WorkflowComponentId) => void
  onExplore: (component: WorkflowComponentId) => void
}) {
  const current = telemetry.at(-1) ?? null
  const values: Record<WorkflowComponentId, { value: string; unit: string; series: number[]; max?: number }> = {
    "request-intake": { value: current ? String(current.requests).padStart(2, "0") : "—", unit: "req", series: telemetry.map((point) => point.requests) },
    scheduler: { value: current ? String(current.steps).padStart(2, "0") : "—", unit: "steps", series: telemetry.map((point) => point.steps) },
    "kv-cache": {
      value: current ? current.capacity != null ? `${current.referenced.toLocaleString()}/${current.capacity.toLocaleString()}` : current.referenced.toLocaleString() : "—",
      unit: "ref",
      series: telemetry.map((point) => point.referenced),
      max: current?.capacity ?? undefined,
    },
    "model-execution": { value: current ? `${current.results}/${current.submissions}` : "—", unit: "results", series: telemetry.map((point) => point.results) },
    output: { value: current ? current.accepted.toLocaleString() : "—", unit: "tokens", series: telemetry.map((point) => point.accepted) },
  }
  return (
    <ol className="grid list-none grid-cols-2 border-t border-line bg-[#060a11] p-0 sm:grid-cols-3 lg:grid-cols-5" aria-label="Workflow components">
      {WORKFLOW_COMPONENTS.map((component) => {
        const item = values[component.id]
        const active = selected === component.id
        return (
          <li key={component.id} className={cn("relative min-w-0 border-l border-line first:border-l-0", active && "bg-[linear-gradient(180deg,rgba(63,205,230,0.06),transparent_70%)] shadow-[inset_0_2px_0_theme(colors.accent.400)]")}>
            <div className="flex min-w-0 items-stretch">
              <button
                type="button"
                className="grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] grid-rows-[auto_auto_auto] items-end gap-x-2 rounded-none border-0 bg-transparent px-3.5 pb-2 pt-2.5 text-left"
                aria-pressed={active}
                onClick={() => onSelect(component.id)}
                onDoubleClick={() => onExplore(component.id)}
              >
                <span className={cn("col-span-2 mb-1 truncate text-xs", active ? "text-ink" : "text-muted")}>{component.label}</span>
                <span className="truncate font-mono text-[20px] font-medium leading-none tracking-[-0.02em] text-ink">{item.value}</span>
                <span className="row-span-2 self-end"><Sparkline values={state ? item.series : []} max={item.max} active={active} width={64} height={26} /></span>
                <span className="mt-1 truncate font-mono text-[9.5px] uppercase tracking-[0.1em] text-faint">{item.unit}</span>
              </button>
              <button
                type="button"
                className="grid w-9 min-h-0 place-items-center self-start rounded-none border-0 bg-transparent p-0 pt-2.5 text-faint hover:text-ink"
                aria-label={`Explore ${component.label}`}
                title={`Explore ${component.label}`}
                onClick={() => onExplore(component.id)}
              >
                <Maximize2 className="size-3.5" aria-hidden="true" />
              </button>
            </div>
          </li>
        )
      })}
    </ol>
  )
}

function FocusedObjectBar({
  component,
  state,
  samples,
  selection,
  onSelect,
}: {
  component: WorkflowComponentId
  state: ReplayState | null
  samples: SampleProjection[]
  selection: Selection | null
  onSelect: (selection: Selection) => void
}) {
  const definition = workflowComponent(component)
  const objects = focusedObjects(component, state, samples)
  return (
    <div>
      <p className="m-0 text-pretty text-xs text-muted">{definition.description}</p>
      {objects.length > 0 ? (
        <ul className="mt-2 flex list-none gap-2 overflow-x-auto p-0 pb-1" aria-label={`${definition.label} recorded objects`}>
          {objects.map((object) => (
            <li key={object.key}>
              <button
                type="button"
                className={cn(
                  "min-w-40 border-line-strong bg-raised px-2.5 py-1.5 text-left text-sm",
                  sameSelection(selection, object.selection) && "border-accent-400/50",
                )}
                onClick={() => onSelect(object.selection)}
              >
                <span className="block truncate font-mono text-xs text-ink">{object.label}</span>
                <span className="block truncate text-xs tabular-nums text-muted">{object.detail}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="m-0 mt-2 flex items-center gap-2 text-pretty text-xs text-muted">
          <Box className="size-4" aria-hidden="true" /> No recorded object has reached this playhead yet. Play or step forward to see it.
        </p>
      )}
    </div>
  )
}

interface FocusedObject {
  key: string
  label: string
  detail: string
  selection: Selection
}

function focusedObjects(
  component: WorkflowComponentId,
  state: ReplayState | null,
  samples: SampleProjection[],
): FocusedObject[] {
  if (component === "request-intake" || component === "output") {
    return samples.map((sample) => ({
      key: sample.sample.id,
      label: sample.sample.label,
      detail: component === "output"
        ? `${sample.acceptedTokenCount ?? "—"} accepted · ${sample.phaseLabel}`
        : sample.phaseLabel,
      selection: { kind: "request", id: sample.sample.id },
    }))
  }
  if (!state) return []
  if (component === "kv-cache") {
    return describeBlocks(state).map((block) => ({
      key: block.key,
      label: `${block.pool} block ${block.blockId}`,
      detail: `generation ${block.generation} · refs ${block.refCount}`,
      selection: { kind: "block", key: block.key },
    }))
  }
  const steps = component === "model-execution"
    ? state.steps.filter((step) => step.submittedSequence != null)
    : state.steps
  return steps.map((step) => ({
    key: `${step.id}-${step.sequence}`,
    label: step.id,
    detail: `${step.totalScheduledTokens} tokens · ${step.status.replaceAll("_", " ")}`,
    selection: { kind: "step", id: step.id },
  }))
}

function sameSelection(left: Selection | null, right: Selection): boolean {
  if (!left || left.kind !== right.kind) return false
  if (left.kind === "block" && right.kind === "block") return left.key === right.key
  if (left.kind === "component" && right.kind === "component") return left.id === right.id
  if (left.kind === "request" && right.kind === "request") return left.id === right.id
  if (left.kind === "step" && right.kind === "step") return left.id === right.id
  return false
}
