import type { MeasuredSceneProfile } from "../architecture/model.ts"
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Box, ChevronRight, Maximize2, RotateCcw, Search, ZoomIn, ZoomOut } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { initialState, type RecordingManifest, type ReplayState } from "../../../replay/src/reduce.ts"
import { projectArchitecture, projectKvCache } from "../../../replay/src/learning.ts"
import { architectureFocusLabel, type ArchitectureFocus } from "../architecture/model.ts"
import { ArchitectureNavigation, PoolNavigation } from "./ArchitectureNavigation.tsx"
import { describeBlocks, type SampleProjection } from "../../../replay/src/timeline.ts"
import { componentSnapshot, workflowComponent, WORKFLOW_COMPONENTS } from "../explore/model.ts"
import type { WorkflowComponentId } from "../explore/model.ts"
import { cn } from "../lib/cn.ts"
import { ExecutionScene } from "../scene.tsx"
import type { CameraAction, CameraCommand } from "../scene.tsx"
import { ExecutionBatchStrip } from "./ExecutionBatchStrip.tsx"
import { SampleLanes } from "./SampleLanes.tsx"
import type { Selection, WorkspaceView } from "./types.ts"

const EMPTY_REPLAY = initialState()

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
  highlightedStepIds,
  view,
  onView,
  animate,
  timeBasisLabel,
  evidence,
  warning,
  submittedNote,
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
  warning: string
  submittedNote: string
  onSelect: (selection: Selection) => void
}) {
  const [focus, setFocus] = useState<WorkflowComponentId | null>(null)
  const [architectureFocus, setArchitectureFocus] = useState<ArchitectureFocus | null>(null)
  const [cameraCommand, setCameraCommand] = useState<CameraCommand | null>(null)
  const [requestQuery, setRequestQuery] = useState("")
  const selectedComponent = selection?.kind === "component" ? selection.id : null
  const replay = state ?? EMPTY_REPLAY
  const architecture = useMemo(() => projectArchitecture(manifest), [manifest])
  const cache = useMemo(() => projectKvCache(replay, manifest), [manifest, replay])
  const visibleRequests = useMemo(() => {
    const query = requestQuery.trim().toLocaleLowerCase()
    if (!query) return samples
    return samples.filter((sample) => (
      sample.sample.label.toLocaleLowerCase().includes(query)
      || sample.sample.id.toLocaleLowerCase().includes(query)
      || sample.phaseLabel.toLocaleLowerCase().includes(query)
    ))
  }, [requestQuery, samples])

  useEffect(() => {
    setFocus(null)
    setArchitectureFocus(null)
    setRequestQuery("")
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

  return (
    <section className="workspace-main min-h-0 min-w-0 bg-canvas" aria-label="Execution explorer">
      <header className="workspace-head border-b border-line bg-surface px-4 py-3">
        <div className="min-w-0">
          <h1 className="text-balance text-base font-semibold text-ink">
            {view === "spatial" ? (architectureFocus ? architectureFocusLabel(architectureFocus) : focus ? workflowComponent(focus).label : "Model and serving architecture") : "Request lanes"}
          </h1>
          <p className="meta truncate text-pretty text-xs text-muted">{timeBasisLabel}</p>
          {submittedNote ? <p className="meta truncate text-pretty text-xs text-muted">{submittedNote}</p> : null}
        </div>
        <div className="row-actions flex flex-wrap items-center gap-2" role="group" aria-label="Execution view">
          <button
            type="button"
            className={cn(
              "rounded-md border border-line bg-raised px-3 py-1.5 text-sm text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
              view === "spatial" && "border-accent text-ink",
            )}
            aria-pressed={view === "spatial"}
            onClick={() => onView("spatial")}
          >
            3D architecture
          </button>
          <button
            type="button"
            className={cn(
              "rounded-md border border-line bg-raised px-3 py-1.5 text-sm text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
              view === "timeline" && "border-accent text-ink",
            )}
            aria-pressed={view === "timeline"}
            onClick={() => onView("timeline")}
          >
            Request lanes
          </button>
        </div>
      </header>

      <div className="space-y-2 px-4 pt-3">
        <p className="banner m-0 border border-line bg-surface px-3 py-2 text-pretty text-xs text-muted">
          {evidence || "No recording is open. Workflow values remain unknown until observations reach the playhead."}
        </p>
        {warning ? (
          <p className="banner warn m-0 border border-amber-500 bg-surface px-3 py-2 text-pretty text-xs text-amber-300" role="status">
            {warning}
          </p>
        ) : null}
      </div>

      {view === "timeline" ? (
        <div className="timeline-view">
          <SampleLanes
            samples={samples}
            colors={colors}
            selection={selection}
            highlightedIds={highlightedIds}
            onSelect={onSelect}
          />
          <ExecutionBatchStrip
            steps={state?.steps ?? []}
            samples={samples}
            selection={selection}
            highlightedStepIds={highlightedStepIds}
            onSelect={onSelect}
          />
        </div>
      ) : (
        <div className="spatial-view relative flex min-h-[24rem] flex-1 flex-col">
          <p className="sr-only">
            The visual overview contains request intake, scheduler, KV-cache manager, model execution, and output.
            Use the component controls below the canvas for equivalent keyboard access.
          </p>

          <div className="scene-viewport relative min-h-[22rem] flex-1">
            <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
              {focus ? (
                <nav className="flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-sm" aria-label="Component breadcrumb">
                  <button
                    type="button"
                    className="border-0 bg-transparent px-1 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                    onClick={backToOverview}
                  >
                    Overview
                  </button>
                  <ChevronRight className="size-4 text-muted" aria-hidden="true" />
                  <span className="max-w-44 truncate text-ink" aria-current="page">{workflowComponent(focus).label}</span>
                </nav>
              ) : (
                <p className="m-0 rounded-md border border-line bg-surface px-3 py-1.5 text-pretty text-xs text-muted">
                  Select once to inspect · double-click to explore
                </p>
              )}
            </div>

            <div className="camera-controls absolute right-3 top-3 z-10 flex items-center gap-1 rounded-md border border-line bg-surface p-1" role="group" aria-label="3D camera controls">
              {([ ["pan-left", "Pan camera left", ArrowLeft], ["pan-right", "Pan camera right", ArrowRight], ["pan-up", "Pan camera up", ArrowUp], ["pan-down", "Pan camera down", ArrowDown] ] as const).map(([action, label, Icon]) => <button key={action} type="button" className="grid size-8 place-items-center border-0 bg-transparent p-0 text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent" aria-label={label} onClick={() => moveCamera(action)}><Icon className="size-4" aria-hidden="true" /></button>)}
              <button
                type="button"
                className="grid size-8 place-items-center border-0 bg-transparent p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                aria-label="Orbit camera left"
                onClick={() => moveCamera("orbit-left")}
              >
                <RotateCcw className="size-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                className="grid size-8 place-items-center border-0 bg-transparent p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                aria-label="Orbit camera right"
                onClick={() => moveCamera("orbit-right")}
              >
                <RotateCcw className="size-4 -scale-x-100" aria-hidden="true" />
              </button>
              <button
                type="button"
                className="grid size-8 place-items-center border-0 bg-transparent p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                aria-label="Zoom camera in"
                onClick={() => moveCamera("zoom-in")}
              >
                <ZoomIn className="size-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                className="grid size-8 place-items-center border-0 bg-transparent p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                aria-label="Zoom camera out"
                onClick={() => moveCamera("zoom-out")}
              >
                <ZoomOut className="size-4" aria-hidden="true" />
              </button>
              <button
                type="button"
                className="grid size-8 place-items-center border-0 bg-transparent p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                aria-label="Reset camera"
                onClick={() => moveCamera("reset")}
              >
                <Maximize2 className="size-4" aria-hidden="true" />
              </button>
            </div>

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
          </div>

          <div className="flex flex-wrap items-start border-t border-line bg-surface">
            <details className="max-h-52 w-[min(26rem,100%)] overflow-auto p-2">
              <summary className="cursor-pointer text-sm font-medium text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent">
                Find a request <span className="tabular-nums text-muted">({samples.length})</span>
              </summary>
              <label className="relative mt-2 block">
                <span className="sr-only">Search requests</span>
                <Search className="pointer-events-none absolute left-2 top-2.5 size-4 text-muted" aria-hidden="true" />
                <input
                  type="search"
                  className="w-full rounded-md border border-line bg-raised py-2 pl-8 pr-2 text-sm text-ink placeholder:text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  value={requestQuery}
                  placeholder="Name, id, or phase"
                  onChange={(event) => setRequestQuery(event.target.value)}
                />
              </label>
              {visibleRequests.length > 0 ? (
                <ul className="mt-2 grid max-h-28 list-none gap-1 overflow-auto p-0">
                  {visibleRequests.map((sample) => (
                    <li key={sample.sample.id}>
                      <button
                        type="button"
                        className={cn(
                          "flex w-full items-center justify-between gap-3 rounded-md border border-transparent bg-transparent px-2 py-1.5 text-left text-sm text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                          selection?.kind === "request" && selection.id === sample.sample.id && "border-accent text-ink",
                        )}
                        onClick={() => onSelect({ kind: "request", id: sample.sample.id })}
                      >
                        <span className="truncate text-ink">{sample.sample.label}</span>
                        <span className="truncate text-xs">{sample.phaseLabel}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="m-0 mt-2 text-pretty text-xs text-muted">No request matches. Clear the search to see every request.</p>
              )}
            </details>

          {focus ? (
            <details className="min-w-64 flex-1 px-3 py-2">
              <summary className="cursor-pointer text-sm font-medium text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{focus === "model-execution" ? "Model structure controls" : focus === "kv-cache" ? "Find a physical KV block" : "Recorded objects"}</summary>
              <div className="pt-3">
              {focus === "model-execution" ? <ArchitectureNavigation architecture={architecture} focus={architectureFocus} onExplore={exploreArchitecture} onSelect={onSelect} progress={teachingProgress} onProgress={onTeachingProgress} /> : focus === "kv-cache" ? <PoolNavigation cache={cache} onSelect={onSelect} /> : <FocusedObjectBar component={focus} state={state} samples={samples} selection={selection} onSelect={onSelect} />}
              </div>
            </details>
          ) : (
            <ol className="grid w-full list-none grid-cols-2 gap-2 border-t border-line bg-surface p-3 md:grid-cols-5">
              {WORKFLOW_COMPONENTS.map((component) => {
                const snapshot = componentSnapshot(state, component.id)
                const selected = selectedComponent === component.id
                return (
                  <li key={component.id} className="min-w-0">
                    <div className={cn("flex min-w-0 items-stretch rounded-md border border-line bg-raised", selected && "border-accent")}>
                      <button
                        type="button"
                        className="min-w-0 flex-1 border-0 bg-transparent px-2 py-2 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        aria-pressed={selected}
                        onClick={() => selectComponent(component.id)}
                        onDoubleClick={() => exploreComponent(component.id)}
                      >
                        <span className="block truncate text-sm font-medium text-ink">{component.shortLabel}</span>
                        <span className="block truncate text-xs tabular-nums text-muted">{snapshot.metric} {snapshot.metricLabel}</span>
                      </button>
                      <button
                        type="button"
                        className="grid size-10 place-items-center self-center border-0 border-l border-line bg-transparent p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        aria-label={`Explore ${component.label}`}
                        onClick={() => exploreComponent(component.id)}
                      >
                        <Maximize2 className="size-4" aria-hidden="true" />
                      </button>
                    </div>
                  </li>
                )
              })}
            </ol>
          )}
          </div>
        </div>
      )}
    </section>
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
                  "min-w-40 rounded-md border border-line bg-raised px-2 py-1.5 text-left text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                  sameSelection(selection, object.selection) && "border-accent",
                )}
                onClick={() => onSelect(object.selection)}
              >
                <span className="block truncate text-ink">{object.label}</span>
                <span className="block truncate text-xs tabular-nums text-muted">{object.detail}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="m-0 mt-2 flex items-center gap-2 text-pretty text-xs text-muted">
          <Box className="size-4" aria-hidden="true" /> No recorded object has reached this playhead.
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
