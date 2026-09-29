import { Box, ChevronRight, Maximize2, RotateCcw, Search, ZoomIn, ZoomOut } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { initialState, type ReplayState } from "../../../replay/src/reduce.ts"
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
  const [cameraCommand, setCameraCommand] = useState<CameraCommand | null>(null)
  const [requestQuery, setRequestQuery] = useState("")
  const selectedComponent = selection?.kind === "component" ? selection.id : null
  const replay = state ?? EMPTY_REPLAY
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
    setRequestQuery("")
  }, [requestOrder])

  function selectComponent(component: WorkflowComponentId) {
    onSelect({ kind: "component", id: component })
  }

  function exploreComponent(component: WorkflowComponentId) {
    selectComponent(component)
    setFocus(component)
  }

  function moveCamera(action: CameraAction) {
    setCameraCommand((current) => ({ id: (current?.id ?? 0) + 1, action }))
  }

  return (
    <section className="workspace-main min-h-0 min-w-0 overflow-hidden bg-canvas" aria-label="Execution explorer">
      <header className="workspace-head border-b border-line bg-surface px-4 py-3">
        <div className="min-w-0">
          <h1 className="text-balance text-base font-semibold text-ink">
            {view === "spatial" ? (focus ? workflowComponent(focus).label : "Execution workflow") : "Request lanes"}
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
            Workflow
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
        <div className="spatial-view relative flex min-h-[28rem] flex-1 flex-col overflow-hidden">
          <p className="sr-only">
            The visual overview contains request intake, scheduler, KV-cache manager, model execution, and output.
            Use the component controls below the canvas for equivalent keyboard access.
          </p>

          <div className="relative min-h-0 flex-1">
            <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
              {focus ? (
                <nav className="flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-sm" aria-label="Component breadcrumb">
                  <button
                    type="button"
                    className="border-0 bg-transparent px-1 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                    onClick={() => setFocus(null)}
                  >
                    Workflow
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

            <div className="absolute right-3 top-3 z-10 flex items-center gap-1 rounded-md border border-line bg-surface p-1" role="group" aria-label="3D camera controls">
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

            <ExecutionScene
              state={replay}
              requestOrder={requestOrder}
              selection={selection}
              animate={animate}
              focus={focus}
              cameraCommand={cameraCommand}
              onSelect={onSelect}
              onExplore={exploreComponent}
            />

            <details className="absolute bottom-3 left-3 z-10 max-h-52 w-[min(26rem,calc(100%-1.5rem))] overflow-auto rounded-md border border-line bg-surface p-2">
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
          </div>

          {focus ? (
            <FocusedObjectBar component={focus} state={state} samples={samples} selection={selection} onSelect={onSelect} onBack={() => setFocus(null)} />
          ) : (
            <ol className="grid list-none grid-cols-2 gap-2 border-t border-line bg-surface p-3 md:grid-cols-5">
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
  onBack,
}: {
  component: WorkflowComponentId
  state: ReplayState | null
  samples: SampleProjection[]
  selection: Selection | null
  onSelect: (selection: Selection) => void
  onBack: () => void
}) {
  const definition = workflowComponent(component)
  const objects = focusedObjects(component, state, samples)
  return (
    <div className="border-t border-line bg-surface p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-balance text-sm font-semibold text-ink">{definition.label}</h2>
          <p className="m-0 truncate text-pretty text-xs text-muted">{definition.description}</p>
        </div>
        <button
          type="button"
          className="shrink-0 rounded-md border border-line bg-raised px-3 py-1.5 text-sm text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          onClick={onBack}
        >
          Back to overview
        </button>
      </div>
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
