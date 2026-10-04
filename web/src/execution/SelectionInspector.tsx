import type { MeasuredProfileProjection, ProfileActivity } from "../../../replay/src/profiling.ts"
import { MeasuredProfileDetails } from "./MeasuredProfileDetails.tsx"
import { explainBlock, explainRequest, explainStep } from "../../../replay/src/explain.ts"
import type { RecordingManifest, ReplayState } from "../../../replay/src/reduce.ts"
import {
  describeBlocks,
  nextStepForSample,
  type SampleProjection,
  type TimelineIndex,
} from "../../../replay/src/timeline.ts"
import { componentSnapshot, workflowComponent } from "../explore/model.ts"
import type { EvidenceKind, WorkflowComponentId } from "../explore/model.ts"
import type { Selection } from "./types.ts"
import { LearningDetails } from "./LearningDetails.tsx"
import { EvidenceTag, Readouts } from "../ui/controls.tsx"

export function SelectionInspector({
  selection,
  samples,
  state,
  index,
  manifest,
  sequence,
  onSeek,
  onSelect,
  onClose,
  showClose,
  measuredProfile,
  activeProfileActivities = [],
  profileError = "",
  measuredMode = false,
  onProfileTime,
}: {
  selection: Selection | null
  samples: SampleProjection[]
  state: ReplayState | null
  index: TimelineIndex | null
  manifest: RecordingManifest | null
  sequence: number
  onSeek: (sequence: number, selection?: Selection) => void
  onSelect: (selection: Selection) => void
  onClose: () => void
  measuredProfile?: MeasuredProfileProjection
  activeProfileActivities?: ProfileActivity[]
  profileError?: string
  measuredMode?: boolean
  onProfileTime?: (value: number) => void
  showClose: boolean
}) {
  const sample = selection?.kind === "request" ? samples.find((item) => item.sample.id === selection.id) ?? null : null
  const missingSample = selection?.kind === "request" && !sample
  const step = selection?.kind === "step" && state ? state.steps.find((item) => item.id === selection.id) ?? null : null
  const blocks = state ? describeBlocks(state) : []
  const block = selection?.kind === "block" ? blocks.find((item) => item.key === selection.key || `${item.pool}:${item.blockId}` === selection.key) ?? null : null
  const component = selection?.kind === "component" ? selection.id : null
  return (
    <aside className="inspector grid min-h-0 content-start gap-3 overflow-auto border-l border-line bg-[#060a11] px-[18px] py-4" aria-label="Inspector">
      <header className="grid gap-1 border-b border-line pb-3">
        <p className="cp-label m-0">{inspectorKind(selection, Boolean(sample), Boolean(step), Boolean(block), component)}</p>
        <div className="flex items-start justify-between gap-2">
        <h2 className="text-balance text-[19px] font-semibold tracking-[-0.01em] text-ink">
          {sample
            ? sample.sample.label
            : step
              ? step.id
              : block
                ? `Block ${block.blockId}`
                : component
                  ? workflowComponent(component).label
                  : selection?.kind === "token" ? "Request token"
                  : selection?.kind === "layer" ? `Layer ${selection.layerIndex + 1}`
                  : selection?.kind === "module" ? (selection.module === "attention" ? "Attention" : "Feed-forward")
                  : selection?.kind === "head" ? `${selection.headKind === "query" ? "Query" : "KV"} head ${selection.headIndex + 1}`
                  : selection?.kind === "block" ? "KV block"
                  : "Inspector"}
        </h2>
        {showClose ? <button type="button" onClick={onClose}>Close inspector</button> : null}
        </div>
      </header>
      {measuredProfile && onProfileTime ? <MeasuredProfileDetails profile={measuredProfile} activities={activeProfileActivities} error={profileError} active={measuredMode} onTime={onProfileTime} onSelect={onSelect} /> : null}
      {!selection || !state ? (
        <p className="m-0 text-pretty text-sm text-muted">Select a workflow component, request, execution step, or KV block to inspect recorded evidence. Click an object in the scene or a lane, or use Find request.</p>
      ) : null}
      {missingSample ? <p>This sample is not in the open view.</p> : null}
      {component ? <ComponentDetails component={component} state={state} /> : null}
      {sample ? (
        <SampleDetails
          sample={sample}
          state={state}
          index={index}
          sequence={sequence}
          fixture={manifest?.execution_kind === "fixture" || manifest?.demo_fixture === true}
          blocks={blocks.filter((item) => item.referencedBy.includes(sample.sample.id))}
          onSeek={onSeek}
          onSelect={onSelect}
        />
      ) : null}
      {step && state ? (
        <section className="grid gap-3">
          <h3>Execution step</h3>
          <EvidenceSection kind="observed" lines={explainStep(state, step.id).observed} />
          <div className="row-actions">
            {step.requests.map((request) => {
              const label = samples.find((item) => item.sample.id === request.requestId)?.sample.label ?? request.requestId
              return (
                <button key={request.requestId} type="button" className="font-mono text-xs" onClick={() => onSelect({ kind: "request", id: request.requestId })}>
                  {label}
                </button>
              )
            })}
          </div>
          <button type="button" className="justify-self-start font-mono text-[11px] uppercase tracking-[0.06em]" onClick={() => onSeek(step.sequence, { kind: "step", id: step.id })}>
            Show this step
          </button>
        </section>
      ) : null}
      <LearningDetails selection={selection} state={state} manifest={manifest} onSelect={onSelect} />
      {block && state ? (
        <details open={!state.cacheInventory || undefined}>
          <summary className="cursor-pointer font-mono text-[10.5px] uppercase tracking-[0.1em] text-muted">Block lifecycle evidence</summary>
          <div className="mt-2 grid gap-3">
          <EvidenceSection kind="observed" lines={explainBlock(state, `${block.pool}:${block.blockId}`).observed} />
          <EvidenceSection kind="explanatory" lines={[block.note]} />
          </div>
        </details>
      ) : null}
      {manifest?.lesson?.summary ? (
        <div className="grid gap-1.5 border border-dashed border-synthetic/35 bg-synthetic/[0.03] px-3 py-2.5">
          <span className="flex items-center gap-2"><EvidenceTag kind={manifest.lesson.evidence === "fixture" || manifest.demo_fixture === true ? "synthetic" : "neutral"}>{manifest.lesson.evidence === "fixture" || manifest.demo_fixture === true ? "Synthetic" : "Lesson"}</EvidenceTag><span className="cp-label truncate">{manifest.lesson.title ?? "Lesson"}</span></span>
          <p className="meta m-0">{manifest.lesson.summary}</p>
        </div>
      ) : null}
    </aside>
  )
}

function ComponentDetails({ component, state }: { component: WorkflowComponentId; state: ReplayState | null }) {
  const definition = workflowComponent(component)
  const snapshot = componentSnapshot(state, component)
  return (
    <div className="inspector-body grid gap-4">
      <p className="m-0 text-pretty text-sm text-muted">{definition.description}</p>
      <Readouts items={[
        { label: snapshot.metricLabel, value: snapshot.metric },
        { label: "Playhead event", value: String(state?.lastSequence ?? 0) },
      ]} />
      <EvidenceSection kind="observed" lines={snapshot.evidence.observed} />
      <EvidenceSection kind="derived" lines={snapshot.evidence.derived} />
      <EvidenceSection kind="unknown" lines={snapshot.evidence.unknown} />
      <EvidenceSection kind="explanatory" lines={[...snapshot.evidence.explanatory, definition.explanation]} />
    </div>
  )
}

function EvidenceSection({ kind, lines }: { kind: EvidenceKind; lines: string[] }) {
  if (lines.length === 0) return null
  const labels: Record<EvidenceKind, string> = {
    observed: "Observed",
    derived: "Derived",
    explanatory: "Explains",
    unknown: "Unknown",
  }
  return (
    <section data-evidence={kind} aria-label={kind === "derived" ? "Derived from observations" : kind === "explanatory" ? "How to read this" : labels[kind]}>
      <ul className="cp-evlist m-0 list-none p-0">
        {lines.map((line) => (
          <li key={line}>
            <span><EvidenceTag kind={kind}>{labels[kind]}</EvidenceTag></span>
            <span className="text-pretty">{line}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function inspectorKind(selection: Selection | null, sample: boolean, step: boolean, block: boolean, component: WorkflowComponentId | null): string {
  if (sample) return "Request"
  if (step) return "Scheduler step"
  if (block) return "Physical block"
  if (component) return "Component"
  if (!selection) return "Inspector"
  if (selection.kind === "token") return "Token position"
  if (selection.kind === "layer") return "Model layer"
  if (selection.kind === "module") return "Layer module"
  if (selection.kind === "head") return "Attention head"
  if (selection.kind === "block") return "Physical block"
  return "Selection"
}

function SampleDetails({
  sample,
  state,
  index,
  sequence,
  fixture,
  blocks,
  onSeek,
  onSelect,
}: {
  sample: SampleProjection
  state: ReplayState | null
  index: TimelineIndex | null
  sequence: number
  fixture: boolean
  blocks: ReturnType<typeof describeBlocks>
  onSeek: (sequence: number, selection?: Selection) => void
  onSelect: (selection: Selection) => void
}) {
  const request = state?.requests[sample.sample.id]
  const next = index ? nextStepForSample(index, sample.sample.id, sequence) : null
  const explanation = state && request ? explainRequest(state, sample.sample.id) : null
  return (
    <div className="inspector-body">
      <p className="phase">{sample.phaseLabel}</p>
      <dl className="facts">
        <div><dt>Request id</dt><dd className="mono">{sample.sample.id}</dd></div>
        <div><dt>Arrival batch</dt><dd>{sample.sample.batchLabel}</dd></div>
        <div>
          <dt>Model route</dt>
          <dd>{modelRouteLabel(request)}</dd>
        </div>
        {request?.adapterId ? <div><dt>Adapter id</dt><dd className="mono">{request.adapterId}</dd></div> : null}
        {request?.loraIntId != null ? <div><dt>LoRA route id</dt><dd className="mono">{request.loraIntId}</dd></div> : null}
        <div>
          <dt>Requested arrival</dt>
          <dd className="mono">{sample.sample.arrivalOffsetMs == null ? "Not a controller offset" : `${sample.sample.arrivalOffsetMs} ms`}</dd>
        </div>
        <div>
          <dt>Observed request receipt</dt>
          <dd className="mono tabular-nums">{sample.observedReceiptNs == null ? (fixture ? "Not applicable to fixture transport" : "Not observed in this prefix") : formatNs(sample.observedReceiptNs)}</dd>
        </div>
        <div>
          <dt>Dispatch returned to controller</dt>
          <dd className="mono tabular-nums">{sample.controllerAddReturnedNs == null ? "Not observed in this prefix" : formatNs(sample.controllerAddReturnedNs)}</dd>
        </div>
        <div>
          <dt>Accepted output</dt>
          <dd className="mono tabular-nums">
            {sample.acceptedTokenCount == null ? "—" : sample.acceptedTokenCount}
            {sample.sample.maxOutputTokens != null ? ` / ${sample.sample.maxOutputTokens} max` : ""}
          </dd>
        </div>
      </dl>
      <h3>Prompt</h3>
      {sample.promptText == null ? (
        <p className="missing">Prompt text is unavailable.</p>
      ) : (
        <p className="text-block">{sample.promptText}</p>
      )}
      {sample.promptSource === "submitted-plan" ? <p className="meta">Shown from the submitted sequence. The recording has not attached this prompt yet.</p> : null}
      <h3>Generated output</h3>
      {sample.outputText ? <p className="text-block">{sample.outputText}</p> : <p className="missing">No emitted text in this prefix.</p>}
      <h3>Scheduled ranges</h3>
      {sample.scheduled.length === 0 ? <p className="meta">No scheduled range in this prefix. A scheduled range is not accepted output.</p> : (
        <ul className="fact-list">
          {sample.scheduled.map((segment) => (
            <li key={`${segment.stepId}-${segment.rangeStart}`}>
              <button type="button" onClick={() => onSeek(segment.sequence, { kind: "step", id: segment.stepId })}>
                {segment.stepId}: {segment.kind} tokens {segment.rangeStart}–{segment.rangeEnd} ({segment.promptTokens} prompt, {segment.generationTokens} generation)
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="row-actions">
        <button type="button" disabled={next == null} onClick={() => { if (next != null) onSeek(next) }}>
          Next step for this sample
        </button>
      </div>
      <h3>Milestones</h3>
      {sample.milestones.length === 0 ? <p className="meta">No milestone in this prefix.</p> : (
        <ul className="fact-list">
          {sample.milestones.map((milestone) => (
            <li key={`${milestone.kind}-${milestone.sequence}`}>
              <button type="button" onClick={() => onSeek(milestone.sequence)}>
                {milestone.label} · event {milestone.sequence}
              </button>
            </li>
          ))}
        </ul>
      )}
      {explanation ? (
        <details>
          <summary>How to read this sample</summary>
          <div className="mt-2 grid gap-3">
            <EvidenceSection kind="observed" lines={explanation.observed} />
            <EvidenceSection kind="explanatory" lines={explanation.explanatory} />
          </div>
        </details>
      ) : null}
      <details>
        <summary>KV blocks and technical details</summary>
        {blocks.length === 0 ? <p className="meta">No KV block for this sample is in the applied prefix.</p> : (
          <ul className="fact-list">
            {blocks.map((item) => (
              <li key={item.key}>
                <button type="button" onClick={() => onSelect({ kind: "block", key: `${item.pool}:${item.blockId}` })}>
                  {item.pool} block {item.blockId}, generation {item.generation}, refs {item.refCount}
                </button>
                <span>{item.note}</span>
              </li>
            ))}
          </ul>
        )}
        {request ? (
          <dl className="facts">
            <div><dt>Status field</dt><dd>{request.status}</dd></div>
            <div><dt>Prompt tokens</dt><dd className="mono">{request.promptTokenCount ?? "—"}</dd></div>
            <div><dt>Preemptions</dt><dd className="mono">{request.preemptionCount}</dd></div>
            <div><dt>Prefix match</dt><dd className="mono">{request.matchedPrefixTokens ?? "—"}</dd></div>
          </dl>
        ) : <p className="meta">This sample has no request event in the applied prefix.</p>}
      </details>
    </div>
  )
}

function formatNs(value: number | string): string {
  return `${value} ns`
}

function modelRouteLabel(request: ReplayState["requests"][string] | undefined): string {
  if (!request?.modelRouteKind) return "Not observed in this prefix"
  if (request.modelRouteKind === "lora_adapter") return "LoRA adapter"
  if (request.modelRouteKind === "base_model") return "Base model"
  return request.modelRouteKind.replaceAll("_", " ")
}
