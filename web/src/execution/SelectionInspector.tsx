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
  showClose: boolean
}) {
  const sample = selection?.kind === "request" ? samples.find((item) => item.sample.id === selection.id) ?? null : null
  const missingSample = selection?.kind === "request" && !sample
  const step = selection?.kind === "step" && state ? state.steps.find((item) => item.id === selection.id) ?? null : null
  const blocks = state ? describeBlocks(state) : []
  const block = selection?.kind === "block" ? blocks.find((item) => item.key === selection.key || `${item.pool}:${item.blockId}` === selection.key) ?? null : null
  const component = selection?.kind === "component" ? selection.id : null
  return (
    <aside className="inspector min-h-0 overflow-auto border-l border-line bg-surface p-4" aria-label="Inspector">
      <header className="inspector-head flex items-center justify-between gap-2">
        <h2 className="text-balance text-base font-semibold text-ink">
          {sample
            ? sample.sample.label
            : step
              ? step.id
              : block
                ? `Block ${block.blockId}`
                : component
                  ? workflowComponent(component).label
                  : "Inspector"}
        </h2>
        {showClose ? <button type="button" onClick={onClose}>Close inspector</button> : null}
      </header>
      {!selection || !state ? (
        <p className="text-pretty text-sm text-muted">Select a workflow component, request, execution step, or KV block to inspect recorded evidence.</p>
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
        <section className="mt-4 grid gap-3">
          <h3 className="text-balance text-sm font-semibold text-ink">Execution step</h3>
          <EvidenceSection kind="observed" lines={explainStep(state, step.id).observed} />
          <div className="row-actions">
            {step.requests.map((request) => {
              const label = samples.find((item) => item.sample.id === request.requestId)?.sample.label ?? request.requestId
              return (
                <button key={request.requestId} type="button" onClick={() => onSelect({ kind: "request", id: request.requestId })}>
                  {label}
                </button>
              )
            })}
          </div>
          <button type="button" onClick={() => onSeek(step.sequence, { kind: "step", id: step.id })}>
            Show this step
          </button>
        </section>
      ) : null}
      {block && state ? (
        <section className="mt-4 grid gap-3">
          <h3 className="text-balance text-sm font-semibold text-ink">KV block</h3>
          <EvidenceSection kind="observed" lines={explainBlock(state, `${block.pool}:${block.blockId}`).observed} />
          <EvidenceSection kind="explanatory" lines={[block.note]} />
        </section>
      ) : null}
      {manifest?.lesson?.summary ? <p className="meta">{manifest.lesson.summary}</p> : null}
    </aside>
  )
}

function ComponentDetails({ component, state }: { component: WorkflowComponentId; state: ReplayState | null }) {
  const definition = workflowComponent(component)
  const snapshot = componentSnapshot(state, component)
  return (
    <div className="inspector-body mt-4 grid gap-4">
      <p className="m-0 text-pretty text-sm text-muted">{definition.description}</p>
      <dl className="facts m-0 grid gap-2 rounded-lg border border-line bg-raised p-3">
        <div>
          <dt>Current summary</dt>
          <dd className="mono tabular-nums">{snapshot.metric} {snapshot.metricLabel}</dd>
        </div>
        <div>
          <dt>Playhead</dt>
          <dd className="mono tabular-nums">event {state?.lastSequence ?? 0}</dd>
        </div>
      </dl>
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
    derived: "Derived from observations",
    explanatory: "How to read this",
    unknown: "Unknown",
  }
  return (
    <section className="grid gap-2" data-evidence={kind}>
      <h3 className="text-balance text-xs font-semibold text-muted">{labels[kind]}</h3>
      <ul className="fact-list grid list-none gap-2 p-0 text-sm text-ink">
        {lines.map((line) => (
          <li key={line} className="rounded-md border border-line bg-raised px-3 py-2 text-pretty">{line}</li>
        ))}
      </ul>
    </section>
  )
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
