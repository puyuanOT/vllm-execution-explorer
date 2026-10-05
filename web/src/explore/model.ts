import type { ReplayState } from "../../../replay/src/reduce.ts"

export type WorkflowComponentId =
  | "request-intake"
  | "scheduler"
  | "kv-cache"
  | "model-execution"
  | "output"

export type EvidenceKind = "observed" | "derived" | "explanatory" | "unknown"

export interface WorkflowComponentDefinition {
  id: WorkflowComponentId
  shortLabel: string
  label: string
  description: string
  explanation: string
}

export interface WorkflowComponentSnapshot {
  metric: string
  metricLabel: string
  evidence: Record<EvidenceKind, string[]>
  unavailable: boolean
}

export const WORKFLOW_COMPONENTS: readonly WorkflowComponentDefinition[] = [
  {
    id: "request-intake",
    shortLabel: "Intake",
    label: "Request intake",
    description: "HTTP receipt, prompt preparation, and admission observations.",
    explanation: "Requests enter independently even when they share a planned arrival batch.",
  },
  {
    id: "scheduler",
    shortLabel: "Scheduler",
    label: "Scheduler",
    description: "Recorded queues, execution steps, token budgets, and preemptions.",
    explanation: "vLLM chooses the participants and token ranges for each engine step.",
  },
  {
    id: "kv-cache",
    shortLabel: "KV cache",
    label: "KV-cache manager",
    description: "Recorded block references, reuse, release, and eviction state.",
    explanation: "KV blocks hold attention state and can be shared when a recorded prefix match permits it.",
  },
  {
    id: "model-execution",
    shortLabel: "Model",
    label: "Model execution",
    description: "Submitted steps, participants, and recorded executor results.",
    explanation: "This surface shows host-observed submissions and results, not GPU kernel timing.",
  },
  {
    id: "output",
    shortLabel: "Output",
    label: "Output",
    description: "Accepted tokens, emitted text, and terminal request state.",
    explanation: "Only output reached by the current playhead is shown.",
  },
] as const

export function workflowComponent(id: WorkflowComponentId): WorkflowComponentDefinition {
  return WORKFLOW_COMPONENTS.find((item) => item.id === id) ?? WORKFLOW_COMPONENTS[0]
}

/** Requests are named by their workload label; the request id is the fallback and the secondary detail. */
export function requestLabel(labels: ReadonlyMap<string, string> | undefined, requestId: string): string {
  return labels?.get(requestId)?.trim() || requestId
}

const UNGROUPED_PREFIX = "ungrouped:"

/** Recordings without a workload plan give every request its own synthetic batch; those share one "not recorded" group. */
export function arrivalGroup(sample: { batchId: string; batchLabel: string }): { key: string; label: string; recorded: boolean } {
  return sample.batchId.startsWith(UNGROUPED_PREFIX)
    ? { key: UNGROUPED_PREFIX, label: "Arrival group not recorded", recorded: false }
    : { key: sample.batchId, label: `Batch ${sample.batchLabel}`, recorded: true }
}

export function componentSnapshot(
  state: ReplayState | null,
  id: WorkflowComponentId,
): WorkflowComponentSnapshot {
  if (!state) {
    return emptySnapshot("No recorded execution state is available at this playhead.")
  }

  const evidence = evidenceRecord()
  const requests = Object.values(state.requests)
  const steps = state.steps
  const latestStep = steps[steps.length - 1] ?? null
  const blocks = Object.values(state.blocks)
  const prefixCaveat = state.trust === "unknown"
    ? "A trace gap makes state after the last trusted event unknown. Counts describe only the applied prefix."
    : ""
  if (prefixCaveat) evidence.unknown.push(prefixCaveat)

  if (id === "request-intake") {
    const queued = requests.filter((request) => request.status === "queued").length
    evidence.derived.push(`${requests.length} request${requests.length === 1 ? "" : "s"} appear in the applied recording prefix.`)
    evidence.derived.push(`${queued} currently have the recorded status queued.`)
    if (requests.length === 0) evidence.unknown.push("No request receipt has reached this playhead.")
    return snapshot(String(requests.length), "requests observed", evidence, requests.length === 0)
  }

  if (id === "scheduler") {
    evidence.derived.push(`${steps.length} scheduler step${steps.length === 1 ? "" : "s"} appear in the applied prefix.`)
    if (latestStep) {
      evidence.observed.push(
        `${latestStep.id} is ${latestStep.status.replaceAll("_", " ")} with ${latestStep.totalScheduledTokens} scheduled tokens against a recorded budget of ${latestStep.tokenBudget}.`,
      )
      if (latestStep.preemptedRequestIds.length > 0) {
        evidence.observed.push(`Recorded preemptions: ${latestStep.preemptedRequestIds.join(", ")}.`)
      }
    } else {
      evidence.unknown.push("No scheduler step has reached this playhead.")
    }
    evidence.explanatory.push("A planned arrival batch does not force one atomic scheduler step.")
    return snapshot(String(steps.length), "steps observed", evidence, steps.length === 0)
  }

  if (id === "kv-cache") {
    const referenced = blocks.filter((block) => block.refCount > 0).length
    const cached = blocks.filter((block) => block.cachedAtZeroRefs).length
    evidence.derived.push(`${blocks.length} allocation generation${blocks.length === 1 ? "" : "s"} ${blocks.length === 1 ? "is" : "are"} represented in the current block state.`)
    evidence.derived.push(`${referenced} referenced and ${cached} cached-at-zero-reference block${cached === 1 ? "" : "s"} are visible.`)
    if (blocks.length === 0) evidence.unknown.push("No cache block transition has reached this playhead.")
    evidence.explanatory.push("Pool, block id, and allocation generation together identify a recorded allocation.")
    return snapshot(String(blocks.length), "block allocations", evidence, blocks.length === 0)
  }

  if (id === "model-execution") {
    const submitted = steps.filter((step) => step.submittedSequence != null).length
    const results = steps.filter((step) => step.resultSequence != null).length
    evidence.derived.push(`${submitted} steps have a recorded executor submission; ${results} have a recorded result.`)
    if (latestStep?.resultObservation) {
      evidence.observed.push(`Latest recorded result observation: ${latestStep.resultObservation}.`)
    }
    if (submitted === 0) evidence.unknown.push("No executor submission has reached this playhead.")
    evidence.explanatory.push("No GPU kernel duration is inferred from host-side observations.")
    return snapshot(`${results}/${submitted}`, "results / submissions", evidence, submitted === 0)
  }

  const acceptedTokens = requests.reduce((total, request) => total + request.acceptedTokenCount, 0)
  const completed = requests.filter((request) => request.status === "completed").length
  evidence.derived.push(`${acceptedTokens} accepted output token${acceptedTokens === 1 ? "" : "s"} are visible through this playhead.`)
  evidence.derived.push(`${state.emissions.length} text emission${state.emissions.length === 1 ? "" : "s"} and ${completed} completed request${completed === 1 ? "" : "s"} are recorded.`)
  if (acceptedTokens === 0 && state.emissions.length === 0) {
    evidence.unknown.push("No accepted token or text emission has reached this playhead.")
  }
  evidence.explanatory.push("Future tokens and completion reasons remain hidden until their events reach the playhead.")
  return snapshot(String(acceptedTokens), "accepted tokens", evidence, acceptedTokens === 0 && state.emissions.length === 0)
}

function evidenceRecord(): Record<EvidenceKind, string[]> {
  return { observed: [], derived: [], explanatory: [], unknown: [] }
}

function emptySnapshot(message: string): WorkflowComponentSnapshot {
  const evidence = evidenceRecord()
  evidence.unknown.push(message)
  return snapshot("—", "not observed", evidence, true)
}

function snapshot(
  metric: string,
  metricLabel: string,
  evidence: Record<EvidenceKind, string[]>,
  unavailable: boolean,
): WorkflowComponentSnapshot {
  return { metric, metricLabel, evidence, unavailable }
}
