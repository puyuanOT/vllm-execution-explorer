import type { ReplayState } from "./reduce.ts"

export interface Explanation {
  observed: string[]
  explanatory: string[]
}

export function explainStep(state: ReplayState, stepId: string): Explanation {
  const step = state.steps.find((item) => item.id === stepId)
  if (!step) {
    return { observed: ["This step is outside the applied prefix of the recording."], explanatory: [] }
  }
  const observed = [
    `${step.id} is ${labelStatus(step.status)}. Token budget ${step.tokenBudget}. Scheduled tokens ${step.totalScheduledTokens}.`,
  ]
  if (step.unscheduled) {
    observed.push("This step id appears without a step_scheduled event.")
  }
  for (const request of step.requests) {
    const end = request.rangeStart + Math.max(request.rangeLength - 1, 0)
    observed.push(
      `${request.requestId} reserves tokens ${request.rangeStart}–${end}: ${request.promptTokens} prompt, ${request.generationTokens} generation.`,
    )
  }
  if (step.preemptedRequestIds.length > 0) {
    observed.push(`This step lists preempted request ids: ${step.preemptedRequestIds.join(", ")}.`)
  }
  if (step.resultObservation) {
    observed.push(`Result observation: ${step.resultObservation}.`)
  }
  return { observed, explanatory: [] }
}

export function explainRequest(state: ReplayState, requestId: string): Explanation {
  const request = state.requests[requestId]
  if (!request) {
    return { observed: ["This request is outside the applied prefix of the recording."], explanatory: [] }
  }
  const observed = [`${request.id} is ${request.status}.`, `Accepted output tokens: ${request.acceptedTokenCount}.`]
  if (request.reservedCursor != null) observed.push(`Reserved cursor: ${request.reservedCursor}.`)
  if (request.promptTokenCount != null) observed.push(`Prompt tokens: ${request.promptTokenCount}.`)
  if (request.matchedPrefixTokens != null) {
    observed.push(`Prefix lookup matched ${request.matchedPrefixTokens} tokens.`)
  }
  if (request.preemptionCount > 0) {
    const reason = request.preemptionReason
      ? `Recorded reason: ${request.preemptionReason}.`
      : "No decision reason was recorded."
    observed.push(
      `Preemption count ${request.preemptionCount}. Computed tokens ${request.computedTokensBefore} → ${request.computedTokensAfter}. ${reason}`,
    )
  }
  if (request.recomputing) observed.push("The recording has not reserved this request again after preemption.")
  if (request.finishReason) observed.push(`Terminal status: ${request.finishReason}.`)
  const explanatory: string[] = []
  if ((request.matchedPrefixTokens ?? 0) > 0) {
    explanatory.push("Prefix reuse skips the matched prompt work. It does not skip ordinary decoding.")
  }
  if (request.preemptionCount > 0 && request.preemptionReason == null) {
    explanatory.push(
      "vLLM can preempt a running request when it cannot keep its KV blocks, then recompute that request. This recording did not include a decision reason.",
    )
  }
  if (request.acceptedTokenCount > 0) {
    explanatory.push(
      "Candidate probabilities were not captured. Sampling chooses one next token from the model distribution. This view shows the accepted token only.",
    )
  }
  return { observed, explanatory }
}

export function explainBlock(state: ReplayState, key: string): Explanation {
  const block = state.blocks[key]
  if (!block) {
    return { observed: ["This block is outside the applied prefix of the recording."], explanatory: [] }
  }
  const hash = block.cachedAtZeroRefs
    ? "Reference count is 0 and the block still has a hash, so it is cached."
    : block.hasHash
      ? "The block has a hash."
      : "The block has no hash."
  const observed = [
    `${block.pool} block ${block.blockId}, allocation generation ${block.generation}.`,
    `Last operation ${block.lastOperation}. Reference count ${block.refCount}.`,
    hash,
  ]
  const explanatory =
    block.cachedAtZeroRefs || block.lastOperation === "release" || block.lastOperation === "evict"
      ? [
          "A zero-reference hashed block stays cached until it is evicted or assigned to a new occupant. A new occupant gets a new allocation generation.",
        ]
      : []
  return { observed, explanatory }
}

function labelStatus(status: string): string {
  if (status === "result_received") return "result received"
  return status
}
