import { projectRequestTokens } from "../../../replay/src/learning.ts"
import { createPlayer, type TraceEvent } from "../../../replay/src/reduce.ts"
import { buildTimeline, type IndexedStep } from "../../../replay/src/timeline.ts"
import type { Selection } from "./types.ts"

export interface LearningBookmark {
  id: string
  title: string
  detail: string
  sequence: number
  selection: Selection
}

/** Optional entry points into actual observations, never fixed demo event numbers. */
export function learningBookmarks(events: TraceEvent[]): LearningBookmark[] {
  const gap = events.findIndex((event) => event.event_type === "trace_gap")
  const trusted = gap < 0 ? events : events.slice(0, gap)
  if (!trusted.length) return []
  const timeline = buildTimeline(trusted)
  const bookmarks: LearningBookmark[] = []
  let player: ReturnType<typeof createPlayer> | null = null
  const tokenAt = (event: TraceEvent, requestId: string, position: number) => {
    player ??= createPlayer(trusted)
    return projectRequestTokens(player.at(event.sequence), requestId).find((item) => item.position === position)
  }
  const endpoint = (step: IndexedStep) => trusted.reduce((sequence, event) => event.step_id === step.id ? Math.max(sequence, event.sequence) : sequence, step.sequence)
  const prefill = timeline.steps.find((step) => step.participants.some((request) => request.promptTokens > 0))
  if (prefill) bookmarks.push({
    id: "prefill", title: "A prefill iteration",
    detail: `${prefill.participants.reduce((sum, request) => sum + request.promptTokens, 0)} prompt positions scheduled. Inspect the participating requests and their recorded KV writes.`,
    sequence: endpoint(prefill), selection: { kind: "step", id: prefill.id },
  })

  const decode = timeline.steps.find((step) => step.participants.some((request) => request.generationTokens > 0))
  if (decode) {
    const participant = decode.participants.find((request) => request.generationTokens > 0)!
    const position = participant.rangeStart + participant.promptTokens
    const binding = trusted.find((event) => event.event_type === "kv_block_binding" && event.step_id === decode.id
      && event.request_id === participant.requestId && containsPosition(event, position))
    const token = binding ? tokenAt(binding, participant.requestId, position) : null
    bookmarks.push(token?.kvState === "reserved" && binding ? {
      id: "decode-input", title: "A generated token becomes model input",
      detail: `Position ${position} is bound to a slot, but its KV computation has not yet been observed. Allocation and computation are separate boundaries.`,
      sequence: binding.sequence, selection: { kind: "token", requestId: participant.requestId, position, kvBlockKey: blockKey(binding) ?? undefined },
    } : {
      id: "decode-input", title: "A decode iteration",
      detail: `The scheduler processes generated-token inputs. Sampling a token and processing that token are separate actions.`,
      sequence: decode.sequence, selection: { kind: "step", id: decode.id },
    })
  }

  const released = trusted.find((event) => event.event_type === "block_change" && event.operation === "release"
    && event.ref_count_after === 0 && (event.has_hash_after === true || event.cached_at_zero_refs === true) && blockKey(event))
  if (released) bookmarks.push({
    id: "release", title: "Release ownership, retain a prefix",
    detail: "This block has zero request references and retains a prefix-cache hash. Releasing a holder does not erase recorded KV contents.",
    sequence: released.sequence, selection: { kind: "block", key: blockKey(released)! },
  })

  const match = trusted.find((event) => event.event_type === "prefix_lookup" && typeof event.request_id === "string"
    && typeof event.matched_token_count === "number" && event.matched_token_count > 0)
  if (match) {
    const requestId = match.request_id as string
    const binding = trusted.find((event) => event.sequence > match.sequence && event.event_type === "kv_block_binding"
      && event.request_id === requestId && typeof event.token_range_start === "number" && event.token_range_start < (match.matched_token_count as number)
      && blockKey(event) && tokenAt(event, requestId, event.token_range_start)?.locations.some((location) => location.key === blockKey(event) && location.valid))
    bookmarks.push({
      id: "prefix-reuse", title: binding ? "Reuse an existing prefix" : "A prefix match is observed",
      detail: `${match.matched_token_count} prompt positions matched cached prefix data.${binding ? " Inspect a matched position: the current holder and original writer can differ." : " A physical slot binding has not been captured at this boundary."}`,
      sequence: binding?.sequence ?? match.sequence,
      selection: binding ? { kind: "token", requestId, position: binding.token_range_start as number, kvBlockKey: blockKey(binding)! } : { kind: "request", id: requestId },
    })
  }

  const mixed = timeline.steps.find((step) => step.participants.some((request) => request.promptTokens > 0)
    && step.participants.some((request) => request.generationTokens > 0))
  if (mixed) bookmarks.push({
    id: "mixed-batch", title: "Prefill and decode share an iteration",
    detail: `${mixed.participants.reduce((sum, request) => sum + request.promptTokens, 0)} prompt and ${mixed.participants.reduce((sum, request) => sum + request.generationTokens, 0)} generated-token positions are scheduled together.`,
    sequence: endpoint(mixed), selection: { kind: "step", id: mixed.id },
  })
  return bookmarks
}

function blockKey(event: TraceEvent): string | null {
  return typeof event.pool === "string" && typeof event.block_id === "number" && Number.isSafeInteger(event.block_id) ? `${event.pool}:${event.block_id}` : null
}

function containsPosition(event: TraceEvent, position: number): boolean {
  return typeof event.token_range_start === "number" && typeof event.token_range_end === "number"
    && event.token_range_start <= position && event.token_range_end > position
}
