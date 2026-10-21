import { useEffect, useRef, useState } from "react"

import type { ContentRecord, RecordingManifest, TraceEvent } from "../../../replay/src/reduce.ts"
import {
  fetchTrace,
  loadRunnerRecording,
  runIsActive,
  type LoadedRecording,
  type RunSnapshot,
} from "../api.ts"

export interface LiveTrace {
  events: TraceEvent[]
  content: ContentRecord[]
  manifest: RecordingManifest | null
  arrivals: Array<Record<string, unknown>>
  plan: unknown
  origin: unknown
  connected: boolean
  finalized: boolean
}

const EMPTY: LiveTrace = {
  events: [],
  content: [],
  manifest: null,
  arrivals: [],
  plan: null,
  origin: null,
  connected: true,
  finalized: false,
}

export function useLiveTrace(
  runnerUrl: string,
  live: RunSnapshot | null,
  onFinal: (recording: LoadedRecording) => void,
) {
  const [trace, setTrace] = useState<LiveTrace>(EMPTY)
  const cursors = useRef({ event: 0, content: 0 })
  const handed = useRef("")
  const onFinalRef = useRef(onFinal)
  onFinalRef.current = onFinal
  const runId = live?.run_id ?? ""
  const status = live?.status ?? "idle"

  useEffect(() => {
    cursors.current = { event: 0, content: 0 }
    handed.current = ""
    setTrace(EMPTY)
  }, [runId, runnerUrl])

  useEffect(() => {
    if (!runId) return
    let stopped = false
    let timer = 0
    let controller: AbortController | null = null
    let delay = 350
    const finalize = () => {
      controller = new AbortController()
      return loadRunnerRecording(runnerUrl, runId, controller.signal).then(loaded => {
        if (stopped) return
        handed.current = runId
        onFinalRef.current(loaded)
      }).catch(() => {
        if (!stopped) timer = window.setTimeout(poll, delay)
      })
    }
    const poll = () => {
      if (stopped || handed.current === runId) return
      controller = new AbortController()
      const requested = runId
      void fetchTrace(runnerUrl, requested, cursors.current, controller.signal)
        .then((page) => {
          if (stopped || page.run_id !== requested) return
          delay = 400
          setTrace((current) => ({
            events: mergeEvents(current.events, page.events),
            content: mergeContent(current.content, page.content),
            manifest: page.manifest ?? current.manifest,
            arrivals: page.arrivals ?? current.arrivals,
            plan: page.workload_plan ?? current.plan,
            origin: page.controller_origin ?? current.origin,
            connected: true,
            finalized: page.finalized,
          }))
          cursors.current = {
            event: page.next_event_sequence,
            content: page.next_content_sequence,
          }
          const drained = page.finalized && !page.events_truncated && !page.content_truncated
          if (drained && handed.current !== requested) {
            return finalize()
          }
          const wait = page.events_truncated || page.content_truncated ? 40 : delay
          timer = window.setTimeout(poll, wait)
          return undefined
        })
        .catch((error: unknown) => {
          if (stopped || (error instanceof DOMException && error.name === "AbortError")) return
          // The same hook owns finalized-file recovery when trace pagination is
          // unavailable. An independent App loader must not parse it concurrently.
          if (status === "completed" && live?.finalized === true) return finalize()
          delay = Math.min(delay * 2, 4000)
          setTrace((current) => ({ ...current, connected: false }))
          timer = window.setTimeout(poll, delay)
        })
    }
    const active = runIsActive(status) || handed.current !== runId
    if (active) timer = window.setTimeout(poll, 200)
    return () => {
      stopped = true
      controller?.abort()
      window.clearTimeout(timer)
    }
  }, [runId, runnerUrl, status, live?.finalized])

  return trace
}

function mergeEvents(current: TraceEvent[], incoming: TraceEvent[]): TraceEvent[] {
  if (incoming.length === 0) return current
  const seen = new Set(current.map((event) => event.sequence))
  const next = current.slice()
  for (const event of incoming) {
    if (seen.has(event.sequence)) continue
    seen.add(event.sequence)
    next.push(event)
  }
  next.sort((left, right) => left.sequence - right.sequence)
  return next
}

function mergeContent(current: ContentRecord[], incoming: ContentRecord[]): ContentRecord[] {
  if (incoming.length === 0) return current
  const seen = new Set(current.map((row) => Number(row.content_sequence ?? -1)))
  const next = current.slice()
  for (const row of incoming) {
    const sequence = Number(row.content_sequence ?? -1)
    if (seen.has(sequence)) continue
    seen.add(sequence)
    next.push(row)
  }
  return next
}
