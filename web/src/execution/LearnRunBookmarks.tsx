import { useMemo } from "react"
import type { TraceEvent } from "../../../replay/src/reduce.ts"
import { learningBookmarks } from "./learningBookmarks.ts"
import type { Selection } from "./types.ts"

export function LearnRunBookmarks({ events, onSeek }: { events: TraceEvent[]; onSeek: (sequence: number, selection?: Selection) => void }) {
  const bookmarks = useMemo(() => learningBookmarks(events), [events])
  if (!bookmarks.length) return null
  return <details className="border-t border-line pt-2">
    <summary className="cursor-pointer text-sm text-ink">Learn this run</summary>
    <div className="mt-3 grid gap-2">
      <p className="m-0 text-pretty text-xs text-muted">Optional bookmarks jump to observations in this recording. Explore freely or open a moment below.</p>
      {bookmarks.map((bookmark) => <button key={bookmark.id} type="button" className="grid w-full gap-1 text-left" onClick={() => onSeek(bookmark.sequence, bookmark.selection)}>
        <span className="text-sm">{bookmark.title}</span>
        <span className="text-pretty text-xs text-muted">{bookmark.detail}</span>
        <span className="text-[10px] tabular-nums text-faint">Event {bookmark.sequence}</span>
      </button>)}
    </div>
  </details>
}
