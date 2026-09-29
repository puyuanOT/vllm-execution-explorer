import type { SampleProjection } from "../../../replay/src/timeline.ts"
import { cn } from "../lib/cn.ts"
import type { Selection } from "./types.ts"

export function SampleLanes({
  samples,
  colors,
  selection,
  highlightedIds,
  onSelect,
}: {
  samples: SampleProjection[]
  colors: Record<string, string>
  selection: Selection | null
  highlightedIds: Set<string>
  onSelect: (selection: Selection) => void
}) {
  const groups = groupByBatch(samples)
  return (
    <div className="lanes" aria-label="Sample execution">
      {samples.length === 0 ? <p className="empty text-pretty">Add samples to see a lane for each one.</p> : null}
      {groups.map((group) => (
        <section key={group.batchId} className="lane-group" aria-label={`Arrival batch ${group.batchLabel}`}>
          <h2 className="text-balance">
            {group.batchLabel}
            <span className="mono">{group.arrivalOffsetMs == null ? "Event-triggered" : `${group.arrivalOffsetMs} ms`}</span>
          </h2>
          <ol>
            {group.samples.map((item) => {
              const selected = selection?.kind === "request" && selection.id === item.sample.id
              const highlighted = highlightedIds.has(item.sample.id)
              return (
                <li key={item.sample.id} className={cn("lane", highlighted && "highlighted")}>
                  <button
                    type="button"
                    className="lane-button"
                    aria-pressed={selected}
                    data-sample-id={item.sample.id}
                    onClick={() => onSelect({ kind: "request", id: item.sample.id })}
                  >
                    <span className="swatch" style={{ background: colors[item.sample.id] ?? "#8b97a8" }} aria-hidden="true" />
                    <span className="lane-name">{item.sample.label}</span>
                    <span className="phase">{item.phaseLabel}</span>
                  </button>
                  <p className="lane-facts">
                    <span>Planned {item.sample.arrivalOffsetMs == null ? "unavailable" : `${item.sample.arrivalOffsetMs} ms`}</span>
                    <span className="mono">Accepted {item.acceptedTokenCount == null ? "—" : item.acceptedTokenCount}</span>
                    {item.sample.maxOutputTokens != null ? <span className="mono">Limit {item.sample.maxOutputTokens}</span> : null}
                  </p>
                  <div className="chips">
                    {item.scheduled.map((segment) => (
                      <span key={`${segment.stepId}-${segment.rangeStart}`} className={cn("chip", segment.kind)}>
                        {chipLabel(segment.kind)} {segment.rangeStart}–{segment.rangeEnd}
                      </span>
                    ))}
                    {item.scheduled.length === 0 ? <span className="chip quiet">No scheduled range yet</span> : null}
                  </div>
                </li>
              )
            })}
          </ol>
        </section>
      ))}
    </div>
  )
}

function chipLabel(kind: "prefill" | "decode" | "mixed"): string {
  if (kind === "prefill") return "Prefill"
  if (kind === "decode") return "Decode"
  return "Mixed"
}

function groupByBatch(samples: SampleProjection[]): Array<{
  batchId: string
  batchLabel: string
  arrivalOffsetMs: number | null
  samples: SampleProjection[]
}> {
  const groups: Array<{ batchId: string; batchLabel: string; arrivalOffsetMs: number | null; samples: SampleProjection[] }> = []
  for (const item of samples) {
    const previous = groups[groups.length - 1]
    if (previous && previous.batchId === item.sample.batchId) {
      previous.samples.push(item)
      continue
    }
    groups.push({
      batchId: item.sample.batchId,
      batchLabel: item.sample.batchLabel,
      arrivalOffsetMs: item.sample.arrivalOffsetMs,
      samples: [item],
    })
  }
  return groups
}
