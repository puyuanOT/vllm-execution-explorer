import type { ReplayState } from "../../../replay/src/reduce.ts"
import type { SampleProjection } from "../../../replay/src/timeline.ts"
import { cn } from "../lib/cn.ts"
import type { Selection } from "./types.ts"

export function ExecutionBatchStrip({
  steps,
  samples,
  selection,
  highlightedStepIds,
  onSelect,
}: {
  steps: ReplayState["steps"]
  samples: SampleProjection[]
  selection: Selection | null
  highlightedStepIds: Set<string>
  onSelect: (selection: Selection) => void
}) {
  const labels = new Map(samples.map((item) => [item.sample.id, item.sample.label]))
  return (
    <section className="step-strip" aria-label="Recorded scheduler steps">
      <header>
        <h2 className="text-balance">Recorded scheduler steps</h2>
        <p className="text-pretty">A planned arrival batch can span several steps. One recorded step can include requests from different arrival batches.</p>
      </header>
      {steps.length === 0 ? <p className="meta">No scheduler step is in the applied prefix yet.</p> : null}
      <ol>
        {steps.map((step) => {
          const selected = selection?.kind === "step" && selection.id === step.id
          return (
            <li key={`${step.id}-${step.sequence}`}>
              <button
                type="button"
                className={cn("step-button", highlightedStepIds.has(step.id) && "highlighted")}
                aria-pressed={selected}
                onClick={() => onSelect({ kind: "step", id: step.id })}
              >
                <span className="mono">{step.id}</span>
                <span className="mono">{step.totalScheduledTokens} scheduled · budget {step.tokenBudget}</span>
                <span>
                  {step.requests.map((request) => labels.get(request.requestId) ?? request.requestId).join(", ") || "No participants"}
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
