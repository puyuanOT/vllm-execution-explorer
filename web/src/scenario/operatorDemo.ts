import corpus from "./operatorDemo.json" with { type: "json" }
import type { ScenarioDraft } from "./types.ts"

/** New capture inputs; saved recordings continue to retain their original prompts. */
export function operatorDemoScenario({ loraMode = null, adapterId }: { loraMode?: "none" | "all" | "mixed" | null; adapterId?: string } = {}): ScenarioDraft {
  return {
    version: 1, id: `scenario-operator-demo-v${corpus.version}`, name: corpus.name,
    nextOrder: corpus.batches.length,
    batches: corpus.batches.map((batch, createdOrder) => ({
      id: batch.id, label: batch.label,
      arrivalOffsetMs: batch.arrival_offset_ms + (createdOrder > 0 && (loraMode === "all" || loraMode === "mixed") ? 20000 : 0),
      createdOrder, defaultAdapterId: null,
      samples: batch.request_ids.map((id) => {
        const request = corpus.requests.find((row) => row.id === id)!
        const document = corpus.documents[request.document as keyof typeof corpus.documents]
        return {
          id: request.id, label: request.label,
          prompt: `<|im_start|>user\n${document}\n\nTask\n${request.task}<|im_end|>\n<|im_start|>assistant\n`,
          inputMode: "prompt" as const, syntheticInputTokens: 128, resolvedInputTokens: null,
          maxOutputTokens: request.max_output_tokens,
          adapterId: loraMode == null ? null : loraMode === "all" || loraMode === "mixed" && ["request-3", "request-4", "request-6"].includes(id)
            ? adapterId ?? null : "__base__",
          priority: 0,
        }
      }),
    })),
  }
}
