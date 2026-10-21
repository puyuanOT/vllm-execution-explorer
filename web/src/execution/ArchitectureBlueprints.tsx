/** Model-card architecture only. These blueprints never supply replay state. */
export function ArchitectureBlueprints() {
  return <details className="border-t border-line pt-2">
    <summary className="cursor-pointer text-sm text-ink">Explore hybrid architecture blueprints</summary>
    <div className="mt-3 grid gap-3 text-pretty text-xs text-muted">
      <p className="m-0">Static educational views; no execution capture. These do not change the recorded model, its cache, or the playhead. Request activity, recurrent state, expert routing and numerical values require a capture of the selected architecture.</p>
      <details><summary className="cursor-pointer text-sm text-ink">Qwen3.8-27B · dense hybrid</summary>
        <div className="mt-2 grid gap-2">
          <p className="m-0">64 language-model layers: 48 GatedDeltaNet layers and 16 gated attention layers. Each layer includes a dense feed-forward network.</p>
          <ol className="m-0 list-decimal space-y-1 pl-5" aria-label="Dense hybrid repeating layer group">
            <li>GatedDeltaNet → dense feed-forward</li><li>GatedDeltaNet → dense feed-forward</li><li>GatedDeltaNet → dense feed-forward</li><li>Gated attention → dense feed-forward</li>
          </ol>
          <p className="m-0">Repeat this group 16 times. Recurrent GatedDeltaNet state and attention KV memory are distinct structures; this recording does not measure either for this blueprint.</p>
          <a className="text-accent underline" href="https://huggingface.co/Qwen/Qwen3.8-27B" target="_blank" rel="noreferrer">Official model card</a>
        </div>
      </details>
      <details><summary className="cursor-pointer text-sm text-ink">Qwen3.8-Flash-Next-FP8 · hybrid MoE</summary>
        <div className="mt-2 grid gap-2">
          <p className="m-0">48 layers: 36 GatedDeltaNet layers and 12 Qwen sparse attention layers. Each layer includes a mixture of experts, with 512 experts and 10 routed plus 1 shared expert activated per token.</p>
          <ol className="m-0 list-decimal space-y-1 pl-5" aria-label="Hybrid MoE repeating layer group">
            <li>GatedDeltaNet → MoE</li><li>GatedDeltaNet → MoE</li><li>GatedDeltaNet → MoE</li><li>Qwen sparse attention → MoE</li>
          </ol>
          <p className="m-0">Repeat this group 12 times. The model card specifies the architecture; which experts serve a token and how sparse attention selects memory remain unobserved here.</p>
          <a className="text-accent underline" href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8" target="_blank" rel="noreferrer">Official model card</a>
        </div>
      </details>
    </div>
  </details>
}
