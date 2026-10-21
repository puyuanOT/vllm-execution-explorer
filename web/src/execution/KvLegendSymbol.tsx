import { BLOCK_STATE_COLORS } from "../architecture/model.ts"

type SymbolKind = "referenced" | "cached" | "available" | "null" | "selected"

/** Decorative samples of the block geometry; adjacent text supplies the meaning. */
export function KvLegendSymbol({ kind }: { kind: SymbolKind }) {
  const height = kind === "available" || kind === "null" ? 1 : kind === "cached" ? 4 : 8
  const front = 22 - height
  const back = front - 7
  const neutral = kind === "available" ? BLOCK_STATE_COLORS.available
    : kind === "cached" ? BLOCK_STATE_COLORS.cached
      : kind === "null" ? BLOCK_STATE_COLORS.null : BLOCK_STATE_COLORS.referenced
  return <svg viewBox="0 0 34 26" className="h-6 w-8 shrink-0" aria-hidden="true" focusable="false">
    <path d={`M3 ${front}H25V22H3Z M25 ${front}L31 ${back}V15L25 22Z`} fill={neutral} stroke="currentColor" strokeWidth="0.7" />
    <path d={`M3 ${front}L9 ${back}H31L25 ${front}Z`} fill={neutral} stroke="currentColor" strokeWidth="0.7" />
    {kind === "cached" ? <path d={`M8 ${front - 2}L13 ${back + 2}M14 ${front - 2}L19 ${back + 2}M20 ${front - 2}L25 ${back + 2}`} stroke="currentColor" strokeWidth="1.5" /> : null}
    {kind === "null" ? <path d={`M11 ${back + 1.5}L22 ${front - 1.5}M14 ${front - 1.5}L25 ${back + 1.5}`} stroke="currentColor" strokeWidth="1.5" /> : null}
    {kind === "selected" ? <path d={`M1 ${front + 3}V${front - 1}H5 M6 ${back - 1}H9V${back - 4} M28 ${back - 2}H33V${back + 2} M23 ${front + 2}H27V${front - 1}`} className="text-accent" fill="none" stroke="currentColor" strokeWidth="1.5" /> : null}
  </svg>
}
