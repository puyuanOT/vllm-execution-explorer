import type { ReactNode } from "react"

import { cn } from "../lib/cn.ts"

export type Tone = "neutral" | "ready" | "warning" | "error" | "accent" | "explain" | "synthetic"
export type EvidenceKind = "observed" | "derived" | "explanatory" | "unknown" | "synthetic" | "gpu" | "warning" | "error" | "neutral"

export function Led({ tone = "ready", label }: { tone?: Tone; label?: string }) {
  return <span className="cp-led" data-tone={tone === "ready" ? undefined : tone} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true} />
}

export function Chip({ tone, children, className, title }: { tone?: "synthetic" | "gpu" | "warning" | "accent"; children: ReactNode; className?: string; title?: string }) {
  return <span className={cn("cp-chip", className)} data-tone={tone} title={title}>{children}</span>
}

export function EvidenceTag({ kind, children }: { kind: EvidenceKind; children: ReactNode }) {
  return <span className="cp-tag" data-kind={kind}>{children}</span>
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-2 flex items-center gap-2">
      <h3 className="cp-label m-0 text-muted">{children}</h3>
      <span className="cp-rule" aria-hidden="true" />
      {aside}
    </div>
  )
}

export interface SegmentOption<T extends string | number> {
  value: T
  label: ReactNode
  disabled?: boolean
  title?: string
}

export function Segmented<T extends string | number>({ label, value, options, onChange, className }: {
  label: string
  value: T
  options: Array<SegmentOption<T>>
  onChange: (value: T) => void
  className?: string
}) {
  return (
    <div className={cn("cp-seg", className)} role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          aria-pressed={option.value === value}
          disabled={option.disabled}
          title={option.title}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function Readouts({ items }: { items: Array<{ label: string; value: ReactNode; detail?: ReactNode }> }) {
  return (
    <div className="cp-readouts" style={{ gridTemplateColumns: `repeat(${items.length}, minmax(0, 1fr))` }}>
      {items.map((item) => (
        <div key={item.label}>
          <div className="cp-readout-value">{item.value}{item.detail ? <small>{item.detail}</small> : null}</div>
          <div className="cp-label mt-1.5">{item.label}</div>
        </div>
      ))}
    </div>
  )
}

/** Plots only the values it is given; callers pass the recorded prefix up to the playhead. */
export function Sparkline({ values, max, width = 88, height = 26, active = false, label }: {
  values: number[]
  max?: number
  width?: number
  height?: number
  active?: boolean
  label?: string
}) {
  const top = Math.max(1, max ?? 0, ...values)
  const step = values.length > 1 ? width / (values.length - 1) : 0
  const points = values.map((value, index) => [index * step, height - 1 - (value / top) * (height - 3)] as const)
  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ")
  const stroke = active ? "var(--accent)" : "#5d6f8a"
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true} className="shrink-0">
      {points.length > 1 ? (
        <>
          <path d={`${line} L${points.at(-1)![0].toFixed(1)},${height} L0,${height} Z`} fill={stroke} opacity={0.08} />
          <path d={line} fill="none" stroke={stroke} strokeWidth={1.2} strokeLinejoin="round" />
        </>
      ) : null}
      {points.length > 0 ? <circle cx={points.at(-1)![0]} cy={points.at(-1)![1]} r={1.8} fill={stroke} /> : null}
    </svg>
  )
}
