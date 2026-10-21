import { useEffect, useState, type ReactNode } from "react"

import { cn } from "../lib/cn.ts"
import { EvidenceTag, type EvidenceKind } from "../ui/controls.tsx"

export const inputClass = "block mt-1.5 w-full rounded border border-slate-700 bg-[#060b13] px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus-visible:border-accent-400/70 focus-visible:ring-2 focus-visible:ring-accent-400/20 disabled:cursor-not-allowed disabled:opacity-50"
export const compactInputClass = "block w-full rounded border border-slate-700 bg-[#060b13] px-2.5 py-1.5 text-[13px] leading-5 text-slate-100 outline-none placeholder:text-slate-500 focus-visible:border-accent-400/70 focus-visible:ring-2 focus-visible:ring-accent-400/20 aria-[invalid=true]:border-red-500/60 disabled:cursor-not-allowed disabled:opacity-50"
export const iconButtonClass = "grid size-7 min-h-0 shrink-0 place-items-center rounded-none border-0 bg-transparent p-0 text-faint outline-none hover:text-ink focus-visible:ring-2 focus-visible:ring-accent-400/40 disabled:cursor-not-allowed disabled:opacity-50"
export const buttonClass = "inline-flex min-h-8 items-center justify-center gap-2 rounded border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-[11px] font-medium uppercase tracking-[0.06em] text-slate-100 outline-none hover:border-slate-600 focus-visible:border-accent-400 focus-visible:ring-2 focus-visible:ring-accent-400/25 disabled:cursor-not-allowed disabled:opacity-50"
export const primaryButtonClass = "inline-flex min-h-9 items-center justify-center gap-2 border-0 bg-accent px-5 py-2 font-mono text-xs font-semibold uppercase tracking-[0.1em] text-[#021116] outline-none [clip-path:polygon(6px_0,100%_0,100%_calc(100%-6px),calc(100%-6px)_100%,0_100%,0_6px)] focus-visible:[clip-path:none] focus-visible:ring-2 focus-visible:ring-accent-200 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:bg-[#1a2537] disabled:text-[#6f819c]"
export const dangerButtonClass = "inline-flex min-h-8 items-center justify-center gap-2 rounded border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-[11px] font-medium uppercase tracking-[0.06em] text-red-300 outline-none hover:border-red-800 focus-visible:border-red-500 focus-visible:ring-2 focus-visible:ring-red-500/30 disabled:cursor-not-allowed disabled:opacity-50"

export function Details({
  defaultOpen = false,
  forceOpen = false,
  className,
  children,
}: {
  defaultOpen?: boolean
  forceOpen?: boolean
  className?: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen || forceOpen)
  useEffect(() => {
    if (forceOpen) setOpen(true)
  }, [forceOpen])
  return (
    <details className={className} open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      {children}
    </details>
  )
}

export function ConfigureSection({
  number,
  title,
  summary,
  status,
  statusTone = "neutral",
  defaultOpen = false,
  forceOpen = false,
  children,
}: {
  number: number
  title: string
  summary: string
  status?: string
  statusTone?: "neutral" | "ready" | "warning" | "error"
  defaultOpen?: boolean
  forceOpen?: boolean
  children: ReactNode
}) {
  const headingId = `configure-section-${number}-heading`
  return (
    <section id={`configure-section-${number}`} className="scroll-mt-4 border border-slate-800 bg-surface" aria-labelledby={headingId}>
      <h2 id={headingId} className="sr-only text-balance">{title}</h2>
      <Details
        className="group"
        defaultOpen={defaultOpen}
        forceOpen={forceOpen}
      >
        <summary className="cursor-pointer list-none px-5 py-4 outline-none hover:bg-slate-900/60 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-400 [&::-webkit-details-marker]:hidden">
          <div className="flex items-start gap-4">
            <span className="mt-0.5 shrink-0 font-mono text-xs font-medium tabular-nums text-slate-500 group-open:text-accent" aria-hidden="true">
              {String(number).padStart(2, "0")}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-balance text-base font-semibold text-slate-100">{title}</span>
                {status ? <StatusPill tone={statusTone}>{status}</StatusPill> : null}
              </span>
              <span className="mt-1 block text-pretty text-sm text-slate-400">{summary}</span>
            </span>
            <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-slate-500 group-open:hidden" aria-hidden="true">Show</span>
            <span className="hidden font-mono text-[11px] uppercase tracking-[0.08em] text-slate-500 group-open:inline" aria-hidden="true">Hide</span>
          </div>
        </summary>
        <div className="border-t border-slate-800 px-5 py-5">{children}</div>
      </Details>
    </section>
  )
}

const STATUS_KIND: Record<"neutral" | "ready" | "warning" | "error", EvidenceKind> = {
  neutral: "neutral",
  ready: "gpu",
  warning: "warning",
  error: "error",
}

export function StatusPill({
  tone,
  children,
}: {
  tone: "neutral" | "ready" | "warning" | "error"
  children: ReactNode
}) {
  return <EvidenceTag kind={STATUS_KIND[tone]}>{children}</EvidenceTag>
}

export function FieldError({ id, message }: { id: string; message: string }) {
  if (!message) return null
  return <p id={id} className="mt-1.5 text-pretty text-xs text-red-300" role="alert">{message}</p>
}

export function FieldHint({ id, children }: { id?: string; children: ReactNode }) {
  return <p id={id} className="mt-1.5 text-pretty text-xs leading-5 text-slate-400">{children}</p>
}

export function ChoiceCard({
  name,
  value,
  checked,
  title,
  description,
  onChange,
}: {
  name: string
  value: string
  checked: boolean
  title: string
  description: string
  onChange: () => void
}) {
  return (
    <label className={cn(
      "flex cursor-pointer gap-3 border p-3 outline-none focus-within:ring-2 focus-within:ring-accent-400 focus-within:ring-offset-2 focus-within:ring-offset-slate-950",
      checked ? "border-accent-400/50 bg-accent-400/[0.06] shadow-[inset_2px_0_0_theme(colors.accent.400)]" : "border-slate-800 bg-slate-950 hover:border-slate-700",
    )}>
      <input
        type="radio"
        className="mt-0.5 size-4 shrink-0 accent-accent-400"
        name={name}
        value={value}
        checked={checked}
        onChange={onChange}
      />
      <span>
        <span className="block text-sm font-medium text-slate-200">{title}</span>
        <span className="mt-1 block text-pretty text-xs leading-5 text-slate-400">{description}</span>
      </span>
    </label>
  )
}

export function CheckboxField({
  id,
  checked,
  label,
  description,
  disabled,
  onChange,
}: {
  id: string
  checked: boolean
  label: string
  description?: string
  disabled?: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label htmlFor={id} className={cn("flex gap-3", disabled && "opacity-50")}>
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 size-4 shrink-0 accent-accent-400"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        <span className="block text-sm font-medium text-slate-200">{label}</span>
        {description ? <span className="mt-1 block text-pretty text-xs leading-5 text-slate-400">{description}</span> : null}
      </span>
    </label>
  )
}
