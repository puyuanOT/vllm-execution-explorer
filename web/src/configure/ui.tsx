import { useState, type ReactNode } from "react"

import { cn } from "../lib/cn.ts"

export const inputClass = "mt-1.5 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-400 focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500/30 disabled:cursor-not-allowed disabled:opacity-50"
export const buttonClass = "inline-flex min-h-9 items-center justify-center rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-medium text-slate-200 outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500/30 disabled:cursor-not-allowed disabled:opacity-50"
export const primaryButtonClass = "inline-flex min-h-10 items-center justify-center rounded-md border border-blue-500 bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-50"
export const dangerButtonClass = "inline-flex min-h-9 items-center justify-center rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-medium text-red-300 outline-none focus-visible:border-red-500 focus-visible:ring-2 focus-visible:ring-red-500/30 disabled:cursor-not-allowed disabled:opacity-50"

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
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900 shadow-sm" aria-labelledby={headingId}>
      <h2 id={headingId} className="sr-only text-balance">{title}</h2>
      <details
        className="group"
        open={forceOpen || open}
        onToggle={(event) => {
          if (!forceOpen) setOpen(event.currentTarget.open)
        }}
      >
        <summary className="cursor-pointer list-none rounded-xl px-5 py-4 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500 [&::-webkit-details-marker]:hidden">
          <div className="flex items-start gap-4">
            <span className="flex size-7 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-950 text-xs font-semibold tabular-nums text-slate-400" aria-hidden="true">
              {number}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-balance text-base font-semibold text-slate-100">{title}</span>
                {status ? <StatusPill tone={statusTone}>{status}</StatusPill> : null}
              </span>
              <span className="mt-1 block text-pretty text-sm text-slate-400">{summary}</span>
            </span>
            <span className="text-sm text-slate-400 group-open:hidden" aria-hidden="true">Show</span>
            <span className="hidden text-sm text-slate-400 group-open:inline" aria-hidden="true">Hide</span>
          </div>
        </summary>
        <div className="border-t border-slate-800 px-5 py-5">{children}</div>
      </details>
    </section>
  )
}

export function StatusPill({
  tone,
  children,
}: {
  tone: "neutral" | "ready" | "warning" | "error"
  children: ReactNode
}) {
  return (
    <span
      className={cn(
        "inline-flex min-h-6 items-center rounded-full border px-2 py-0.5 text-xs font-medium",
        tone === "neutral" && "border-slate-700 bg-slate-950 text-slate-400",
        tone === "ready" && "border-emerald-900 bg-emerald-950 text-emerald-300",
        tone === "warning" && "border-amber-900 bg-amber-950 text-amber-300",
        tone === "error" && "border-red-900 bg-red-950 text-red-300",
      )}
    >
      {children}
    </span>
  )
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
      "flex cursor-pointer gap-3 rounded-lg border p-3 outline-none focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2 focus-within:ring-offset-slate-950",
      checked ? "border-blue-500 bg-blue-950/40" : "border-slate-800 bg-slate-950",
    )}>
      <input
        type="radio"
        className="mt-0.5 size-4 shrink-0 accent-blue-500"
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
        className="mt-0.5 size-4 shrink-0 accent-blue-500"
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
