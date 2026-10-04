import { Archive, Disc3, FlaskConical } from "lucide-react"

import { cn } from "../lib/cn.ts"
import type { AppRoute } from "../hooks/useAppRoute.ts"
import { Chip, Led, type Tone } from "../ui/controls.tsx"

export type HeaderEvidence = "synthetic" | "gpu" | "draft" | "recorded" | null

export function AppHeader({
  route,
  status,
  recording,
  evidence,
  controller,
  onNavigate,
  onRecordings,
}: {
  route: AppRoute
  status: string
  recording?: string | null
  evidence?: HeaderEvidence
  controller: { label: string; tone: Tone }
  onNavigate: (route: AppRoute) => void
  onRecordings: () => void
}) {
  return (
    <header className="flex min-h-[52px] flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-[#060a11] px-4 py-1.5">
      <div className="mr-1 flex min-w-0 items-center gap-3">
        <div className="grid size-[30px] shrink-0 place-items-center border border-[#2a3c58]" aria-hidden="true">
          <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="#95a5bb" strokeWidth="1.3" strokeLinejoin="round">
            <path d="M12 2.6 20.4 7.3v9.4L12 21.4 3.6 16.7V7.3Z" />
            <path d="M3.6 7.3 12 12l8.4-4.7M12 12v9.4" opacity=".5" />
            <rect x="10" y="10" width="4" height="4" fill="#3fcde6" stroke="none" transform="rotate(45 12 12)" />
          </svg>
        </div>
        <div className="min-w-0">
          <p className="m-0 truncate text-[14.5px] font-semibold leading-tight text-ink">vLLM <span className="font-medium text-muted">Execution Explorer</span></p>
          <p className="m-0 truncate font-mono text-[9px] uppercase tracking-[0.14em] text-faint"><b className="font-semibold text-muted">vLLM 0.30.0</b> · observed serving mechanics</p>
        </div>
      </div>

      <nav className="flex self-stretch" aria-label="Primary">
        {(["configure", "explore"] as const).map((item, index) => (
          <button
            key={item}
            type="button"
            className={cn(
              "flex min-h-11 items-center gap-2 rounded-none border-0 bg-transparent px-4 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-faint hover:text-ink",
              route === item && "text-ink shadow-[inset_0_-2px_0_theme(colors.accent.400)]",
            )}
            aria-current={route === item ? "page" : undefined}
            onClick={() => onNavigate(item)}
          >
            <span className={route === item ? "text-accent" : "text-[#3a4960]"}>{String(index + 1).padStart(2, "0")}</span>
            {item === "configure" ? "Configure" : "Explore"}
          </button>
        ))}
      </nav>

      <span className="flex-1" />

      <div className="flex min-w-0 flex-wrap items-center gap-2">
        {recording ? <Chip className="hidden max-w-64 sm:inline-flex" title={recording}><Disc3 className="size-3.5 shrink-0" aria-hidden="true" /><span className="text-ink">{recording}</span></Chip> : null}
        {evidence === "synthetic" ? <Chip tone="synthetic"><FlaskConical className="size-3.5 shrink-0" aria-hidden="true" /><span>Synthetic fixture · not GPU evidence</span></Chip> : null}
        {evidence === "gpu" ? <Chip tone="gpu"><Led /><span>Observed GPU recording</span></Chip> : null}
        {evidence === "draft" ? <Chip tone="warning"><span>Draft preview · not execution evidence</span></Chip> : null}
        <span className="flex h-6 max-w-72 items-center gap-2 border-line font-mono text-[10.5px] text-faint sm:border-l sm:pl-3" role="status" title={status}>
          <Led tone={controller.tone} />
          <span className="truncate"><b className="sr-only font-medium text-muted sm:not-sr-only">{controller.label}</b><span className="hidden lg:inline">{status ? ` · ${status}` : ""}</span></span>
        </span>
        <button
          type="button"
          className="grid size-8 place-items-center border-line-strong bg-raised p-0 text-muted hover:text-ink"
          aria-label="Open recordings"
          title="Recordings"
          onClick={onRecordings}
        >
          <Archive aria-hidden="true" className="size-4" />
        </button>
      </div>
    </header>
  )
}
