import { Archive, Boxes, Settings2 } from "lucide-react"

import { cn } from "../lib/cn.ts"
import type { AppRoute } from "../hooks/useAppRoute.ts"

export function AppHeader({
  route,
  status,
  onNavigate,
  onRecordings,
}: {
  route: AppRoute
  status: string
  onNavigate: (route: AppRoute) => void
  onRecordings: () => void
}) {
  return (
    <header className="flex min-h-16 flex-wrap items-center gap-3 border-b border-slate-800 bg-slate-950 px-4 py-3 sm:px-6">
      <div className="mr-auto flex min-w-0 items-center gap-3">
        <div className="grid size-9 shrink-0 place-items-center rounded-lg border border-slate-700 bg-slate-900 text-sky-300">
          <Boxes aria-hidden="true" className="size-5" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-balance text-base font-semibold text-slate-100">vLLM Execution Explorer</p>
          <p className="truncate text-xs text-slate-400">Observed serving mechanics</p>
        </div>
      </div>

      <nav className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1" aria-label="Primary">
        <button
          type="button"
          className={cn(
            "rounded-md px-3 py-1.5 text-sm text-slate-300 hover:text-white",
            route === "configure" && "bg-slate-700 text-white",
          )}
          aria-current={route === "configure" ? "page" : undefined}
          onClick={() => onNavigate("configure")}
        >
          <Settings2 aria-hidden="true" className="mr-1.5 inline size-4" />
          Configure
        </button>
        <button
          type="button"
          className={cn(
            "rounded-md px-3 py-1.5 text-sm text-slate-300 hover:text-white",
            route === "explore" && "bg-slate-700 text-white",
          )}
          aria-current={route === "explore" ? "page" : undefined}
          onClick={() => onNavigate("explore")}
        >
          <Boxes aria-hidden="true" className="mr-1.5 inline size-4" />
          Explore
        </button>
      </nav>

      <p className="hidden max-w-64 truncate text-xs text-slate-400 lg:block" role="status">{status}</p>
      <button
        type="button"
        className="grid size-9 place-items-center rounded-lg border border-slate-700 bg-slate-900 text-slate-300 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400"
        aria-label="Open recordings"
        onClick={onRecordings}
      >
        <Archive aria-hidden="true" className="size-4" />
      </button>
    </header>
  )
}
