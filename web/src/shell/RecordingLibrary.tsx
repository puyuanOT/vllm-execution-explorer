import { FileUp, History, Play } from "lucide-react"

import type { CatalogEntry } from "../api.ts"
import { cn } from "../lib/cn.ts"

export function RecordingLibrary({
  catalog,
  catalogError,
  loadError,
  activeId,
  recent,
  onOpenRecent,
  onOpen,
  onFiles,
}: {
  catalog: CatalogEntry[]
  catalogError: string
  loadError: string
  activeId: string
  recent?: { id: string; label: string } | null
  onOpenRecent?: () => void
  onOpen: (id: string) => void
  onFiles: (files: FileList) => void
}) {
  return (
    <div className="grid gap-5">
      {(catalogError || loadError) ? (
        <p className="rounded-lg border border-rose-900 bg-rose-950/40 px-3 py-2 text-pretty text-sm text-rose-200" role="alert">
          {loadError || catalogError}
        </p>
      ) : null}

      {recent && onOpenRecent ? (
        <button
          type="button"
          aria-pressed={activeId === recent.id}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg border border-slate-800 bg-slate-900 p-3 text-left hover:border-slate-600",
            activeId === recent.id && "border-sky-500",
          )}
          onClick={onOpenRecent}
        >
          <History aria-hidden="true" className="size-4 shrink-0 text-sky-300" />
          <span className="truncate text-sm font-medium text-slate-100">{recent.label}</span>
        </button>
      ) : null}

      <section aria-labelledby="lesson-heading">
        <h3 id="lesson-heading" className="text-balance text-sm font-semibold text-slate-200">Bundled lessons</h3>
        {catalog.length > 0 ? (
          <ul className="mt-2 grid gap-2">
            {catalog.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  aria-pressed={activeId === item.id}
                  className={cn(
                    "flex w-full items-start gap-3 rounded-lg border border-slate-800 bg-slate-900 p-3 text-left hover:border-slate-600",
                    activeId === item.id && "border-sky-500",
                  )}
                  onClick={() => onOpen(item.id)}
                >
                  <Play aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-sky-300" />
                  <span className="min-w-0">
                    <strong className="block truncate text-sm text-slate-100">{item.title}</strong>
                    <span className="mt-1 line-clamp-2 block text-pretty text-xs text-slate-400">{item.summary}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-2 rounded-lg border border-dashed border-slate-700 p-3 text-pretty text-sm text-slate-400">
            No bundled lesson is available. Import a recording below.
          </p>
        )}
      </section>

      <label className="grid cursor-pointer gap-2 rounded-lg border border-dashed border-slate-700 bg-slate-900 p-4 text-sm text-slate-300 hover:border-slate-500">
        <span className="flex items-center gap-2 font-medium text-slate-200">
          <FileUp aria-hidden="true" className="size-4 text-sky-300" />
          Import a recording
        </span>
        <span className="text-pretty text-xs text-slate-400">
          Choose manifest.json, events.jsonl, optional content.jsonl, and any declared normalized profile JSON from the same recording.
        </span>
        <input
          type="file"
          multiple
          accept=".json,.jsonl,application/json"
          className="block w-full text-xs text-slate-400 file:mr-3 file:rounded-md file:border file:border-slate-700 file:bg-slate-800 file:px-3 file:py-1.5 file:text-slate-200"
          onChange={(event) => {
            if (event.target.files && event.target.files.length > 0) onFiles(event.target.files)
          }}
        />
      </label>
    </div>
  )
}
