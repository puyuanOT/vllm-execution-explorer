import { Activity, Gauge, GitMerge, Scissors, Undo2 } from "lucide-react"
import { useEffect, useState, type ComponentType } from "react"

import { cn } from "../lib/cn.ts"
import { LEARNING_EXAMPLES, type LearningExampleId } from "../scenario/learningExamples.ts"
import { EvidenceTag, Led, SectionTitle, type Tone } from "../ui/controls.tsx"
import type { ReadinessState } from "./types.ts"
import { buttonClass } from "./ui.tsx"

export interface ConsoleModule {
  number: number
  title: string
  summary: string
  tone: ReadinessState["tone"]
}

const LED_TONE: Record<ReadinessState["tone"], Tone> = { neutral: "neutral", ready: "ready", warning: "warning", error: "error" }
const TONE_LABEL: Record<ReadinessState["tone"], string> = { neutral: "unchecked", ready: "ready", warning: "needs attention", error: "blocked" }
const EXAMPLE_ICONS: Record<LearningExampleId, ComponentType<{ className?: string; "aria-hidden"?: boolean }>> = {
  staggered: Activity,
  prefix: GitMerge,
  chunked: Scissors,
  pressure: Gauge,
}

/** Opens and scrolls to a Configure section; the sections stay the single source of form state. */
export function openConfigureSection(number: number) {
  const section = document.getElementById(`configure-section-${number}`)
  const details = section?.querySelector("details")
  if (details && !details.open) details.open = true
  section?.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" })
  section?.querySelector<HTMLElement>("summary")?.focus({ preventScroll: true })
}

/** The section being edited: it follows focus, and falls back to the topmost visible section once it scrolls away. */
function useEditedSection(numbers: number[]): number {
  const [active, setActive] = useState(numbers[0] ?? 1)
  const key = numbers.join(",")
  useEffect(() => {
    const sectionNumber = (element: Element | null) => {
      const section = element?.closest("[id^='configure-section-']")
      return section ? Number(section.id.replace("configure-section-", "")) : null
    }
    const visible = (number: number) => {
      const rect = document.getElementById(`configure-section-${number}`)?.getBoundingClientRect()
      return rect ? rect.bottom > 64 && rect.top < window.innerHeight - 96 : false
    }
    let frame = 0
    const onFocus = (event: FocusEvent) => {
      const number = sectionNumber(event.target instanceof Element ? event.target : null)
      if (number != null) setActive(number)
    }
    const onScroll = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => setActive((current) => (
        visible(current) || sectionNumber(document.activeElement) === current ? current : numbers.find(visible) ?? current
      )))
    }
    document.addEventListener("focusin", onFocus)
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener("focusin", onFocus)
      window.removeEventListener("scroll", onScroll)
    }
  }, [key])
  return active
}

export function ConsoleRail({ modules, loadedExample, disabled, canUndo, storageNote, storageError, onLoadExample, onUndo, onStartEmpty, onReset }: {
  modules: ConsoleModule[]
  loadedExample: LearningExampleId | null
  disabled: boolean
  canUndo: boolean
  storageNote?: string
  storageError?: string
  onLoadExample: (id: LearningExampleId) => void
  onUndo?: () => void
  onStartEmpty?: () => void
  onReset?: () => void
}) {
  const active = useEditedSection(modules.map((module) => module.number))
  return (
    <nav className="sticky top-4 hidden max-h-[calc(100dvh-140px)] flex-col gap-5 self-start overflow-y-auto pb-2 xl:flex" aria-label="Capture console">
      <div>
        <p className="cp-label m-0 mb-2 px-1">Capture console</p>
        <ol className="m-0 grid list-none gap-1 p-0">
          {modules.map((module) => {
            const current = module.number === active
            return (
              <li key={module.number}>
                <button
                  type="button"
                  className={cn(
                    "grid w-full grid-cols-[24px_minmax(0,1fr)_auto] items-center gap-x-2.5 gap-y-0.5 rounded-none border px-2.5 py-2.5 text-left",
                    current ? "border-line-strong bg-accent-400/[0.05] shadow-[inset_2px_0_0_theme(colors.accent.400)]" : "border-transparent bg-transparent hover:border-line-strong hover:bg-surface",
                  )}
                  aria-current={current ? "true" : undefined}
                  onClick={() => openConfigureSection(module.number)}
                >
                  <span className={cn("row-span-2 font-mono text-xs font-medium tabular-nums", current ? "text-accent" : "text-[#3a4960]")} aria-hidden="true">{String(module.number).padStart(2, "0")}</span>
                  <span className={cn("truncate font-mono text-[11px] font-medium uppercase tracking-[0.08em]", current ? "text-ink" : "text-muted")}>{module.title}</span>
                  <Led tone={LED_TONE[module.tone]} label={`${module.title}: ${TONE_LABEL[module.tone]}`} />
                  <span className="col-span-2 truncate text-[11.5px] text-faint">{module.summary}</span>
                </button>
              </li>
            )
          })}
        </ol>
      </div>

      <div>
        <SectionTitle>Learning examples</SectionTitle>
        <ul className="m-0 grid list-none gap-1 p-0">
          {LEARNING_EXAMPLES.map((example) => {
            const Icon = EXAMPLE_ICONS[example.id]
            const loaded = example.id === loadedExample
            return (
              <li key={example.id}>
                <button
                  type="button"
                  className={cn(
                    "flex w-full items-center gap-2 rounded-none border bg-transparent px-2.5 py-1.5 text-left text-xs",
                    loaded ? "border-slate-600 text-ink" : "border-line text-muted hover:text-ink",
                  )}
                  disabled={disabled}
                  title={`${example.description} Replaces the draft and applies the example's scheduling and cache settings.`}
                  onClick={() => onLoadExample(example.id)}
                >
                  <Icon className={cn("size-3.5 shrink-0", loaded ? "text-muted" : "text-faint")} aria-hidden={true} />
                  <span className="min-w-0 flex-1 truncate">{example.label}</span>
                  {loaded ? <EvidenceTag kind="neutral">Loaded</EvidenceTag> : null}
                </button>
              </li>
            )
          })}
        </ul>
        <p className="m-0 mt-2 px-0.5 text-pretty text-[11px] leading-4 text-faint">Examples build an editable workload for live vLLM. Open the real GPU demo in the launch bar to explore a saved run.</p>
      </div>

      {onUndo || onStartEmpty || onReset ? (
        <div>
          <SectionTitle>Draft</SectionTitle>
          <div className="flex flex-wrap gap-1.5">
            {onUndo ? <button className={buttonClass} type="button" disabled={!canUndo || disabled} onClick={onUndo}><Undo2 className="size-3.5" aria-hidden="true" />Undo deletion</button> : null}
            {onStartEmpty ? <button className={buttonClass} type="button" disabled={disabled} onClick={onStartEmpty}>Start empty</button> : null}
            {onReset ? <button className={buttonClass} type="button" disabled={disabled} onClick={onReset}>Reset</button> : null}
          </div>
          <p className={cn("m-0 mt-2 px-0.5 text-pretty font-mono text-[10px] tracking-[0.04em]", storageError ? "text-red-300" : "text-faint")} role={storageError ? "alert" : undefined}>
            {storageError || `${storageNote || "Saved on this device"} · secrets never stored`}
          </p>
        </div>
      ) : null}
    </nav>
  )
}
