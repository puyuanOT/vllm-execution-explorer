import { Menu } from "@base-ui/react/menu"
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pause,
  Play,
  Radio,
  RotateCcw,
} from "lucide-react"
import { useLayoutEffect, useMemo, useRef, useState } from "react"

import { cn } from "../lib/cn.ts"
import { playbackSpeeds } from "../playback.ts"
import { EvidenceTag, Segmented } from "../ui/controls.tsx"
import { buildScope, SCOPE_CHANNELS, type ScopeMarker, type ScopeTick } from "./scope.ts"
import type { PlaybackMode } from "./types.ts"
import type { TimePoint } from "../../../replay/src/timeline.ts"
import { RecordedTimeAxis } from "./RecordedTimeAxis.tsx"

const GUTTER = 34
const ROW = 8
const ROW_GAP = 3
const TOP = 14

export function PlaybackTransport({
  disabled,
  playing,
  atEnd,
  speed,
  mode,
  recordedAvailable,
  measuredAvailable,
  iterationsAvailable,
  timePoints,
  eventLabels,
  sequence,
  maxSequence,
  valueText,
  followLive,
  followAvailable,
  connected,
  ticks,
  markers,
  gapPosition,
  currentPosition,
  onTogglePlay,
  onReplay,
  onPreviousEvent,
  onNextEvent,
  onPreviousStep,
  onNextStep,
  onScrub,
  onSpeed,
  onMode,
  onFollow,
}: {
  disabled: boolean
  playing: boolean
  atEnd: boolean
  speed: number
  mode: PlaybackMode
  recordedAvailable: boolean
  measuredAvailable: boolean
  iterationsAvailable: boolean
  timePoints: TimePoint[]
  eventLabels: ReadonlyMap<number, string>
  sequence: number
  maxSequence: number
  valueText: string
  followLive: boolean
  followAvailable: boolean
  connected: boolean
  ticks: ScopeTick[]
  markers: ScopeMarker[]
  gapPosition: number | null
  currentPosition: number | null
  onTogglePlay: () => void
  onReplay: () => void
  onPreviousEvent: () => void
  onNextEvent: () => void
  onPreviousStep: () => void
  onNextStep: () => void
  onScrub: (sequence: number) => void
  onSpeed: (speed: number) => void
  onMode: (mode: PlaybackMode) => void
  onFollow: () => void
}) {
  const playbackDisabled = disabled || maxSequence <= 0
  const playLabel = playing ? "Pause" : atEnd ? "Replay" : "Play"
  const play = () => {
    if (!playing && atEnd) onReplay()
    else onTogglePlay()
  }
  const iconButton = "grid size-8 shrink-0 place-items-center border-line-strong bg-raised p-0 text-muted hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
  const measured = mode === "measured"
  const value = Math.min(sequence, Math.max(maxSequence, 0))
  const readout = measured ? `${(sequence / 1000).toFixed(3)} / ${(maxSequence / 1000).toFixed(3)} ms` : `EVT ${value} / ${Math.max(maxSequence, 0)}`
  const note = measured
    ? "GPU activity highlights recorded active layers. Host replay and memory keep their own playhead."
    : mode === "iterations"
      ? "Iteration stepping includes result-associated KV and output observations. Raw events retain every intermediate transition in observer order."
      : mode === "events"
        ? "Raw events advance through recorded observations in order; playback speed is illustrative."
        : "Host time preserves observed host intervals at the selected speed; it is not GPU kernel timing."
  const caveat = measured ? "GPU activity · separate from host"
    : mode === "iterations" ? "Iteration boundaries · observer order"
      : mode === "events" ? "Event order · not elapsed time"
        : "Host intervals · not GPU kernel time"
  return (
    <div className="grid gap-1 border-t border-line bg-[#060a11] px-4 pb-[calc(0.5rem+env(safe-area-inset-bottom))] pt-2" role="group" aria-label="Recording playback">
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex shrink-0 items-center gap-1">
          <button type="button" className={iconButton} onClick={onReplay} disabled={playbackDisabled} aria-label={measured ? "Restart profiler playback" : "Restart recording"} title="Restart">
            <RotateCcw className="size-4" aria-hidden="true" />
          </button>
          <span className="mx-1 h-5 w-px bg-line-strong" aria-hidden="true" />
          <button type="button" className={iconButton} onClick={onPreviousStep} disabled={playbackDisabled} aria-label={measured ? "Previous profiled batch" : "Previous scheduler step"} title={measured ? "Previous profiled batch  [" : "Previous scheduler step  ["}>
            <ChevronsLeft className="size-4" aria-hidden="true" />
          </button>
          <button type="button" className={iconButton} onClick={onPreviousEvent} disabled={playbackDisabled} aria-label={measured ? "Previous measured activity" : mode === "iterations" ? "Previous model iteration" : "Previous recorded event"} title={measured ? "Previous measured activity  ←" : mode === "iterations" ? "Previous model iteration  ←" : "Previous recorded event  ←"}>
            <ChevronLeft className="size-4" aria-hidden="true" />
          </button>
          <button
            type="button"
            className="mx-1 inline-flex min-h-8 items-center gap-2 border-0 bg-accent px-4 font-mono text-[11px] font-semibold uppercase tracking-[0.1em] text-[#021116] [clip-path:polygon(6px_0,100%_0,100%_calc(100%-6px),calc(100%-6px)_100%,0_100%,0_6px)] focus-visible:[clip-path:none] disabled:bg-[#1a2537] disabled:text-[#6f819c]"
            aria-label={playLabel}
            aria-pressed={playing}
            disabled={playbackDisabled}
            onClick={play}
            title={`${playLabel}  Space`}
          >
            {playing ? <Pause className="size-4" aria-hidden="true" /> : <Play className="size-4" aria-hidden="true" />}
            {playLabel}
          </button>
          <button type="button" className={iconButton} onClick={onNextEvent} disabled={playbackDisabled} aria-label={measured ? "Next measured activity" : mode === "iterations" ? "Next model iteration" : "Next recorded event"} title={measured ? "Next measured activity  →" : mode === "iterations" ? "Next model iteration  →" : "Next recorded event  →"}>
            <ChevronRight className="size-4" aria-hidden="true" />
          </button>
          <button type="button" className={iconButton} onClick={onNextStep} disabled={playbackDisabled} aria-label={measured ? "Next profiled batch" : "Next scheduler step"} title={measured ? "Next profiled batch  ]" : "Next scheduler step  ]"}>
            <ChevronsRight className="size-4" aria-hidden="true" />
          </button>
          {followAvailable ? (
            <button
              type="button"
              className={cn(
                "ml-1 inline-flex min-h-8 items-center gap-2 border-line-strong bg-raised px-3 font-mono text-[11px] uppercase tracking-[0.06em] text-muted",
                followLive && "border-accent-400/45 text-accent",
              )}
              aria-pressed={followLive}
              onClick={onFollow}
            >
              <Radio className="size-4" aria-hidden="true" />
              {connected ? "Follow live" : "Reconnect live"}
            </button>
          ) : null}
        </div>
        {mode === "recorded" ? <RecordedTimeAxis points={timePoints} sequence={value} labels={eventLabels} onSeek={onScrub} /> : <Scope
          label={measured ? "GPU activity" : "Playhead"}
          readout={readout}
          ticks={ticks}
          markers={markers}
          gapPosition={gapPosition}
          currentPosition={currentPosition}
          value={value}
          maxSequence={maxSequence}
          disabled={playbackDisabled}
          valueText={valueText}
          onScrub={onScrub}
        />}
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1.5">
        <p className="m-0 min-w-0 flex-1 truncate font-mono text-[10.5px] text-faint" title={`${valueText} ${note}${followAvailable ? (connected ? " Live capture continues while replay is paused." : " The live trace is reconnecting.") : ""}`}>
          <span className="text-muted">{valueText}</span>
          <span className="mx-2 text-line-strong" aria-hidden="true">|</span>
          {note}
          {followAvailable ? (connected ? " Live capture continues while replay is paused." : " The live trace is reconnecting.") : ""}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <EvidenceTag kind="neutral">{caveat}</EvidenceTag>
          <span className="cp-label">Clock</span>
          <Segmented<PlaybackMode>
            label="Playback mode"
            value={mode}
            onChange={onMode}
            options={[
              { value: "iterations", label: "Iterations", disabled: !iterationsAvailable, title: "Step through observed model iterations" },
              { value: "measured", label: "GPU activity", disabled: !measuredAvailable, title: measuredAvailable ? "Scrub measured CUDA activity to highlight recorded active layers" : "GPU activity unavailable: this recording has no profiler attachment" },
            ]}
          />
          <Menu.Root>
            <Menu.Trigger className={cn(
              "inline-flex min-h-7 items-center gap-1.5 border-line-strong bg-surface px-2 font-mono text-[11px] text-muted",
              (mode === "events" || mode === "recorded") && "border-accent-400/50 text-accent",
            )}>
              Advanced playback{mode === "events" ? " · Raw events" : mode === "recorded" ? " · Host time" : ""}
              <ChevronDown className="size-3" aria-hidden="true" />
            </Menu.Trigger>
            <Menu.Portal>
              <Menu.Positioner side="top" align="end" sideOffset={8} className="z-30">
                <Menu.Popup className="min-w-64 border border-line-strong bg-surface p-1 shadow-lg outline-none" aria-label="Advanced playback modes">
                  <Menu.RadioGroup value={mode} onValueChange={onMode}>
                    <Menu.RadioItem value="events" label="Raw events" disabled={maxSequence <= 0 && !measuredAvailable} closeOnClick className="flex cursor-default items-start gap-2 px-3 py-2 text-sm text-ink outline-none data-[highlighted]:bg-raised data-[disabled]:opacity-50">
                      <span className="grid size-4 shrink-0 place-items-center"><Menu.RadioItemIndicator><Check className="size-4 text-accent" aria-hidden="true" /></Menu.RadioItemIndicator></span>
                      <span>Raw events<span className="mt-0.5 block text-xs text-muted">Every observation in event order</span></span>
                    </Menu.RadioItem>
                    <Menu.RadioItem value="recorded" label="Host time" disabled={!recordedAvailable} closeOnClick className="flex cursor-default items-start gap-2 px-3 py-2 text-sm text-ink outline-none data-[highlighted]:bg-raised data-[disabled]:opacity-50">
                      <span className="grid size-4 shrink-0 place-items-center"><Menu.RadioItemIndicator><Check className="size-4 text-accent" aria-hidden="true" /></Menu.RadioItemIndicator></span>
                      <span>Host time<span className="mt-0.5 block text-xs text-muted">Recorded host intervals and waits</span></span>
                    </Menu.RadioItem>
                  </Menu.RadioGroup>
                </Menu.Popup>
              </Menu.Positioner>
            </Menu.Portal>
          </Menu.Root>
          <label className="flex items-center gap-2">
            <span className="cp-label">Speed</span>
            <select
              className="h-7 w-auto rounded border-line-strong bg-surface py-0 pl-2 pr-6 font-mono text-[11px] tabular-nums text-ink"
              value={speed}
              disabled={playbackDisabled}
              onChange={(event) => onSpeed(Number(event.target.value))}
            >
              {playbackSpeeds(mode).map((option) => <option key={option} value={option}>{option}×</option>)}
            </select>
          </label>
        </div>
      </div>
    </div>
  )
}

function Scope({ label, readout, ticks, markers, gapPosition, currentPosition, value, maxSequence, disabled, valueText, onScrub }: {
  label: string
  readout: string
  ticks: ScopeTick[]
  markers: ScopeMarker[]
  gapPosition: number | null
  currentPosition: number | null
  value: number
  maxSequence: number
  disabled: boolean
  valueText: string
  onScrub: (value: number) => void
}) {
  const container = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)
  useLayoutEffect(() => {
    const element = container.current
    if (!element) return
    const update = () => setWidth(Math.max(120, Math.floor(element.clientWidth)))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const columns = Math.max(16, width - GUTTER - 4)
  const playhead = maxSequence > 0 ? value / maxSequence : 0
  const model = useMemo(() => buildScope({ ticks, markers, playhead, gap: gapPosition, columns, currentPosition }), [ticks, markers, playhead, gapPosition, columns, currentPosition])
  const x = (column: number) => GUTTER + column + 0.5
  const bottom = TOP + SCOPE_CHANNELS.length * (ROW + ROW_GAP) - ROW_GAP
  const head = x(model.playheadColumn)
  let lastLabel = -Infinity
  return (
    <div ref={container} className="cp-scope">
      <svg aria-hidden="true">
        <defs>
          <pattern id="scope-unplayed" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="#131f33" strokeWidth="1.5" />
          </pattern>
          <pattern id="scope-gap" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="#e5687d" strokeWidth="1.5" opacity="0.45" />
          </pattern>
        </defs>
        <text x={width - 2} y={8} textAnchor="end" className="fill-muted font-mono text-[9.5px]">{readout}</text>
        {SCOPE_CHANNELS.map((channel, index) => {
          const y = TOP + index * (ROW + ROW_GAP)
          const path = model.bins[channel.id].map((column) => `M${x(column)},${y + 1}v${ROW - 2}`).join("")
          return (
            <g key={channel.id}>
              <text x={0} y={y + ROW - 1} className="fill-faint font-mono text-[8.5px] tracking-[0.06em]">{channel.code}</text>
              <rect x={GUTTER} y={y} width={columns + 1} height={ROW} fill="#070c15" stroke="#111c2d" />
              {path ? <path d={path} stroke="#7d90ad" strokeOpacity={0.6} strokeWidth={1.2} /> : null}
            </g>
          )
        })}
        {model.current ? (() => {
          const index = SCOPE_CHANNELS.findIndex((channel) => channel.id === model.current!.channel)
          const y = TOP + index * (ROW + ROW_GAP)
          return <rect x={x(model.current.column) - 0.75} y={y} width={1.5} height={ROW} fill="var(--accent)" />
        })() : null}
        {head < GUTTER + columns ? <rect x={head} y={TOP} width={GUTTER + columns - head + 1} height={bottom - TOP} fill="url(#scope-unplayed)" /> : null}
        {model.gapColumn != null ? (
          <g>
            <rect x={x(model.gapColumn)} y={TOP - 2} width={Math.max(2, GUTTER + columns - x(model.gapColumn))} height={bottom - TOP + 4} fill="url(#scope-gap)" />
            <line x1={x(model.gapColumn)} x2={x(model.gapColumn)} y1={TOP - 4} y2={bottom + 2} stroke="var(--danger)" strokeWidth={1.5} />
          </g>
        ) : null}
        {model.markers.map((marker, index) => {
          const mx = x(marker.column)
          const showLabel = mx - lastLabel > 22
          if (showLabel) lastLabel = mx
          return (
            <g key={`${marker.label}-${index}`}>
              <line x1={mx} x2={mx} y1={TOP - 4} y2={TOP} stroke="#40516b" />
              {showLabel ? <text x={mx + 2} y={TOP - 4} className="fill-faint font-mono text-[8px]">{marker.label}</text> : null}
            </g>
          )
        })}
        <line x1={head} x2={head} y1={TOP - 6} y2={bottom + 2} stroke="var(--accent)" strokeWidth={1.5} />
        <path d={`M${head - 4.5},${TOP - 10} L${head + 4.5},${TOP - 10} L${head},${TOP - 5} Z`} fill="var(--accent)" />
      </svg>
      <input
        type="range"
        aria-label={label}
        min={0}
        max={Math.max(maxSequence, 0)}
        step={1}
        value={value}
        disabled={disabled}
        aria-valuemin={0}
        aria-valuemax={maxSequence}
        aria-valuenow={value}
        aria-valuetext={valueText}
        onChange={(event) => onScrub(Number(event.target.value))}
      />
    </div>
  )
}
