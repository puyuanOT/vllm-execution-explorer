import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pause,
  Play,
  Radio,
  RotateCcw,
} from "lucide-react"

import { cn } from "../lib/cn.ts"
import { SPEEDS } from "../playback.ts"
import type { PlaybackMode } from "./types.ts"

export function PlaybackTransport({
  disabled,
  playing,
  atEnd,
  speed,
  mode,
  recordedAvailable,
  sequence,
  maxSequence,
  valueText,
  followLive,
  followAvailable,
  connected,
  reduced,
  onReduced,
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
  speed: (typeof SPEEDS)[number]
  mode: PlaybackMode
  recordedAvailable: boolean
  sequence: number
  maxSequence: number
  valueText: string
  followLive: boolean
  followAvailable: boolean
  connected: boolean
  reduced: boolean
  onReduced: (value: boolean) => void
  onTogglePlay: () => void
  onReplay: () => void
  onPreviousEvent: () => void
  onNextEvent: () => void
  onPreviousStep: () => void
  onNextStep: () => void
  onScrub: (sequence: number) => void
  onSpeed: (speed: (typeof SPEEDS)[number]) => void
  onMode: (mode: PlaybackMode) => void
  onFollow: () => void
}) {
  const playbackDisabled = disabled || maxSequence <= 0
  const playLabel = playing ? "Pause" : atEnd ? "Replay" : "Play"
  const play = () => {
    if (!playing && atEnd) onReplay()
    else onTogglePlay()
  }
  const iconButton = "grid size-9 shrink-0 place-items-center rounded-md border border-line bg-raised p-0 text-muted hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50"
  return (
    <div className="transport border-t border-line bg-surface px-4 py-3" role="group" aria-label="Recording playback">
      <div className="transport-buttons flex flex-wrap items-center gap-1.5">
        <button type="button" className={iconButton} onClick={onReplay} disabled={playbackDisabled} aria-label="Restart recording">
          <RotateCcw className="size-4" aria-hidden="true" />
        </button>
        <span className="mx-1 h-6 w-px bg-line" aria-hidden="true" />
        <button type="button" className={iconButton} onClick={onPreviousStep} disabled={playbackDisabled} aria-label="Previous scheduler step">
          <ChevronsLeft className="size-4" aria-hidden="true" />
        </button>
        <button type="button" className={iconButton} onClick={onPreviousEvent} disabled={playbackDisabled} aria-label="Previous recorded event">
          <ChevronLeft className="size-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="flex min-h-9 items-center gap-2 rounded-md border border-accent bg-accent/15 px-3 py-1.5 text-sm font-medium text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={playLabel}
          aria-pressed={playing}
          disabled={playbackDisabled}
          onClick={play}
        >
          {playing ? <Pause className="size-4" aria-hidden="true" /> : <Play className="size-4" aria-hidden="true" />}
          {playLabel}
        </button>
        <button type="button" className={iconButton} onClick={onNextEvent} disabled={playbackDisabled} aria-label="Next recorded event">
          <ChevronRight className="size-4" aria-hidden="true" />
        </button>
        <button type="button" className={iconButton} onClick={onNextStep} disabled={playbackDisabled} aria-label="Next scheduler step">
          <ChevronsRight className="size-4" aria-hidden="true" />
        </button>
        {followAvailable ? (
          <button
            type="button"
            className={cn(
              "ml-1 flex min-h-9 items-center gap-2 rounded-md border border-line bg-raised px-3 py-1.5 text-sm text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
              followLive && "border-accent text-ink",
            )}
            aria-pressed={followLive}
            onClick={onFollow}
          >
            <Radio className="size-4" aria-hidden="true" />
            {connected ? "Follow live" : "Reconnect live"}
          </button>
        ) : null}
      </div>

      <label className="timeline min-w-56 flex-1 text-xs text-muted">
        <span className="flex items-center justify-between gap-3">
          <span>Playhead</span>
          <span className="tabular-nums">Event {Math.min(sequence, Math.max(maxSequence, 0))} / {Math.max(maxSequence, 0)}</span>
        </span>
        <input
          type="range"
          className="mt-1 accent-accent"
          min={0}
          max={Math.max(maxSequence, 0)}
          step={1}
          value={Math.min(sequence, Math.max(maxSequence, 0))}
          disabled={playbackDisabled}
          aria-valuemin={0}
          aria-valuemax={maxSequence}
          aria-valuenow={Math.min(sequence, Math.max(maxSequence, 0))}
          aria-valuetext={valueText}
          onChange={(event) => onScrub(Number(event.target.value))}
        />
      </label>

      <label className="min-w-36 text-xs text-muted">
        Playback mode
        <select
          className="mt-1 rounded-md border border-line bg-raised px-2 py-1.5 text-sm text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          value={mode}
          disabled={playbackDisabled}
          onChange={(event) => onMode(event.target.value as PlaybackMode)}
        >
          <option value="events">Recorded events</option>
          <option value="recorded" disabled={!recordedAvailable}>
            {recordedAvailable ? "Recorded time" : "Recorded time unavailable"}
          </option>
        </select>
      </label>
      <label className="min-w-24 text-xs text-muted">
        Speed
        <select
          className="mt-1 rounded-md border border-line bg-raised px-2 py-1.5 text-sm tabular-nums text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          value={speed}
          disabled={playbackDisabled}
          onChange={(event) => onSpeed(Number(event.target.value) as (typeof SPEEDS)[number])}
        >
          {SPEEDS.map((value) => (
            <option key={value} value={value}>{value}×</option>
          ))}
        </select>
      </label>
      <label className="check flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={reduced} onChange={(event) => onReduced(event.target.checked)} />
        Reduce motion
      </label>

      <div className="basis-full">
        <p className="meta m-0 truncate text-pretty text-xs tabular-nums text-ink">{valueText}</p>
        <p className="meta m-0 mt-1 text-pretty text-xs text-muted">
          {mode === "events"
            ? "Event mode advances through recorded observations at a teaching pace."
            : "Recorded-time mode preserves observed host intervals at the selected speed; it does not represent GPU kernel timing."}
          {followAvailable ? (connected ? " Live capture continues while replay is paused." : " The live trace is reconnecting.") : ""}
        </p>
      </div>
    </div>
  )
}
