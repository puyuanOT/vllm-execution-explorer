import type { TimeUnit } from "./types.ts"

export function formatTime(ms: number, unit: TimeUnit): string {
  if (unit === "ms") return String(ms)
  const whole = Math.floor(ms / 1000)
  const fraction = ms % 1000
  if (fraction === 0) return String(whole)
  const digits = String(fraction).padStart(3, "0").replace(/0+$/, "")
  return `${whole}.${digits}`
}

export function parseTime(text: string, unit: TimeUnit): number | null {
  return unit === "ms" ? parseMilliseconds(text) : parseSecondsToMilliseconds(text)
}

export function parseMilliseconds(text: string): number | null {
  const trimmed = text.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const value = Number(trimmed)
  if (!Number.isSafeInteger(value)) return null
  return value
}

/** Scale the arrival axis so the latest batch is not pinned to the right edge. */
export function axisSpanMs(offsets: number[], capMs: number): number {
  const latest = offsets.reduce((max, value) => Math.max(max, value), 0)
  const cap = Number.isFinite(capMs) && capMs > 0 ? capMs : 120_000
  return Math.min(cap, Math.max(200, Math.ceil(latest * 1.25)))
}

/** Map a pointer on the arrival axis to an integer millisecond offset. */
export function millisecondsFromAxis(
  pointerX: number,
  left: number,
  width: number,
  spanMs: number,
  capMs: number,
): number {
  if (!(width > 0) || !Number.isFinite(spanMs) || spanMs < 0) return 0
  const cap = Number.isFinite(capMs) && capMs > 0 ? capMs : spanMs
  const ratio = Math.min(1, Math.max(0, (pointerX - left) / width))
  const ms = Math.round(ratio * spanMs)
  return Math.min(cap, Math.max(0, ms))
}

/** Convert a seconds string to integer milliseconds. Sub-millisecond values are rejected. */
export function parseSecondsToMilliseconds(text: string): number | null {
  const trimmed = text.trim()
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return null
  const [whole, fraction = ""] = trimmed.split(".")
  if (fraction.length > 3) return null
  const milliseconds = Number(whole) * 1000 + Number(fraction.padEnd(3, "0"))
  if (!Number.isSafeInteger(milliseconds)) return null
  return milliseconds
}
