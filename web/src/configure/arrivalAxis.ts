/** Whole-millisecond ticks match the precision accepted by the workload editor. */
export function arrivalAxis(latest: number): { span: number; ticks: number[] } {
  const target = Math.max(10, latest * 1.25)
  const magnitude = 10 ** Math.floor(Math.log10(target))
  const span = [1, 1.5, 2, 2.5, 3, 5, 7.5, 10].map((factor) => factor * magnitude).find((value) => value >= target) ?? 10 * magnitude
  const raw = span / 4
  const stepMagnitude = 10 ** Math.floor(Math.log10(raw))
  const step = Math.ceil([1, 2, 2.5, 5, 10].map((factor) => factor * stepMagnitude).find((value) => value >= raw) ?? stepMagnitude * 10)
  return { span, ticks: Array.from({ length: Math.floor(span / step) + 1 }, (_, index) => index * step) }
}
