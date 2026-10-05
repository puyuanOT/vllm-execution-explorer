/** Keep the current column visible beside sticky labels, including on narrow screens. */
export function sequencerScrollLeft(currentIndex: number, columnWidth: number, labelWidth: number, viewportWidth: number, scrollLeft: number): number {
  const left = labelWidth + currentIndex * columnWidth
  const right = left + columnWidth
  if (left >= scrollLeft + labelWidth && right <= scrollLeft + viewportWidth) return scrollLeft
  const trailing = Math.min(columnWidth * 3, Math.max(0, viewportWidth - labelWidth - columnWidth))
  return Math.max(0, right - viewportWidth + trailing)
}
