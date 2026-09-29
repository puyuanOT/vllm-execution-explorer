export const REQUEST_COLORS = [
  "#79c0ff",
  "#f0a35e",
  "#7dcea0",
  "#e08aa4",
  "#c9b6ff",
  "#e3d36b",
  "#6ed0d0",
  "#f0b089",
]

export function requestColor(order: string[], id: string): string {
  const index = order.indexOf(id)
  if (index < 0) return "#8a8175"
  return REQUEST_COLORS[index % REQUEST_COLORS.length]
}
