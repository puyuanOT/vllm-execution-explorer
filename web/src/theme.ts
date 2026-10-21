export const REQUEST_COLORS = [
  "#6aa6e8",
  "#e2709c",
  "#5ccc9e",
  "#e3a35f",
  "#d4c96a",
  "#b4a2e3",
  "#ddb093",
  "#63c6ba",
]

export function requestColor(order: string[], id: string): string {
  const index = order.indexOf(id)
  if (index < 0) return "#7a8799"
  return REQUEST_COLORS[index % REQUEST_COLORS.length]
}
