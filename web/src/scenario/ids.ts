export function createId(prefix: string): string {
  const bytes = new Uint8Array(8)
  crypto.getRandomValues(bytes)
  const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("")
  return `${prefix}-${hex}`
}
