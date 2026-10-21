export interface RecordingLoadCallbacks<T> {
  prepare(label: string, kind: "demo" | "recording"): void
  reading(label: string): void
  accept(value: T, kind: "demo" | "recording"): void
  reject(error: unknown, kind: "demo" | "recording"): void
  finish(kind: "demo" | "recording"): void
}

const afterPaint = () => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))

/** Reads begin only after the old view and its alternate render have committed
 * empty states. Never capture old render callbacks across the asynchronous read. */
export function createRecordingLoader<T>(callbacks: { current: RecordingLoadCallbacks<T> | null }, settle = afterPaint) {
  let busy = false
  let generation = 0
  return {
    get busy() { return busy },
    async open(read: () => Promise<T>, label: string, kind: "demo" | "recording" = "recording"): Promise<boolean> {
      if (busy) return false
      busy = true
      const current = ++generation
      try {
        callbacks.current?.prepare(label, kind)
        await settle()
        if (generation !== current) return false
        callbacks.current?.reading(label)
        await settle()
        if (generation !== current) return false
        const loaded = await read()
        if (generation !== current) return false
        callbacks.current?.accept(loaded, kind)
        return true
      } catch (error) {
        if (generation === current) callbacks.current?.reject(error, kind)
        return false
      } finally {
        if (generation === current) {
          busy = false
          callbacks.current?.finish(kind)
        }
      }
    },
    cancel() { generation += 1; busy = false },
  }
}
