import { useFrame, useThree } from "@react-three/fiber"
import { useLayoutEffect, useRef } from "react"
import { REVISION, Vector2, type Camera, type Object3D } from "three"

import type { SceneProps } from "../scene.tsx"

const IDLE_MS = 250
const MAX_FRAMES = 2048

interface FrameSample {
  renderCpuMs: number
  callbackToRenderEndMs: number | null
  intervalMs: number | null
  calls: number
  triangles: number
  geometries: number
  textures: number
}

function distribution(values: number[]) {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const round = (value: number) => Math.round(value * 1000) / 1000
  return {
    count: values.length, min: round(sorted[0]), max: round(sorted.at(-1)!),
    mean: round(values.reduce((sum, value) => sum + value, 0) / values.length),
    p50: round(sorted[Math.ceil(sorted.length * 0.5) - 1]),
    p95: round(sorted[Math.ceil(sorted.length * 0.95) - 1]),
  }
}

/** Opt-in observations of existing demand frames; never schedules or invalidates one. */
export function SceneProfileDiagnostics({ sceneProps, focusKey }: { sceneProps: SceneProps; focusKey: string }) {
  const gl = useThree((state) => state.gl)
  const get = useThree((state) => state.get)
  const frameStart = useRef<number | null>(null)
  const context = useRef({ runId: "", focusKey: "", mode: "", sequence: 0, poolCount: 0, poolCapacity: 0, poolCapacities: [] as number[] })
  const flush = useRef<(reason: string) => void>(() => {})
  const mode = sceneProps.measuredProfile ? "measured" : sceneProps.teachingProgress != null ? "teaching" : "events"
  const runId = sceneProps.manifest?.run_id ?? "unknown"

  useLayoutEffect(() => {
    if (context.current.runId !== runId || context.current.focusKey !== focusKey || context.current.mode !== mode) {
      flush.current("context-change")
    }
    const pools = sceneProps.kvCache?.pools ?? []
    context.current = {
      runId, focusKey, mode, sequence: sceneProps.state.lastSequence,
      poolCount: pools.length, poolCapacity: pools.reduce((sum, pool) => sum + pool.capacity, 0),
      poolCapacities: pools.slice(0, 16).map((pool) => pool.capacity),
    }
  }, [runId, focusKey, mode, sceneProps.state.lastSequence, sceneProps.kvCache])

  // Negative priority observes the normal R3F loop without taking over rendering.
  useFrame(() => { frameStart.current = performance.now() }, -1000)

  useLayoutEffect(() => {
    const sessionId = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
    const original = gl.render
    const bufferSize = new Vector2()
    let samples: FrameSample[] = []
    let lastFrameStart: number | null = null
    let firstSequence = 0
    let lastSequence = 0
    let burst = 0
    let frameCount = 0
    let idleTimer: ReturnType<typeof setTimeout> | null = null
    const resources = () => ({
      geometries: gl.info.memory.geometries, textures: gl.info.memory.textures,
      programs: gl.info.programs?.length ?? null,
    })
    const log = (payload: object) => console.info(`[scene-profile] ${JSON.stringify({ session_id: sessionId, ...payload })}`)
    const flushBurst = (reason: string) => {
      if (idleTimer != null) clearTimeout(idleTimer)
      idleTimer = null
      if (!samples.length) return
      // The first interval may include an idle period or scene transition. Keep it
      // separate without filtering slow frames out of the remaining distribution.
      const intervals = samples.slice(1).flatMap((sample) => sample.intervalMs == null ? [] : [sample.intervalMs])
      gl.getDrawingBufferSize(bufferSize)
      log({
        event: "frames", burst: ++burst, reason,
        source_recording_run_id: context.current.runId.slice(0, 256),
        focus: context.current.focusKey.slice(0, 256), clock_mode: context.current.mode,
        sequence_first: firstSequence, sequence_last: lastSequence,
        frame_first: frameCount - samples.length + 1, frame_last: frameCount,
        viewport_css: { width: get().size.width, height: get().size.height },
        drawing_buffer: { width: bufferSize.x, height: bufferSize.y }, dpr: gl.getPixelRatio(),
        pool_count: context.current.poolCount, physical_capacity_total: context.current.poolCapacity,
        first_16_pool_capacities: context.current.poolCapacities,
        first_frame: samples[0],
        render_cpu_ms: distribution(samples.map((sample) => sample.renderCpuMs)),
        callback_to_render_end_ms: distribution(samples.flatMap((sample) => sample.callbackToRenderEndMs == null ? [] : [sample.callbackToRenderEndMs])),
        frame_interval_ms: distribution(intervals), preceding_frame_interval_ms: samples[0].intervalMs,
        draw_calls: distribution(samples.map((sample) => sample.calls)), triangles: distribution(samples.map((sample) => sample.triangles)),
        rendered_resources_first: { geometries: samples[0].geometries, textures: samples[0].textures },
        rendered_resources_last: { geometries: samples.at(-1)!.geometries, textures: samples.at(-1)!.textures },
        resources_at_flush: resources(),
      })
      samples = []
    }
    flush.current = flushBurst
    const render = (scene: Object3D, camera: Camera) => {
      const start = performance.now()
      const callbackStart = frameStart.current
      frameStart.current = null
      original.call(gl, scene, camera)
      const end = performance.now()
      const interval = callbackStart != null && lastFrameStart != null ? callbackStart - lastFrameStart : null
      if (callbackStart != null) lastFrameStart = callbackStart
      if (!samples.length) firstSequence = context.current.sequence
      lastSequence = context.current.sequence
      samples.push({
        renderCpuMs: end - start, callbackToRenderEndMs: callbackStart == null ? null : end - callbackStart,
        intervalMs: interval, calls: gl.info.render.calls, triangles: gl.info.render.triangles,
        geometries: gl.info.memory.geometries, textures: gl.info.memory.textures,
      })
      frameCount++
      if (samples.length >= MAX_FRAMES) flushBurst("sample-limit")
      else {
        if (idleTimer != null) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => flushBurst("idle"), IDLE_MS)
      }
    }
    gl.render = render
    log({
      event: "start", three_revision: REVISION, source_recording_run_id: context.current.runId.slice(0, 256),
      timing_scope: "gl.render synchronous CPU submission and R3F callback-to-render-end; excludes React commits, GPU completion and browser presentation",
      interval_scope: "interval between rendered R3F frame callbacks within each burst; the preceding interval may include demand-loop idle time or a transition and is reported separately",
      resource_scope: "Three.js gl.info counts, not GPU bytes or complete scene resource ownership",
      idle_flush_ms: IDLE_MS, max_frames_per_summary: MAX_FRAMES,
      info_auto_reset: gl.info.autoReset, user_agent: navigator.userAgent.slice(0, 256),
    })
    return () => {
      flushBurst("unmount")
      log({ event: "stop", total_frames: frameCount, resources_at_cleanup: resources() })
      if (gl.render === render) gl.render = original
      flush.current = () => {}
    }
  }, [gl, get])
  return null
}
