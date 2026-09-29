import type { WorkflowComponentId } from "../explore/model.ts"

export type Selection =
  | { kind: "request"; id: string }
  | { kind: "block"; key: string }
  | { kind: "step"; id: string }
  | { kind: "component"; id: WorkflowComponentId }

export type PlaybackMode = "events" | "recorded"
export type WorkspaceView = "timeline" | "spatial"
export type NarrowPane = "compose" | "observe"
