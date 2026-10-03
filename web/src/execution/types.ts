import type { WorkflowComponentId } from "../explore/model.ts"

export type Selection =
  | { kind: "request"; id: string }
  | { kind: "block"; key: string }
  | { kind: "step"; id: string }
  | { kind: "component"; id: WorkflowComponentId }
  | { kind: "layer"; layerIndex: number }
  | { kind: "module"; layerIndex: number; module: "attention" | "ffn" }
  | { kind: "head"; layerIndex: number; headIndex: number; headKind: "query" | "kv" }
  | { kind: "token"; requestId: string; position: number }

export type PlaybackMode = "events" | "recorded" | "teaching" | "measured"
export type WorkspaceView = "timeline" | "spatial"
export type NarrowPane = "compose" | "observe"
