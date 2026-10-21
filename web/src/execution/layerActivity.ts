import type { MeasuredSceneProfile, ModelArchitecture } from "../architecture/model.ts"

const MODULE_LABELS = { attention: "Self-attention", ffn: "Feed-forward", recurrent: "Gated DeltaNet" } as const

/** Only mapped native activity determines execution; host participants and inspected layers never fill gaps. */
export function measuredLayerActivity(profile: MeasuredSceneProfile, architecture: ModelArchitecture) {
  const count = architecture.layerCount ?? 0
  return [...new Set(profile.activeLayers.filter((layer) => Number.isInteger(layer) && layer >= 0 && layer < count))]
    .sort((a, b) => a - b)
    .map((layerIndex) => {
      const modules = [...new Set((profile.activeModules ?? []).filter((entry) => entry.layerIndex === layerIndex).map((entry) => entry.module))]
      const detail = modules.length ? modules.map((module) => MODULE_LABELS[module]).join(" + ")
        : architecture.layerTypes[layerIndex] === "linear_attention" ? "Gated DeltaNet layer" : "Transformer layer"
      return { layerIndex, detail }
    })
}
