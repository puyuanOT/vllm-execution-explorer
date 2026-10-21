import type { KvLayerConfiguration } from "../../../replay/src/kv.ts"

/** Name axes only for the observed layout whose dimensions we can verify. */
export function namedKvDimensions(layer: KvLayerConfiguration): string | null {
  const { numKernelBlocks: blocks, numKvHeads: heads, kernelBlockSize: slots, headDim: features } = layer
  if (layer.layoutStatus !== "qualified" || layer.layout !== "kernel_block,kv_head,token_slot,concatenated_K_V"
    || blocks == null || heads == null || slots == null || features == null
    || layer.shape.length !== 4 || layer.shape[0] !== blocks || layer.shape[1] !== heads || layer.shape[2] !== slots || layer.shape[3] !== 2 * features) return null
  return `${blocks} kernel blocks × ${heads} KV heads × ${slots} token slots × (${features} key + ${features} value features)`
}
