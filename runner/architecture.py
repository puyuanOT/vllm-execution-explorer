"""Loaded model geometry, with explicit support boundaries for model drills."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path
from types import SimpleNamespace


def describe_architecture(model_config: Any, *, cache_groups: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    config = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    model_type = str(getattr(config, "model_type", "unknown"))
    architectures = list(getattr(config, "architectures", None) or [])
    result: dict[str, Any] = {
        "source": "loaded_model_config", "model_type": model_type, "architectures": architectures,
        "supported": False, "kv_groups": cache_groups or [],
    }
    hybrid = model_type == "qwen3_5_text"
    if model_type not in {"qwen3", "qwen3_5_text"} or getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None):
        result["reason"] = "Model drills support dense Qwen3 and dense Qwen hybrid decoder models; this architecture has not been qualified."
        return result
    names = {
        "num_layers": "num_hidden_layers", "hidden_size": "hidden_size", "intermediate_size": "intermediate_size",
        "num_attention_heads": "num_attention_heads", "num_key_value_heads": "num_key_value_heads", "vocab_size": "vocab_size",
    }
    for target, source in names.items():
        value = getattr(config, source, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            result["reason"] = f"Loaded model configuration does not expose valid {source}; model drill geometry is unavailable."
            return result
        result[target] = value
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and result["hidden_size"] % result["num_attention_heads"] == 0:
        head_dim = result["hidden_size"] // result["num_attention_heads"]
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0 or result["num_attention_heads"] % result["num_key_value_heads"]:
        result["reason"] = "Loaded attention dimensions do not expose supported grouped-query geometry."
        return result
    result.update(
        supported=True, head_dim=head_dim, activation=getattr(config, "hidden_act", None),
        norm="RMSNorm", norm_epsilon=getattr(config, "rms_norm_eps", None),
        position_encoding="RoPE", rope_theta=getattr(config, "rope_theta", None),
        layer_names=[f"model.layers.{index}" for index in range(result["num_layers"])],
        evidence="Dimensions read from the initialized model configuration; operator layout is the supported Qwen3 architecture, not an observed GPU timeline.",
    )
    if hybrid:
        layer_types = getattr(config, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) != result["num_layers"] or any(kind not in {"linear_attention", "full_attention"} for kind in layer_types):
            result.update(supported=False, reason="Hybrid layer types must be read explicitly from the loaded configuration.")
            return result
        recurrent_names = {
            "key_heads": "linear_num_key_heads", "value_heads": "linear_num_value_heads",
            "key_head_dim": "linear_key_head_dim", "value_head_dim": "linear_value_head_dim",
            "conv_kernel_size": "linear_conv_kernel_dim",
        }
        recurrent = {key: getattr(config, name, None) for key, name in recurrent_names.items()}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in recurrent.values()):
            result.update(supported=False, reason="Loaded Gated DeltaNet dimensions are incomplete.")
            return result
        result.update(layer_types=list(layer_types), recurrent=recurrent,
                      attention_output_gate=bool(getattr(config, "attn_output_gate", True)),
                      attention_layer_indices=[index for index, kind in enumerate(layer_types) if kind == "full_attention"],
                      recurrent_layer_indices=[index for index, kind in enumerate(layer_types) if kind == "linear_attention"],
                      evidence="Hybrid topology and dimensions read from the initialized model configuration. Gated DeltaNet recurrent state is separate from attention KV; this layout does not imply observed execution timing.")
    return result


def fixture_architecture() -> dict[str, Any]:
    reference = json.loads(Path(__file__).with_name("qwen3_reference.json").read_text(encoding="utf-8"))
    result = describe_architecture(SimpleNamespace(hf_config=SimpleNamespace(**reference["config"])))
    result.update(source="pinned_model_config_reference", reference_repository=reference["repository"],
                  reference_revision=reference["revision"], reference_url=reference["source_url"],
                  evidence="Teaching fixture uses dimensions from this pinned public Qwen3 config. It does not load weights or observe model-layer execution.")
    return result
