import logging
from typing import Any


def resolve_lora_layer_indices(
    llm_cfg: dict[str, Any],
    heatmap_cfg: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
) -> list[int] | None:
    """Resolve LoRA target layers with explicit config taking priority."""
    explicit_layers = llm_cfg.get("lora_layer_indices")
    if explicit_layers is not None:
        resolved_layers = list(explicit_layers)
        source = "llm.lora_layer_indices"
    else:
        heatmap_layers = None if heatmap_cfg is None else heatmap_cfg.get("llm_layer_indices")
        if heatmap_layers is not None:
            resolved_layers = list(heatmap_layers)
            source = "heatmap.llm_layer_indices"
        else:
            resolved_layers = None
            source = f"last {llm_cfg.get('lora_num_layers', 4)} layers"

    if logger is not None and llm_cfg.get("use_lora", False):
        if resolved_layers is None:
            logger.info(
                "%sLoRA enabled: no explicit layer list found, fallback source=%s",
                log_prefix,
                source,
            )
        else:
            logger.info(
                "%sLoRA enabled: resolved layers=%s (source=%s)",
                log_prefix,
                resolved_layers,
                source,
            )

    return resolved_layers
