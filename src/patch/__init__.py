"""Attention patch entrypoints."""

from src.patch.patch_glm import patch_glm4_attention_forward_model
from src.patch.patch_llama import patch_llama3_attention_forward_model
from src.patch.patch_qwen import patch_qwen3_attention_forward_model


def _infer_patch_target(model) -> str:
    """Infer patch target from model config and class metadata."""
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    class_name = model.__class__.__name__.lower()
    combined = f"{model_type} {class_name}"

    if "glm4" in combined or "chatglm" in combined or "glm" in combined:
        return "glm4"
    if "qwen3" in combined or "qwen" in combined:
        return "qwen"
    if "llama" in combined:
        return "llama"

    raise ValueError(
        f"Unable to infer patch target from model_type='{model_type}', class='{model.__class__.__name__}'. "
        "Please set patch_target explicitly to one of: qwen, llama, glm4."
    )


def patch_attention_forward_model(
    model,
    *,
    mode: str = "sage_w",
    window_size: int = 256,
    top_ratio: float = 0.05,
    report_stats: bool = False,
    use_vertical_indices: bool = True,
    patch_target: str = "auto",
):
    """Patch attention forward for qwen/llama/glm4 with optional auto detection."""
    if patch_target == "auto":
        patch_target = _infer_patch_target(model)

    if patch_target == "qwen":
        return patch_qwen3_attention_forward_model(
            model,
            mode=mode,
            window_size=window_size,
            top_ratio=top_ratio,
            report_stats=report_stats,
            use_vertical_indices=use_vertical_indices,
        )
    if patch_target == "llama":
        return patch_llama3_attention_forward_model(
            model,
            mode=mode,
            window_size=window_size,
            top_ratio=top_ratio,
            report_stats=report_stats,
            use_vertical_indices=use_vertical_indices,
        )
    if patch_target == "glm4":
        return patch_glm4_attention_forward_model(
            model,
            mode=mode,
            window_size=window_size,
            top_ratio=top_ratio,
            report_stats=report_stats,
            use_vertical_indices=use_vertical_indices,
        )

    raise ValueError("Unsupported patch_target. Expected one of: auto, qwen, llama, glm4.")

__all__ = [
    "patch_attention_forward_model",
    "patch_glm4_attention_forward_model",
    "patch_llama3_attention_forward_model",
    "patch_qwen3_attention_forward_model",
]

