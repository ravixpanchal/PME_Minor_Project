"""Shared lazy model loading for extraction and region detection."""
import os


def load_vlm(model_id):
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("VLM dependencies missing. Run: uv sync --extra vlm") from exc
    device = os.getenv("VLM_DEVICE", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build or set VLM_DEVICE=cpu.")
    if device.startswith("cuda"):
        with torch.cuda.device(device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    return torch, processor, model
