import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import LUNG_HF_MODEL_ID, IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_pipeline  = None
_model     = None
_processor = None


def get_lung_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"Loading lung model: {LUNG_HF_MODEL_ID}")
        _pipeline = pipeline(
            "image-classification",
            model=LUNG_HF_MODEL_ID,
            device=0 if str(DEVICE) == "cuda" else -1,
        )
        print("Lung pipeline ready.")
    return _pipeline


def get_lung_model_and_processor():
    global _model, _processor
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(LUNG_HF_MODEL_ID)
        _model     = AutoModelForImageClassification.from_pretrained(LUNG_HF_MODEL_ID)
        _model.to(DEVICE)
        _model.eval()
    return _model, _processor


def predict(image: Image.Image) -> dict:
    clf     = get_lung_pipeline()
    results = clf(image.convert("RGB"))
    top     = results[0]
    return {
        "label":         top["label"],
        "confidence":    round(top["score"], 4),
        "probabilities": {r["label"]: round(r["score"], 4) for r in results},
    }


def attention_rollout(attentions: list, discard_ratio: float = 0.9) -> np.ndarray:
    result = torch.eye(attentions[0].size(-1))

    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention.mean(dim=1)
            flat                  = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            threshold             = flat.quantile(discard_ratio, dim=-1, keepdim=True)
            threshold             = threshold.unsqueeze(-1)
            attention_heads_fused = torch.where(
                attention_heads_fused < threshold,
                torch.zeros_like(attention_heads_fused),
                attention_heads_fused,
            )

            identity = torch.eye(attention_heads_fused.size(-1))
            a        = (attention_heads_fused + identity) / 2
            a        = a / a.sum(dim=-1, keepdim=True)
            result   = torch.matmul(a, result)

    mask = result[0, 0, 1:]
    n    = int(mask.size(0) ** 0.5)
    mask = mask[:n * n].reshape(n, n)
    mask = mask / mask.max()
    return mask.numpy()


def get_attention_map(image: Image.Image) -> np.ndarray:
    model, processor = get_lung_model_and_processor()

    img_rgb   = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_rgb).astype(np.float32) / 255.0
    inputs    = processor(images=img_rgb, return_tensors="pt").to(DEVICE)

    all_attentions = []

    hooks = []
    for layer in model.vit.encoder.layer:
        def make_hook():
            def hook_fn(module, input, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                all_attentions.append(attn_out.detach().cpu())
            return hook_fn
        hooks.append(
            layer.attention.attention.register_forward_hook(make_hook())
        )

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    if not all_attentions:
        return img_array

    try:
        rollout = attention_rollout(all_attentions, discard_ratio=0.9)
        heatmap = cv2.resize(rollout, (IMG_SIZE, IMG_SIZE))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        result  = show_cam_on_image(img_array, heatmap, use_rgb=True)
        return result
    except Exception:
        attn     = all_attentions[-1]
        if attn.ndim == 4:
            attn = attn.mean(dim=1)
        cls_attn = attn[0, 0, 1:]
        n        = int(cls_attn.shape[0] ** 0.5)
        cls_attn = cls_attn[:n * n].reshape(n, n).numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        heatmap  = cv2.resize(cls_attn, (IMG_SIZE, IMG_SIZE))
        return show_cam_on_image(img_array, np.clip(heatmap, 0, 1), use_rgb=True)