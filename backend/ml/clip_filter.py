import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import CLIP_MODEL_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model     = None
_processor = None

LUNG_PROMPTS = [
    "a chest x-ray image",
    "a lung radiograph",
    "a chest radiograph",
]

LUNG_NEGATIVE = [
    "a brain mri scan",
    "a photo of a person",
    "a random image",
    "a game screenshot",
    "a cartoon image",
    "a document or text",
]

BRAIN_PROMPTS = [
    "a brain mri scan",
    "a brain magnetic resonance image",
    "a brain mri",
]

BRAIN_NEGATIVE = [
    "a chest x-ray image",
    "a photo of a person",
    "a random image",
    "a game screenshot",
    "a cartoon image",
    "a document or text",
]


def get_clip():
    global _model, _processor
    if _model is None:
        print(f"Loading CLIP model: {CLIP_MODEL_ID}")
        _model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _model.eval()
        print("CLIP model ready.")
    return _model, _processor


def is_valid(image: Image.Image, modality: str = "lung", threshold: float = 0.45) -> tuple:
    model, processor = get_clip()

    if modality == "lung":
        positive = LUNG_PROMPTS
        negative = LUNG_NEGATIVE
    else:
        positive = BRAIN_PROMPTS
        negative = BRAIN_NEGATIVE

    all_prompts = positive + negative
    inputs      = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
    inputs      = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs   = outputs.logits_per_image.squeeze().softmax(dim=0).cpu().numpy()

    positive_score = float(probs[:len(positive)].sum())
    valid          = positive_score >= threshold

    return valid, round(positive_score, 3)