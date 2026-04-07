import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import LUNG_MODEL_PATH, IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LUNG_CLASSES = ["NORMAL", "PNEUMONIA"]

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.CenterCrop((220, 220)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model   = None
_classes = None


def load_model():
    global _model, _classes
    checkpoint = torch.load(LUNG_MODEL_PATH, map_location=DEVICE)
    _classes   = checkpoint.get("classes", LUNG_CLASSES)
    _model     = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(_classes),drop_rate =0.3)
    _model.load_state_dict(checkpoint["model_state"])
    _model.to(DEVICE)
    _model.eval()
    print(f"Lung model loaded. Classes: {_classes} | Val acc: {checkpoint.get('val_acc', 0):.4f}")
    return _model, _classes


def get_lung_model():
    global _model, _classes
    if _model is None:
        load_model()
    return _model, _classes


def predict(image: Image.Image) -> dict:
    model, classes = get_lung_model()
    tensor = VAL_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "label":         classes[pred_idx],
        "confidence":    round(float(probs[pred_idx]), 4),
        "probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
    }


def get_gradcam(image: Image.Image) -> np.ndarray:
    model, classes = get_lung_model()

    img_rgb   = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    tensor    = VAL_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    target_layer = model.blocks[-2]
    cam          = GradCAM(model=model, target_layers=[target_layer])

    with torch.no_grad():
        probs    = torch.softmax(model(tensor), dim=1).squeeze()
        pred_idx = int(probs.argmax().item())

    targets   = [ClassifierOutputTarget(pred_idx)]
    grayscale = cam(input_tensor=tensor, targets=targets)[0]
    return show_cam_on_image(img_rgb, grayscale, use_rgb=True)