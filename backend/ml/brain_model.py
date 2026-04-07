import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import timm
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    BRAIN_MODEL_PATH, BRAIN_DATA_DIR, BRAIN_CLASSES,
    BRAIN_LABEL_MAP, IMG_SIZE, BATCH_SIZE, BRAIN_EPOCHS
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model   = None
_classes = None


def build_model(num_classes: int, pretrained: bool = True):
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    model.to(DEVICE)
    return model


def train(data_dir: str = BRAIN_DATA_DIR, save_path: str = BRAIN_MODEL_PATH):
    train_dir = os.path.join(data_dir, "Training")
    test_dir  = os.path.join(data_dir, "Testing")

    train_ds = ImageFolder(train_dir, transform=TRAIN_TRANSFORMS)
    val_ds   = ImageFolder(test_dir,  transform=VAL_TRANSFORMS)

    print(f"Train: {len(train_ds)} images | Classes: {train_ds.classes}")
    print(f"Val:   {len(val_ds)} images")
    print(f"Device: {DEVICE}")

    counts         = np.bincount(train_ds.targets)
    sample_weights = [1.0 / counts[t] for t in train_ds.targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=2, pin_memory=True)

    model     = build_model(num_classes=len(train_ds.classes), pretrained=True)
    cls_w     = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cls_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BRAIN_EPOCHS)

    best_acc   = 0.0
    no_improve = 0
    patience   = 3

    for epoch in range(BRAIN_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss(weight=cls_w)(model(images), labels).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                correct += (model(images).argmax(1) == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{BRAIN_EPOCHS} | Val Acc: {val_acc:.4f}")
        scheduler.step()

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes":     train_ds.classes,
                "val_acc":     best_acc,
            }, save_path)
            print(f"  Saved -> {save_path} (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val acc: {best_acc:.4f}")


def load_model(save_path: str = BRAIN_MODEL_PATH):
    global _model, _classes
    checkpoint = torch.load(save_path, map_location=DEVICE)
    _classes   = checkpoint["classes"]
    _model     = build_model(num_classes=len(_classes), pretrained=False)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print(f"Brain model loaded. Classes: {_classes} | Val acc: {checkpoint.get('val_acc', 0):.4f}")
    return _model, _classes


def predict(image: Image.Image) -> dict:
    global _model, _classes
    if _model is None:
        load_model()

    tensor = VAL_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    raw_label  = _classes[pred_idx]
    return {
        "label":         BRAIN_LABEL_MAP.get(raw_label, raw_label),
        "raw_label":     raw_label,
        "confidence":    round(float(probs[pred_idx]), 4),
        "probabilities": {
            BRAIN_LABEL_MAP.get(c, c): round(float(probs[i]), 4)
            for i, c in enumerate(_classes)
        },
    }


def get_gradcam(image: Image.Image) -> np.ndarray:
    global _model, _classes
    if _model is None:
        load_model()

    img_rgb   = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    tensor    = VAL_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    target_layer = _model.blocks[-1]
    cam       = GradCAM(model=_model, target_layers=[target_layer])
    grayscale = cam(input_tensor=tensor, targets=None)[0]
    return show_cam_on_image(img_rgb, grayscale, use_rgb=True)