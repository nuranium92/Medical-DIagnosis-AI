import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from backend.config import LUNG_MODEL_PATH

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "data/chest_xray"
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 5e-5

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.CenterCrop((220, 220)),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.CenterCrop((220, 220)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def is_valid_image(path: str) -> bool:
    name = os.path.basename(path)
    return (
        not name.startswith("._") and
        not name.startswith(".") and
        path.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def get_sampler(dataset):
    counts         = np.bincount(dataset.targets)
    sample_weights = [1.0 / counts[t] for t in dataset.targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train():
    print(f"Device: {DEVICE}")
    print(f"Loading EfficientNet-B0 pretrained weights (ImageNet)...")
    print(f"Data dir: {DATA_DIR}")
    print("=" * 50)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2,drop_rate = 0.3)
    model.to(DEVICE)

    train_ds = ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=TRAIN_TRANSFORMS,
        is_valid_file=is_valid_image,
    )
    val_ds = ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=VAL_TRANSFORMS,
        is_valid_file=is_valid_image,
    )

    print(f"Train: {len(train_ds)} images | Classes: {train_ds.classes}")
    print(f"Val:   {len(val_ds)} images")

    class_counts = np.bincount(train_ds.targets)
    for cls, cnt in zip(train_ds.classes, class_counts):
        print(f"  {cls}: {cnt} images")
    print()

    sampler      = get_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    counts    = np.bincount(train_ds.targets)
    cls_w     = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cls_w)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc   = 0.0
    no_improve = 0
    patience   = 4

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total   = 0
        train_loss    = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds          = model(images).argmax(dim=1)
                val_correct   += (preds == labels).sum().item()
                val_total     += labels.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Loss: {train_loss/len(train_loader):.4f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            os.makedirs(os.path.dirname(LUNG_MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes":     train_ds.classes,
                "val_acc":     best_acc,
            }, LUNG_MODEL_PATH)
            print(f"  Saved -> {LUNG_MODEL_PATH} (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("Early stopping.")
                break

    print("=" * 50)
    print(f"Training complete. Best val acc: {best_acc:.4f}")
    print(f"Model saved: {LUNG_MODEL_PATH}")


if __name__ == "__main__":
    print("Lung EfficientNet-B0 fine-tuning started...")
    print("=" * 50)
    train()