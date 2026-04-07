import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.ml.brain_model import train

if __name__ == "__main__":
    print("Brain model fine-tuning started...")
    print("Dataset: EfficientNet-B0 (ImageNet pretrained) + your brain tumor dataset")
    print("="*50)
    train()
    print("="*50)
    print("Done. Model saved to saved_models/brain_efficientnet_b0.pth")