import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.ml.symptom_checker import train

if __name__ == "__main__":
    print("Symptom XGBoost training started...")
    print("="*50)
    train()
    print("="*50)
    print("Done. Models saved to saved_models/")