import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DATA_DIR         = BASE_DIR / "data"

LUNG_HF_MODEL_ID   = "lxyuan/vit-xray-pneumonia-classification"
BRAIN_MODEL_PATH   = str(SAVED_MODELS_DIR / "brain_efficientnet_b0.pth")
SYMPTOM_XGB_PATH   = str(SAVED_MODELS_DIR / "symptom_xgb.pkl")
LABEL_ENCODER_PATH = str(SAVED_MODELS_DIR / "label_encoder.pkl")
SYMPTOM_LIST_PATH  = str(SAVED_MODELS_DIR / "symptom_list.pkl")

BRAIN_DATA_DIR   = str(DATA_DIR / "brain_tumor")
SYMPTOM_CSV_PATH = str(DATA_DIR / "symptoms" / "dataset.csv")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

IMG_SIZE     = 224
BATCH_SIZE   = 16
BRAIN_EPOCHS = 5

LUNG_MODEL_PATH = str(SAVED_MODELS_DIR / "lung_efficientnet_b0.pth")
CLIP_MODEL_ID      = "openai/clip-vit-base-patch32"
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

BRAIN_LABEL_MAP = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "notumor":    "No Tumor",
    "pituitary":  "Pituitary Tumor",
}