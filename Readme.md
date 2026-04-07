# MedAI Diagnosis

An AI-powered medical diagnostic system that combines computer vision, natural language processing, and large language models to assist healthcare professionals across three distinct clinical pathways.

---

## Overview

MedAI integrates four independent diagnostic modules into a unified web interface. Each module targets a different input modality — radiological imaging, MRI scanning, and free-text symptom description — and produces interpretable results supported by visual and textual explanations.

The system is designed around the principle that AI should explain its reasoning, not just deliver a verdict. Every prediction is accompanied by a visualisation or explanation that allows a clinician to assess the model's confidence and the features driving its decision.
Check out [Link Text](https://example.com)
---

## Modules

**Chest X-Ray Analysis**
Classifies chest radiographs as Normal or Pneumonia using a fine-tuned EfficientNet-B0 convolutional network. GradCAM overlays highlight the lung regions most influential in the classification decision, providing spatial accountability for each prediction.

**Brain MRI Analysis**
Identifies four neurological conditions — Glioma, Meningioma, Pituitary Tumor, and No Tumor — from MRI scans. A second EfficientNet-B0 model trained on a merged dataset of 8,470 scans generates GradCAM heatmaps that localise the suspected lesion site.

**Symptom Checker**
Accepts free-text symptom descriptions in Azerbaijani, English, or Russian. A two-stage LLM pipeline translates and extracts structured symptom data, which is fed into an XGBoost classifier covering 41 diseases across 133 binary symptom features. SHAP values quantify each symptom's contribution to the predicted diagnosis.

**Medical Chatbot**
A retrieval-augmented conversational assistant that answers questions about diagnosis results and general medical topics. Responses are grounded in a Wikipedia-derived knowledge base indexed in FAISS, with the patient's most recent diagnostic result injected as additional context.

---

## Architecture

```
Browser (HTML / CSS / JS)
        |
        | HTTP / SSE
        v
FastAPI Backend
        |
   _____|_____
  |           |
Services    Routers
  |
  |___________________________
  |          |        |       |
lung      brain   symptom   chat
  |          |        |       |
EfficientNet  EfficientNet  XGBoost  Llama-3.3-70B
+ GradCAM    + GradCAM    + SHAP   + RAG (FAISS)
  |
CLIP Filter (image validation)
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| Computer Vision | PyTorch, timm, pytorch-grad-cam |
| Image Validation | CLIP (openai/clip-vit-base-patch32) |
| Classical ML | XGBoost, SHAP, scikit-learn |
| NLP / Embeddings | sentence-transformers (MiniLM-L12-v2) |
| Vector Search | FAISS (IndexFlatIP) |
| LLM | Groq API — Llama-3.3-70B-versatile |
| Knowledge Base | Wikipedia via RAG |
| Frontend | Vanilla HTML, CSS, JavaScript |

---

## Models

| Module | Architecture | Dataset | Accuracy |
|---|---|---|---|
| Lung | EfficientNet-B0 | Kaggle chest-xray-pneumonia (5,216 train) | 91.19% |
| Brain | EfficientNet-B0 | Two merged Kaggle datasets (8,470 train) | 93.23% |
| Symptom | XGBoost | Disease Symptom Prediction (4,920 rows) | 100% val |

---

## Installation

**Requirements:** Python 3.11, CUDA-capable GPU recommended

```bash
git clone https://github.com/your-username/Medical-AI.git
cd Medical-AI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Dataset Setup

Download the following datasets and place them under `data/`:

| Dataset | Destination |
|---|---|
| [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | `data/chest_xray/` |
| [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | `data/brain_tumor/` (merge both datasets) |
| [Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) | `data/symptoms/` |

---

## Training

Run each training script from the project root with the virtual environment active:

```bash
python scripts/train_lung.py
python scripts/train_brain.py
python scripts/train_symptom.py
python scripts/build_rag.py
```

Trained weights are saved to `saved_models/`.

---

## Running the Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser.

All models are pre-loaded at startup. The first request requires no additional loading time.

---

## Project Structure

```
Medical-AI/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── ml/
│   │   ├── lung_model.py
│   │   ├── brain_model.py
│   │   ├── symptom_checker.py
│   │   ├── clip_filter.py
│   │   └── rag.py
│   ├── services/
│   ├── routers/
│   ├── schemas/
│   └── utils/
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
├── scripts/
│   ├── train_lung.py
│   ├── train_brain.py
│   ├── train_symptom.py
│   └── build_rag.py
├── saved_models/
├── data/
├── requirements.txt
├── Dockerfile
└── .env
```

---

## Key Design Decisions

**Transfer Learning over training from scratch** — EfficientNet-B0 pre-trained on ImageNet provides strong visual feature extraction from as few as 5,000 medical images. Training from random weights on this scale of data would produce significantly lower accuracy.

**GradCAM for explainability** — Medical AI systems must justify their predictions. GradCAM computes gradient-weighted class activation maps that highlight which image regions drove the classification, giving clinicians a basis for evaluating model reliability.

**Two-stage LLM pipeline for multilingual input** — Symptom descriptions arrive in Azerbaijani, Russian, or English. A dedicated translation call normalises the input to English before symptom extraction, dramatically improving extraction accuracy compared to a single combined prompt.

**RAG over a static knowledge base** — Rather than relying solely on the LLM's training data, the chatbot retrieves relevant passages from a curated Wikipedia knowledge base at query time. This reduces hallucination and keeps medical explanations grounded in verifiable sources.

**Vanilla JavaScript frontend** — No build tooling, no framework, no bundler. The frontend is served as static files directly from FastAPI. This eliminates an entire deployment layer and keeps the system self-contained.

---

## Limitations

This system is intended for research and educational purposes only. It is not a certified medical device and must not be used as a substitute for professional clinical judgment. All outputs should be reviewed by a qualified healthcare provider before any clinical decision is made.

---

## License

MIT License
