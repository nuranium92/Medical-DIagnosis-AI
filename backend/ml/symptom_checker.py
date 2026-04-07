import os
import sys
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from groq import Groq

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.config import (
    SYMPTOM_XGB_PATH, LABEL_ENCODER_PATH,
    SYMPTOM_LIST_PATH, SYMPTOM_CSV_PATH,
    EMBEDDING_MODEL_ID, GROQ_API_KEY, GROQ_MODEL,
)

_clf          = None
_le           = None
_symptom_cols = None
_matcher      = None


def train(csv_path: str = SYMPTOM_CSV_PATH):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    symptom_cols = [c for c in df.columns if c.lower() != "prognosis"]
    df[symptom_cols] = df[symptom_cols].fillna(0)

    X  = df[symptom_cols].values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df["prognosis"].values)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        device="cpu",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_acc = (clf.predict(X_val) == y_val).mean()
    print(f"Val Accuracy: {val_acc:.4f}")

    os.makedirs(os.path.dirname(SYMPTOM_XGB_PATH), exist_ok=True)
    with open(SYMPTOM_XGB_PATH,   "wb") as f: pickle.dump(clf,          f)
    with open(LABEL_ENCODER_PATH, "wb") as f: pickle.dump(le,           f)
    with open(SYMPTOM_LIST_PATH,  "wb") as f: pickle.dump(symptom_cols, f)

    print(f"Saved: {SYMPTOM_XGB_PATH}")
    print(f"Saved: {LABEL_ENCODER_PATH}")
    print(f"Saved: {SYMPTOM_LIST_PATH}")
    return clf, le, symptom_cols


def load_models():
    global _clf, _le, _symptom_cols, _matcher
    with open(SYMPTOM_XGB_PATH,   "rb") as f: _clf          = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f: _le           = pickle.load(f)
    with open(SYMPTOM_LIST_PATH,  "rb") as f: _symptom_cols = pickle.load(f)
    _matcher = SemanticMatcher(_symptom_cols)
    print(f"Symptom models loaded. Diseases: {len(_le.classes_)}")


class SemanticMatcher:
    def __init__(self, symptom_cols: list):
        self.symptom_cols = symptom_cols
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
        texts = [s.replace("_", " ").lower() for s in symptom_cols]
        self.embeddings = self.embedder.encode(
            texts, normalize_embeddings=True
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print(f"Semantic matcher ready. {len(symptom_cols)} symptoms indexed.")

    def match(self, user_text: str, top_k: int = 5, threshold: float = 0.35) -> list:
        query = self.embedder.encode(
            [user_text.lower()], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        return [
            self.symptom_cols[idx]
            for score, idx in zip(scores[0], indices[0])
            if score >= threshold
        ]


def translate_to_english(user_text: str) -> str:
    if not GROQ_API_KEY:
        return user_text
    try:
        prompt = f"""You are a medical translator. Translate the following text to English accurately.
Pay special attention to medical symptoms.

Important Azerbaijani medical terms:
- qızdırma, hərarət = fever
- baş ağrısı = headache
- öskürək = cough
- halsızlıq = fatigue, weakness
- ürəkbulanma = nausea
- qusma = vomiting
- ishal = diarrhea
- iştahsızlıq = loss of appetite
- boğaz ağrısı = sore throat
- burun axması = runny nose
- əzələ ağrısı = muscle pain
- oynaq ağrısı = joint pain
- nəfəs darlığı = shortness of breath
- döküntü = rash
- qaşınma = itching
- sarılıq = jaundice
- ağızdan qan = blood in sputum
- sinə ağrısı = chest pain
- tər = sweating
- üşütmə = chills

If already in English, return as is.
Return only the translation, nothing else.

Text: "{user_text}"

Translation:"""

        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return user_text


def extract_symptoms_with_llm(english_text: str) -> list:
    if not GROQ_API_KEY:
        return []
    try:
        symptom_list_str = ", ".join(_symptom_cols) if _symptom_cols else ""
        prompt = f"""You are a medical assistant. Extract medical symptoms from the text below.

Symptom list: {symptom_list_str}

Text: "{english_text}"

Rules:
- Return only symptom names that exist in the symptom list above
- Separate with comma
- If no symptoms found, return empty
- Do not explain, just return symptom names
- Use underscore between words exactly as in the symptom list
- Map "fever" to "high_fever" if high_fever exists in the list
- Map "loss of appetite" to "loss_of_appetite"
- Map "fatigue" to "fatigue" or "malaise" whichever exists in the list
- Example output: high_fever, headache, loss_of_appetite

Output:"""

        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        raw     = resp.choices[0].message.content.strip()
        symbols = [s.strip().lower().replace(" ", "_") for s in raw.split(",")]
        matched = [s for s in symbols if s in _symptom_cols]
        return matched
    except Exception:
        return []


def generate_llm_summary(
    matched_symptoms: list,
    predictions: list,
    explanation: list,
    original_text: str = "",
) -> str:
    if not GROQ_API_KEY:
        return ""

    try:
        top_disease  = predictions[0]["disease"] if predictions else "unknown"
        top_prob     = predictions[0]["probability"] * 100 if predictions else 0
        top_symptoms = [e["symptom"] for e in explanation[:3] if e["impact"] > 0]

        prompt = f"""You are a medical assistant. Based on symptom analysis results, write a brief 2-3 sentence summary in the same language as the original user text.

Original user text: "{original_text}"
Matched symptoms: {', '.join(matched_symptoms[:8])}
Top diagnosis: {top_disease} ({top_prob:.1f}% probability)
Most influential symptoms: {', '.join(top_symptoms)}

Write a concise, clear summary explaining:
1. What the main symptoms suggest
2. Why the top diagnosis was predicted
3. A gentle reminder to consult a doctor

Keep it simple and understandable for a non-medical person. Do not use bullet points."""

        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception:
        return ""


def predict(user_text: str) -> dict:
    global _clf, _le, _symptom_cols, _matcher
    if _clf is None:
        load_models()

    english_text = translate_to_english(user_text)
    print(f"[Symptom] Original:   {user_text}")
    print(f"[Symptom] Translated: {english_text}")

    llm_matched = extract_symptoms_with_llm(english_text)
    combined    = list(set(llm_matched))

    print(f"[Symptom] LLM:      {llm_matched}")
    print(f"[Symptom] Combined: {combined}")

    if not combined or len(combined) < 2:
        return {
            "error": (
                "Yalnız 1-2 simptom tapıldı. Daha dəqiq nəticə üçün "
                "ən azı 3-4 simptom yazın. Məsələn: "
                "'qızdırma, baş ağrısı, halsızlıq, iştahsızlıq var'."
            )
        }

    vec = np.zeros((1, len(_symptom_cols)), dtype=np.float32)
    for sym in combined:
        if sym in _symptom_cols:
            vec[0, _symptom_cols.index(sym)] = 1.0

    proba    = _clf.predict_proba(vec)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    predictions = [
        {"disease": _le.classes_[i], "probability": round(float(proba[i]), 4)}
        for i in top3_idx
    ]

    explainer   = shap.TreeExplainer(_clf)
    shap_values = explainer.shap_values(vec)
    pred_class  = int(_clf.predict(vec)[0])
    sv          = shap_values[0, :, pred_class]
    top_idx     = np.argsort(np.abs(sv))[::-1][:8]
    explanation = [
        {"symptom": _symptom_cols[i].replace("_", " "), "impact": round(float(sv[i]), 4)}
        for i in top_idx
        if abs(float(sv[i])) > 0.001
    ]

    llm_summary    = generate_llm_summary(combined, predictions, explanation, user_text)
    low_confidence = predictions[0]["probability"] < 0.25

    return {
        "matched_symptoms": combined,
        "predictions":      predictions,
        "explanation":      explanation,
        "llm_summary":      llm_summary,
        "low_confidence":   low_confidence,
    }