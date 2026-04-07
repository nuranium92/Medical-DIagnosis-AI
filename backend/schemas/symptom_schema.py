from pydantic import BaseModel
from typing import List, Optional


class SymptomRequest(BaseModel):
    text: str


class DiagnosisPrediction(BaseModel):
    disease:     str
    probability: float


class ShapItem(BaseModel):
    symptom: str
    impact:  float


class SymptomResponse(BaseModel):
    matched_symptoms: List[str]
    predictions:      List[DiagnosisPrediction]
    explanation:      List[ShapItem]
    shap_chart_b64:   str
    llm_summary:      str
    low_confidence:   bool = False
    error:            Optional[str] = None