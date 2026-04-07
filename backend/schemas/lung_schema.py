from pydantic import BaseModel
from typing import Dict


class LungRequest(BaseModel):
    image_b64: str


class LungResponse(BaseModel):
    label:         str
    confidence:    float
    probabilities: Dict[str, float]
    heatmap_b64:   str
    prob_chart_b64: str