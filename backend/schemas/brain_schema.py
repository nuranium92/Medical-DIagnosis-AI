from pydantic import BaseModel
from typing import Dict


class BrainRequest(BaseModel):
    image_b64: str


class BrainResponse(BaseModel):
    label:          str
    raw_label:      str
    confidence:     float
    probabilities:  Dict[str, float]
    gradcam_b64:    str
    prob_chart_b64: str