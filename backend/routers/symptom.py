import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, HTTPException
from backend.schemas.symptom_schema import SymptomRequest, SymptomResponse
from backend.services.symptom_service import process_symptom

router = APIRouter(prefix="/api/symptom", tags=["symptom"])


@router.post("/predict", response_model=SymptomResponse)
async def predict_symptom(req: SymptomRequest):
    try:
        result = process_symptom(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))