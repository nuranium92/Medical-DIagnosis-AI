import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, HTTPException
from backend.schemas.brain_schema import BrainRequest, BrainResponse
from backend.services.brain_service import process_brain

router = APIRouter(prefix="/api/brain", tags=["brain"])


@router.post("/predict", response_model=BrainResponse)
async def predict_brain(req: BrainRequest):
    try:
        result = process_brain(req.image_b64)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))