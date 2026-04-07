import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, HTTPException
from backend.schemas.lung_schema import LungRequest, LungResponse
from backend.services.lung_service import process_lung

router = APIRouter(prefix="/api/lung", tags=["lung"])


@router.post("/predict", response_model=LungResponse)
async def predict_lung(req: LungRequest):
    try:
        result = process_lung(req.image_b64)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))