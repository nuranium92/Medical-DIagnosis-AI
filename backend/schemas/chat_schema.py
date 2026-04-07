from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    message:           str
    history:           List[ChatMessage] = []
    diagnosis_context: Optional[str]     = ""


class ChatResponse(BaseModel):
    response: str
    history:  List[ChatMessage]