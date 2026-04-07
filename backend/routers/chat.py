import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from groq import Groq
from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.services.chat_service import (
    process_chat, is_non_medical, is_greeting, SYSTEM_PROMPT
)
from backend.ml.rag import retrieve, format_context
from backend.config import GROQ_API_KEY, GROQ_MODEL

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        result  = process_chat(req.message, history, req.diagnosis_context or "")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        message = req.message
        context = req.diagnosis_context or ""

        if is_non_medical(message) and not context:
            refuse = (
                "Üzr istəyirəm, bu mövzu mənim ixtisasım xaricindədir. "
                "Mən yalnız tibbi diaqnoz, simptomlar və sağlamlıq məsələləri ilə bağlı "
                "kömək edə bilərəm. Sağlamlığınızla bağlı sualınız varsa, məmnuniyyətlə kömək edərəm."
            )
            history.append({"role": "user",      "content": message})
            history.append({"role": "assistant", "content": refuse})

            async def refuse_gen():
                for word in refuse.split(" "):
                    yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True, 'history': history})}\n\n"

            return StreamingResponse(refuse_gen(), media_type="text/event-stream")

        if not GROQ_API_KEY:
            async def no_key():
                yield f"data: {json.dumps({'token': 'GROQ_API_KEY təyin edilməyib.', 'done': True, 'history': history})}\n\n"
            return StreamingResponse(no_key(), media_type="text/event-stream")

        rag_results = []
        if not is_greeting(message):
            rag_results = retrieve(message, top_k=3)

        rag_context = format_context(rag_results)

        client  = Groq(api_key=GROQ_API_KEY)
        system  = SYSTEM_PROMPT
        if rag_context:
            system += f"\n\n{rag_context}"
        if context:
            system += f"\n\nCURRENT PATIENT DIAGNOSTIC DATA:\n{context}"

        messages = [{"role": "system", "content": system}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        async def token_gen():
            full_response = ""
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

            history.append({"role": "user",      "content": message})
            history.append({"role": "assistant", "content": full_response})
            yield f"data: {json.dumps({'token': '', 'done': True, 'history': history})}\n\n"

        return StreamingResponse(token_gen(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))