import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from backend.routers.lung    import router as lung_router
from backend.routers.brain   import router as brain_router
from backend.routers.symptom import router as symptom_router
from backend.routers.chat    import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading all models on startup...")

    from backend.ml.lung_model      import get_lung_model
    from backend.ml.brain_model     import load_model
    from backend.ml.symptom_checker import load_models
    from backend.ml.clip_filter     import get_clip
    from backend.ml.rag             import load_rag

    get_lung_model()
    load_model()
    load_models()
    get_clip()
    load_rag()

    print("All models loaded. Server ready.")
    yield


app = FastAPI(
    title="AI Medical Diagnosis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lung_router)
app.include_router(brain_router)
app.include_router(symptom_router)
app.include_router(chat_router)

frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok"}