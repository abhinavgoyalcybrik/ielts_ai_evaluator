from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evaluators.api.speaking import router as speaking_router
from evaluators.api.writing import router as writing_router
from evaluators.api.reading import router as reading_router
from evaluators.api.listening import router as listening_router

app = FastAPI(
    title="IELTS AI Evaluator API",
    description="AI-powered IELTS Speaking, Writing, Reading & Listening Evaluation API",
    version="1.0.0"
)

# --------------------
# CORS Configuration
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Include Routers
# --------------------
app.include_router(speaking_router)
app.include_router(writing_router)
app.include_router(reading_router)
app.include_router(listening_router)

# --------------------
# Health Check
# --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "IELTS AI Evaluator API"
    }

# --------------------
# Root
# --------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "IELTS AI Evaluator API is live"
    }
