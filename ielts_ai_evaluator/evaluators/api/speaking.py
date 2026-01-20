from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from uuid import uuid4

from utils.audio_transcriber import transcribe_audio
from utils.audio_features import extract_audio_features
from evaluators.speaking import evaluate_speaking_part
from storage.speaking_store import SPEAKING_ATTEMPTS

router = APIRouter(prefix="/speaking", tags=["Speaking"])


@router.post("/part/{part}/audio")
async def upload_speaking_audio(
    part: int,
    file: UploadFile = File(...),
    attempt_id: str | None = Form(None)

):
    if part not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Invalid part number")

    # 🔑 Generate or use provided attempt_id
    # For Part 1: Generate if not provided
    # For Parts 2 & 3: Require attempt_id but auto-create entry if not exists
    if not attempt_id:
        if part == 1:
            attempt_id = uuid4().hex
        else:
            raise HTTPException(
                status_code=400,
                detail="attempt_id is required for Part 2 and Part 3"
            )
    
    # Auto-create entry if it doesn't exist (supports frontend-generated IDs)
    if attempt_id not in SPEAKING_ATTEMPTS:
        SPEAKING_ATTEMPTS[attempt_id] = {"parts": {}}

    # ---- AUDIO PROCESSING ----
    transcript_result = transcribe_audio(file)
    audio_metrics = extract_audio_features(file)

    # Handle text extraction
    if isinstance(transcript_result, dict):
        transcript_text = transcript_result.get("text", "")
    else:
        transcript_text = str(transcript_result)
        transcript_result = {"text": transcript_text}

    # Speech rate (WPM)
    words = len(transcript_text.split())
    duration = audio_metrics.get("duration_sec", 1)
    speech_rate = round((words / duration) * 60) if duration > 0 else 0

    audio_metrics["speech_rate_wpm"] = speech_rate

    # ---- EVALUATION ----
    result = evaluate_speaking_part(
        part=part,
        transcript=transcript_result,  # Pass full object
        audio_metrics=audio_metrics
    )

    # ---- STORE RESULT (SAME attempt_id) ----
    SPEAKING_ATTEMPTS[attempt_id]["parts"][part] = result

    return {
        "attempt_id": attempt_id,
        "part": part,
        "result": result
    }
