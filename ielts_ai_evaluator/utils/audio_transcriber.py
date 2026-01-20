import whisper
import tempfile
import os
from fastapi import UploadFile
from pydub import AudioSegment

model = whisper.load_model("base")


def transcribe_audio(file: UploadFile) -> dict:
    """
    Transcribe audio from an UploadFile object.
    Supports webm, mp3, wav formats.
    """
    # Read the uploaded file content
    content = file.file.read()
    file.file.seek(0)  # Reset for potential re-use
    
    # Get file extension
    filename = file.filename or "audio.webm"
    ext = os.path.splitext(filename)[1].lower() or ".webm"
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_input:
        temp_input.write(content)
        temp_input_path = temp_input.name
    
    try:
        # Convert to WAV if not already (Whisper works best with WAV)
        if ext != ".wav":
            audio = AudioSegment.from_file(temp_input_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            audio.export(temp_wav_path, format="wav")
            transcribe_path = temp_wav_path
        else:
            transcribe_path = temp_input_path
            temp_wav_path = None
        
        # Transcribe with Whisper (verbose=True for segments/words)
        result = model.transcribe(transcribe_path, word_timestamps=True)
        
        # Return structured data
        return {
            "text": result["text"],
            "segments": result["segments"],  # Includes start, end, avg_logprob
            "language": result["language"]
        }
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if ext != ".wav" and temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

