import librosa
import tempfile
import os
from fastapi import UploadFile
from pydub import AudioSegment


def extract_audio_features(file: UploadFile) -> dict:
    """
    Extract audio features from an UploadFile object.
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
    
    temp_wav_path = None
    try:
        # Convert to WAV if not already (librosa works best with WAV)
        if ext != ".wav":
            audio = AudioSegment.from_file(temp_input_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            audio.export(temp_wav_path, format="wav")
            load_path = temp_wav_path
        else:
            load_path = temp_input_path
        
        # Load and analyze with librosa
        y, sr = librosa.load(load_path, sr=16000)
        
        duration = librosa.get_duration(y=y, sr=sr)
        intervals = librosa.effects.split(y, top_db=25)
        pause_count = max(0, len(intervals) - 1)
        
        return {
            "duration_sec": round(duration, 2),
            "pause_count": pause_count
        }
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

