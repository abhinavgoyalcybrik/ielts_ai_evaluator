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
        
        # Detect non-silent intervals
        non_silent_intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
        
        # Calculate silent intervals (pauses) in seconds
        pauses = []
        last_end = 0.0
        
        for start_sample, end_sample in non_silent_intervals:
            start_sec = float(start_sample) / sr
            end_sec = float(end_sample) / sr
            
            if start_sec > last_end:
                pause_dur = start_sec - last_end
                if pause_dur > 0.3:  # Only count pauses > 300ms
                    pauses.append({
                        "start": round(last_end, 2),
                        "end": round(start_sec, 2),
                        "duration": round(pause_dur, 2)
                    })
            last_end = end_sec

        return {
            "duration_sec": round(duration, 2),
            "pause_count": len(pauses),
            "pauses": pauses
        }
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

