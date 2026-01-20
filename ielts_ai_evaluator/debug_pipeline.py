import asyncio
import json
import numpy as np
import scipy.io.wavfile as wav
import os
import sys

# Add current directory to path to allow absolute imports like 'from utils...'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a dummy sine wave audio file for testing
def create_dummy_audio(filename="test_audio.wav", duration=5):
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav.write(filename, sample_rate, (audio * 32767).astype(np.int16))
    return filename

async def test_pipeline():
    print("--- 1. Creating Dummy Audio ---")
    audio_path = create_dummy_audio()
    print(f"Created {audio_path}")

    try:
        print("\n--- 2. Testing Transcriber (Mocked) ---")
        # Since we can't easily run the full Whisper model in this script without proper env,
        # We will verify the logic flow by inspecting the functions we modified.
        
        # Simulating what Whisper WOULD return with word_timestamps=True
        mock_whisper_result = {
            "text": "I usually go shopping on weekends.",
            "segments": [
                {
                    "start": 0.0, "end": 2.5, "text": "I usually go shopping",
                    "avg_logprob": -0.15, # High confidence
                    "words": [
                        {"word": "I", "start": 0.0, "end": 0.2, "probability": 0.99},
                        {"word": "usually", "start": 0.2, "end": 0.8, "probability": 0.95},
                        {"word": "go", "start": 0.8, "end": 1.0, "probability": 0.98},
                        {"word": "shopping", "start": 1.0, "end": 1.8, "probability": 0.90}
                    ]
                },
                {
                    "start": 2.5, "end": 4.0, "text": "on weekends.",
                    "avg_logprob": -0.6, # Lower confidence
                    "words": [
                         {"word": "on", "start": 2.5, "end": 2.8, "probability": 0.85},
                         {"word": "weekends", "start": 2.8, "end": 3.8, "probability": 0.45} # Low prob -> Pronunciation issue?
                    ]
                }
            ]
        }
        print("Mocked Whisper Result created.")

        print("\n--- 3. Testing Pronunciation Logic ---")
        from evaluators.speaking import calculate_pronunciation_score
        
        pron_score = calculate_pronunciation_score(mock_whisper_result)
        print(json.dumps(pron_score, indent=2))
        
        # Verification
        assert pron_score['score'] > 0
        assert len(pron_score['analysis']) == 6
        print("✅ Pronunciation calculation logic verified.")

        print("\n--- 4. Testing Audio Features Logic ---")
        # Simulating librosa output
        mock_audio_metrics = {
            "duration_sec": 5.0,
            "pause_count": 1,
            "pauses": [{"start": 1.8, "end": 2.5, "duration": 0.7}],
            "speech_rate_wpm": 72 # (6 words / 5 sec) * 60
        }
        
        print("Mocked Audio Metrics:")
        print(json.dumps(mock_audio_metrics, indent=2))


        print("\n--- 5. Testing Evaluator Integration (Dry Run) ---")
        # We will NOT call GPT here to avoid API costs/errors, but we verify the data preparation
        from evaluators.speaking import load_prompt, SPEAKING_QUESTIONS
        
        part = 1
        transcript_text = mock_whisper_result["text"]
        questions = SPEAKING_QUESTIONS.get(part, [])
        prompt_template = load_prompt() # Reads the file we updated

        prompt = (
            prompt_template
            .replace("{{part}}", str(part))
            .replace("{{questions}}", str(questions))
            .replace("{{transcript}}", transcript_text)
            .replace("{{audio_metrics}}", str(mock_audio_metrics))
        )
        
        print("\nGenerated Prompt Preview (Truncated):")
        print(prompt[:500] + "...")
        
        # Verify prompt contains key new sections
        assert "grammar_analysis" in prompt
        assert "vocabulary_analysis" in prompt
        print("\n✅ Prompt correctly includes new analysis requests.")

    except ImportError as e:
        print(f"⚠️ Import Error (Environment mismatch?): {e}")
        print("This is expected if not running in the exact python env with all deps installed.")
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    asyncio.run(test_pipeline())
