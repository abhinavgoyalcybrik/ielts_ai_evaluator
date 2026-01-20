from utils.gpt_client import call_gpt
from utils.band import round_band
from pathlib import Path

SPEAKING_QUESTIONS = {
    1: [
        "What is your hometown?",
        "Do you like living there?"
    ],
    2: [
        "Describe a memorable trip you took."
    ],
    3: [
        "Why do people like travelling?",
        "How has tourism changed recently?"
    ]
}


def load_prompt():
    return Path("prompts/speaking_prompt.txt").read_text(encoding="utf-8")


def calculate_pronunciation_score(transcript_data: dict) -> dict:
    """
    Calculate pronunciation score based on Whisper logprobs.
    Returns band score component and word-level analysis.
    """
    if not isinstance(transcript_data, dict) or "segments" not in transcript_data:
        return {"score": 6.0, "analysis": []}  # Default fallback

    total_prob = 0
    count = 0
    word_analysis = []
    
    # Process segments
    for segment in transcript_data.get("segments", []):
        # Use segment confidence (avg_logprob) as proxy for clarity
        # logprob is usually negative, close to 0 is good.
        # e.g., -0.1 is 90% confidence, -1.0 is 36%
        # Simple mapping: > -0.2 Good, > -0.6 Fair, else Poor
        # Note: This is an approximation.
        
        confidence = float(segment.get("avg_logprob", -1.0))
        # Convert logprob to probability: e^logprob
        prob = 2.718 ** confidence
        
        status = "Needs Improvement"
        if prob > 0.8:
            status = "Good"
        elif prob > 0.5:
            status = "Fair"
            
        # Add word analysis (Whisper segments are usually phrases, but words if verbose)
        # If words available, use them
        if "words" in segment:
            for word_info in segment["words"]:
                 w_prob = word_info.get("probability", prob) # Use word prob if avail
                 w_status = "Good" if w_prob > 0.8 else ("Fair" if w_prob > 0.5 else "Needs Improvement")
                 word_analysis.append({
                     "word": word_info["word"].strip(),
                     "status": w_status,
                     "score": round(w_prob * 100)
                 })
                 total_prob += w_prob
                 count += 1
        else:
            # Fallback to segment text
            text = segment.get("text", "").strip()
            total_prob += prob
            count += 1
            word_analysis.append({
                "word": text, # Might be a phrase
                "status": status,
                "score": round(prob * 100)
            })

    if count == 0:
        return {"score": 6.0, "analysis": []}

    avg_prob = total_prob / count
    # Map avg probability (0-1) to Band Score (0-9)
    # A generous mapping: 0.8+ -> 8-9, 0.6 -> 6-7, etc.
    pronunciation_band = min(9.0, max(1.0, avg_prob * 10))
    
    return {
        "score": round(pronunciation_band, 1),
        "analysis": word_analysis
    }


def evaluate_speaking_part(part, transcript, audio_metrics):
    questions = SPEAKING_QUESTIONS.get(part, [])
    prompt_template = load_prompt()
    
    # Handle transcript input (dict or str)
    if isinstance(transcript, dict):
        transcript_text = transcript.get("text", "")
        transcript_data = transcript
    else:
        transcript_text = str(transcript)
        transcript_data = {}

    prompt = (
        prompt_template
        .replace("{{part}}", str(part))
        .replace("{{questions}}", str(questions))
        .replace("{{transcript}}", transcript_text)
        .replace("{{audio_metrics}}", str(audio_metrics))
    )

    # Base GPT evaluation
    result_str = call_gpt(prompt)
    
    try:
        import json
        # Clean up code blocks if Present
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0].strip()
        elif "```" in result_str:
            result_str = result_str.split("```")[1].split("```")[0].strip()
            
        result = json.loads(result_str)
    except Exception as e:
        print(f"Failed to parse GPT response: {result_str}")
        # Fallback to basic result
        result = {
            "fluency": 5, "lexical": 5, "grammar": 5, "pronunciation": 5,
            "overall_band": 5, "feedback": {"strengths": "Error parsing result", "improvements": str(e)}
        }

    # 🔴 FLUENCY HARD RULES (IELTS-STYLE) & ANALYSIS
    fluency = result.get("fluency", 0)
    wpm = audio_metrics.get("speech_rate_wpm", 0)
    pauses = audio_metrics.get("pause_count", 0)

    if wpm < 90: fluency -= 1
    if wpm > 180: fluency -= 1
    if pauses > 5: fluency -= 0.5  # Soften penalty

    result["fluency"] = max(0, min(9, fluency))
    
    # 🔴 PRONUNCIATION ANALYSIS
    pron_analysis = calculate_pronunciation_score(transcript_data)
    # Blend GPT score with Audio Analysis score (50/50?)
    gpt_pron = result.get("pronunciation", 6)
    audio_pron = pron_analysis["score"]
    
    # Weighted average: Audio metrics 60%, GPT (content-based) 40%
    final_pron = (audio_pron * 0.6) + (gpt_pron * 0.4)
    result["pronunciation"] = round(final_pron * 2) / 2 # Round to nearest 0.5
    
    # Add detailed analysis to result
    result["transcript"] = transcript_text
    result["pronunciation_analysis"] = pron_analysis["analysis"]
    result["fluency_analysis"] = {
        "wpm": wpm,
        "pauses": audio_metrics.get("pauses", []),
        "pause_count": pauses
    }

    # Recalculate overall band
    avg = (
        result["fluency"]
        + result["lexical"]
        + result["grammar"]
        + result["pronunciation"]
    ) / 4

    result["overall_band"] = round_band(avg)
    return result


# 🔁 BACKWARD-COMPATIBILITY WRAPPER
def evaluate_speaking(data: dict):
    return evaluate_speaking_part(
        part=data["part"],
        transcript=data["transcript"],
        audio_metrics=data["audio_metrics"]
    )
