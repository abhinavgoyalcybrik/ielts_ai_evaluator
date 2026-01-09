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


def evaluate_speaking_part(part, transcript, audio_metrics):
    questions = SPEAKING_QUESTIONS.get(part, [])
    prompt_template = load_prompt()

    prompt = (
        prompt_template
        .replace("{{part}}", str(part))
        .replace("{{questions}}", str(questions))
        .replace("{{transcript}}", transcript)
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
        # Fallback or re-raise
        raise ValueError(f"Invalid evaluator response: {result_str}") from e

    # 🔴 FLUENCY HARD RULES (IELTS-STYLE)
    fluency = result.get("fluency", 0)

    wpm = audio_metrics.get("speech_rate_wpm", 0)
    pauses = audio_metrics.get("pause_count", 0)

    if wpm < 90:
        fluency -= 1
    if wpm > 180:
        fluency -= 1
    if pauses > 5:
        fluency -= 1

    result["fluency"] = max(0, min(9, fluency))

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
