from pathlib import Path
from utils.band import round_band
from utils.ai_client import call_gpt_writing, call_gpt_refine_answer


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "prompts"


def clamp(score):
    try:
        return max(0.0, min(9.0, float(score)))
    except Exception:
        return 5.0


def count_words(text: str) -> int:
    return len(text.split())


def validate_word_count(task_type: str, essay: str):
    wc = count_words(essay)
    if task_type == "task_1" and wc < 150:
        raise ValueError("Task 1 requires at least 150 words")
    if task_type == "task_2" and wc < 250:
        raise ValueError("Task 2 requires at least 250 words")
    return wc


def evaluate_writing(data: dict):

    metadata = data.get("metadata", {})
    question = metadata.get("question", "").strip()
    essay = data.get("user_answers", {}).get("text", "").strip()

    if not essay:
        raise ValueError("Essay text missing")

    task_type = "task_1" if metadata.get("task_type") in ("task1", "task_1") else "task_2"
    word_count = validate_word_count(task_type, essay)

    prompt_file = (
        PROMPTS_DIR / "writing_task1_prompt.txt"
        if task_type == "task_1"
        else PROMPTS_DIR / "writing_task2_prompt.txt"
    )

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = (
        prompt_template
        .replace("<<<QUESTION>>>", question)
        .replace("<<<ESSAY_TEXT>>>", essay)
    )

    ai = call_gpt_writing(prompt)

    tr = clamp(ai.get("task_response", 5))
    cc = clamp(ai.get("coherence_cohesion", 5))
    lr = clamp(ai.get("lexical_resource", 5))
    gr = clamp(ai.get("grammar_accuracy", 5))

    if task_type == "task_1":
        overall = tr * 0.3 + cc * 0.25 + lr * 0.25 + gr * 0.2
    else:
        overall = tr * 0.4 + cc * 0.3 + lr * 0.2 + gr * 0.1

    band = round_band(overall)

    refined = call_gpt_refine_answer(question, essay, 8)

    return {
        "overall_band": band,
        "criteria_scores": {
            "task_response": tr,
            "coherence_cohesion": cc,
            "lexical_resource": lr,
            "grammar_accuracy": gr
        },
        "mistakes": ai.get("mistakes", []),
        "refined_answer": refined,
        "word_count": word_count
    }
