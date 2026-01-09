from evaluators.writing import evaluate_writing
from evaluators.speaking import evaluate_speaking   # ✅ NEW LINE


def evaluate_attempt(data):
    """
    Central evaluator dispatcher
    ENABLED: Writing + Speaking
    """

    test_type = data.get("test_type")

    if test_type == "writing":
        return evaluate_writing(data)

    if test_type == "speaking":                      # ✅ NEW BLOCK
        return evaluate_speaking(data)

    raise ValueError("Only writing and speaking evaluation are enabled right now")
