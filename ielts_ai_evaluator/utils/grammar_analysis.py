import re


COMPLEX_MARKERS = [
    "because", "although", "though", "while", "whereas",
    "if", "unless", "when", "whenever",
    "which", "that", "who", "whom", "whose",
    "so that", "even though", "in order to"
]


def split_sentences(text: str):
    return re.split(r"[.!?]+", text)


def grammar_range_metrics(transcript: str):
    sentences = [
        s.strip().lower()
        for s in split_sentences(transcript)
        if s.strip()
    ]

    if not sentences:
        return {
            "simple_sentences": 0,
            "complex_sentences": 0,
            "complex_ratio": 0.0
        }

    complex_count = 0

    for sentence in sentences:
        if any(marker in sentence for marker in COMPLEX_MARKERS):
            complex_count += 1

    total = len(sentences)
    simple_count = total - complex_count

    complex_ratio = round(complex_count / total, 3)

    return {
        "simple_sentences": simple_count,
        "complex_sentences": complex_count,
        "complex_ratio": complex_ratio
    }
