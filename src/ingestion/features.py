from src.ingestion.schema import ArticleIn


def extract_features(article: ArticleIn) -> dict[str, float]:
    text = article.text.strip()

    if not text:
        return {
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "digit_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "exclamation_count": 0.0,
            "question_count": 0.0,
            "text_length": 0.0,
        }

    words = text.split()
    word_count = float(len(words))
    avg_word_length = float(sum(len(w) for w in words) / len(words)) if words else 0.0

    total_chars = len(text)
    digit_ratio = float(sum(c.isdigit() for c in text) / total_chars)
    uppercase_ratio = float(sum(c.isupper() for c in text) / total_chars)
    exclamation_count = float(text.count("!"))
    question_count = float(text.count("?"))
    text_length = float(total_chars)

    return {
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "digit_ratio": digit_ratio,
        "uppercase_ratio": uppercase_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "text_length": text_length,
    }
