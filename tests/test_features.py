from src.ingestion.features import extract_features
from src.ingestion.schema import ArticleIn

EXPECTED_KEYS = {
    "word_count",
    "avg_word_length",
    "digit_ratio",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "text_length",
}


def make_article(text: str) -> ArticleIn:
    if len(text) < 10:
        text = text.ljust(10)
    return ArticleIn(text=text)


def test_all_seven_keys_present() -> None:
    article = make_article("The market dropped 5% today in New York.")
    features = extract_features(article)
    assert set(features.keys()) == EXPECTED_KEYS


def test_all_values_are_float() -> None:
    article = make_article("The market dropped 5% today in New York.")
    features = extract_features(article)
    for key, value in features.items():
        assert isinstance(value, float), f"{key} is not float: {type(value)}"


def test_empty_text_returns_all_zeros() -> None:
    # 10 spaces — min length met but empty after strip
    article = ArticleIn(text="          ")
    features = extract_features(article)
    for key, value in features.items():
        assert value == 0.0, f"{key} should be 0.0 for empty text, got {value}"


def test_word_count_correct() -> None:
    article = make_article("one two three four five")
    features = extract_features(article)
    assert features["word_count"] == 5.0


def test_text_length_correct() -> None:
    text = "Hello world test"
    article = make_article(text)
    features = extract_features(article)
    assert features["text_length"] == float(len(text))


def test_digit_ratio_between_zero_and_one() -> None:
    article = make_article("abc 123 xyz 456 hello world today")
    features = extract_features(article)
    assert 0.0 <= features["digit_ratio"] <= 1.0


def test_uppercase_ratio_between_zero_and_one() -> None:
    article = make_article("ABC def GHI jkl MNO pqr STU vwx")
    features = extract_features(article)
    assert 0.0 <= features["uppercase_ratio"] <= 1.0


def test_all_digits_gives_digit_ratio_one() -> None:
    article = ArticleIn(text="1234567890")
    features = extract_features(article)
    assert features["digit_ratio"] == 1.0


def test_all_uppercase_gives_uppercase_ratio_one() -> None:
    article = ArticleIn(text="ABCDEFGHIJ")
    features = extract_features(article)
    assert features["uppercase_ratio"] == 1.0


def test_exclamation_count_correct() -> None:
    article = make_article("Wow! This is amazing! Really great stuff here!!")
    features = extract_features(article)
    assert features["exclamation_count"] == 4.0


def test_question_count_correct() -> None:
    article = make_article("What happened? Where? Why did this occur here today?")
    features = extract_features(article)
    assert features["question_count"] == 3.0


def test_no_digits_gives_zero_digit_ratio() -> None:
    article = make_article("No digits here at all in this sentence.")
    features = extract_features(article)
    assert features["digit_ratio"] == 0.0


def test_avg_word_length_correct() -> None:
    article = ArticleIn(text="ab cde fghi padding")
    features = extract_features(article)
    words = article.text.split()
    expected = sum(len(w) for w in words) / len(words)
    assert abs(features["avg_word_length"] - expected) < 1e-9
