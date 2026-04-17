from collections import Counter

import pytest

from src.preprocessing.pipeline import TextCleaner


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


# --- TextCleaner.clean ---


def test_html_stripped(cleaner: TextCleaner) -> None:
    result = cleaner.clean("<p>Hello <b>world</b></p>")
    assert "<" not in result
    assert "hello" in result
    assert "world" in result


def test_urls_removed_https(cleaner: TextCleaner) -> None:
    result = cleaner.clean("Visit https://example.com for details.")
    assert "https" not in result
    assert "example.com" not in result


def test_urls_removed_http(cleaner: TextCleaner) -> None:
    result = cleaner.clean("Go to http://news.site.org/article/123 now.")
    assert "http" not in result


def test_urls_removed_www(cleaner: TextCleaner) -> None:
    result = cleaner.clean("Check www.reuters.com for updates.")
    assert "www" not in result


def test_non_ascii_removed(cleaner: TextCleaner) -> None:
    result = cleaner.clean("caf\u00e9 and na\u00efve r\u00e9sum\u00e9")
    assert "\u00e9" not in result
    assert "\u00ef" not in result
    assert result.isascii()


def test_whitespace_normalised(cleaner: TextCleaner) -> None:
    result = cleaner.clean("hello   world\n\nfoo\tbar")
    assert "  " not in result
    assert "\n" not in result
    assert "\t" not in result


def test_lowercase_applied(cleaner: TextCleaner) -> None:
    result = cleaner.clean("GLOBAL MARKETS FELL TODAY")
    assert result == result.lower()


def test_leading_trailing_whitespace_stripped(cleaner: TextCleaner) -> None:
    result = cleaner.clean("   leading and trailing   ")
    assert result == result.strip()


def test_empty_string_returns_empty(cleaner: TextCleaner) -> None:
    result = cleaner.clean("")
    assert result == ""


def test_clean_combined(cleaner: TextCleaner) -> None:
    raw = "  <h1>BREAKING NEWS</h1> Visit https://cnn.com for more!  Caf\u00e9  "
    result = cleaner.clean(raw)
    assert "<h1>" not in result
    assert "https" not in result
    assert result == result.lower()
    assert result == result.strip()
    assert result.isascii()


# --- create_splits ---


def _make_dummy_data(n: int = 100) -> tuple[list[str], list[int]]:
    texts = [f"article text number {i} about finance and markets" for i in range(n)]
    labels = [i % 4 for i in range(n)]
    return texts, labels


def test_split_sizes_sum_to_total(cleaner: TextCleaner) -> None:
    texts, labels = _make_dummy_data(100)
    x_train, x_val, x_test, y_train, y_val, y_test = cleaner.create_splits(
        texts, labels, test_size=0.2, val_size=0.1
    )
    assert len(x_train) + len(x_val) + len(x_test) == 100
    assert len(y_train) + len(y_val) + len(y_test) == 100


def test_test_size_respected(cleaner: TextCleaner) -> None:
    texts, labels = _make_dummy_data(100)
    x_train, x_val, x_test, *_ = cleaner.create_splits(
        texts, labels, test_size=0.2, val_size=0.1
    )
    assert len(x_test) == 20


def test_val_size_respected(cleaner: TextCleaner) -> None:
    texts, labels = _make_dummy_data(100)
    x_train, x_val, x_test, *_ = cleaner.create_splits(
        texts, labels, test_size=0.2, val_size=0.1
    )
    assert len(x_val) == 10


def test_stratified_split_preserves_class_ratios(cleaner: TextCleaner) -> None:
    texts, labels = _make_dummy_data(100)
    _, _, _, y_train, y_val, y_test = cleaner.create_splits(
        texts, labels, test_size=0.2, val_size=0.1
    )
    for split in [y_train, y_val, y_test]:
        counts = Counter(split)
        total = len(split)
        for label in range(4):
            ratio = counts[label] / total
            assert abs(ratio - 0.25) < 0.15, (
                f"Class {label} ratio {ratio:.2f} too far from 0.25"
            )


def test_no_leakage_splits_return_raw_lists(cleaner: TextCleaner) -> None:
    texts, labels = _make_dummy_data(100)
    x_train, x_val, x_test, y_train, y_val, y_test = cleaner.create_splits(
        texts, labels
    )
    assert isinstance(x_train, list)
    assert isinstance(x_val, list)
    assert isinstance(x_test, list)
    assert all(isinstance(t, str) for t in x_train)
    assert all(isinstance(t, str) for t in x_val)
    assert all(isinstance(t, str) for t in x_test)
    # No vectoriser was fitted — TextCleaner has no vocabulary_ or tfidf attribute
    assert not hasattr(cleaner, "vocabulary_")
    assert not hasattr(cleaner, "vectorizer")
    assert not hasattr(cleaner, "tfidf")


def test_splits_are_disjoint(cleaner: TextCleaner) -> None:
    texts, labels = _make_dummy_data(100)
    x_train, x_val, x_test, *_ = cleaner.create_splits(texts, labels)
    train_set = set(x_train)
    val_set = set(x_val)
    test_set = set(x_test)
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
