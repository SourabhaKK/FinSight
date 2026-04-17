import re
from html.parser import HTMLParser

from sklearn.model_selection import train_test_split


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


class TextCleaner:
    def clean(self, text: str) -> str:
        # 1. Lowercase
        text = text.lower()

        # 2. Strip HTML tags
        stripper = _HTMLStripper()
        stripper.feed(text)
        text = stripper.get_text()

        # 3. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 4. Remove non-ASCII characters
        text = text.encode("ascii", errors="ignore").decode("ascii")

        # 5. Normalise whitespace
        text = re.sub(r"\s+", " ", text)

        # 6. Strip leading/trailing whitespace
        return text.strip()

    def create_splits(
        self,
        texts: list[str],
        labels: list[int],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
        # First split off test set
        x_temp, x_test, y_temp, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # val_size is fraction of TOTAL dataset; adjust for remaining split
        adjusted_val = val_size / (1.0 - test_size)

        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=adjusted_val,
            random_state=random_state,
            stratify=y_temp,
        )

        return x_train, x_val, x_test, y_train, y_val, y_test
