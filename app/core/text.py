from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata


def strip_accents(value: object) -> str:
    text = "" if value is None else str(value)
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def normalize_text(value: object) -> str:
    text = strip_accents(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_column(value: object) -> str:
    return normalize_text(value).lower().replace(" ", "_")


def similarity(left: object, right: object) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_set_ratio(a, b)) / 100.0
    except Exception:
        return SequenceMatcher(None, a, b).ratio()


