"""
Malayalam/English tokenizer for BM25.

Design notes (based on an audit of the indexed corpus, not assumptions):

* The corpus encodes chillu letters the legacy way (consonant + virama + ZWJ,
  e.g. "സര്‍ക്കാര്‍"), while modern keyboards produce atomic chillus
  ("സർക്കാർ"). Both are folded to consonant + virama, and ZWJ/ZWNJ are removed,
  so queries match regardless of input method.
* Legacy-font conversions leave "ളള" for "ള്ള" and "ററ" for "റ്റ"; the old
  two-part au sign is mapped to the modern length mark. Malayalam digits are
  mapped to ASCII digits.
* Malayalam is agglutinative: one stem appears in many inflected surface
  forms (നികുതി, നികുതിയുടെ, നികുതിയും ...). Whitespace tokens alone cannot
  match a query stem against inflected document forms, so each Malayalam word
  additionally emits a subword feature (a fixed-length prefix by default, or
  character n-grams). No English stemming or stop-word removal is applied;
  BM25's IDF already down-weights very common words in both languages.
* Legal identifiers such as "18/ലെഗ്.സി3/2024/നിയമം", "64/Leg.A1/2024/Law" and
  dates like "05-09-2012" are emitted whole AND as their parts, so both exact
  and partial identifier queries match.

Python's \\w does not match Malayalam vowel signs or virama (Unicode Mn/Mc),
so the token pattern lists the Malayalam block explicitly.
"""

import re
import unicodedata
from typing import List

# Bump when normalization or token emission changes; a persisted BM25 index
# built with a different version is rebuilt automatically.
TOKENIZER_VERSION = "ml-norm-v1"

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

_CHAR_MAP = str.maketrans({
    # Atomic chillus -> consonant + virama (legacy form after ZWJ removal)
    "ൺ": "ണ്",  # ൺ
    "ൻ": "ന്",  # ൻ
    "ർ": "ര്",  # ർ
    "ൽ": "ല്",  # ൽ
    "ൾ": "ള്",  # ൾ
    "ൿ": "ക്",  # ൿ
    "ൔ": "മ്",  # ൔ
    "ൕ": "യ്",  # ൕ
    "ൖ": "ഴ്",  # ൖ
    # Old two-part au sign -> modern au length mark
    "ൌ": "ൗ",
    # Malayalam digits -> ASCII
    **{chr(0x0D66 + i): str(i) for i in range(10)},
})

_LEGACY_SEQUENCES = (
    ("ളള", "ള്ള"),  # ളള -> ള്ള
    ("ററ", "റ്റ"),  # ററ -> റ്റ
)

_MALAYALAM = "ഀ-ൿ"
_TOKEN_RE = re.compile(rf"[0-9a-zà-ɏ{_MALAYALAM}]+")
_HAS_MALAYALAM = re.compile(rf"[{_MALAYALAM}]")
_HAS_DIGIT = re.compile(r"\d")
_DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")
_IDENTIFIER_SEPARATOR = re.compile(r"[0-9a-zഀ-ൿ][/.\-:][0-9a-zഀ-ൿ]")
_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>«»“”‘’—–-=_|*©°#~^+/\\"

SUBWORD_MODES = ("none", "prefix", "ngram")
PREFIX_MARK = "~"
NGRAM_MARK = "#"


def normalize(text: str) -> str:
    """Canonical form used for both indexing and querying."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH).translate(_CHAR_MAP)
    for legacy, modern in _LEGACY_SEQUENCES:
        text = text.replace(legacy, modern)
    return text.casefold()


class Tokenizer:
    """Deterministic tokenizer; its `version` string identifies the exact settings."""

    def __init__(self, subword: str = "prefix", prefix_len: int = 5, ngram_n: int = 4):
        if subword not in SUBWORD_MODES:
            raise ValueError(f"subword must be one of {SUBWORD_MODES}")
        self.subword = subword
        self.prefix_len = prefix_len
        self.ngram_n = ngram_n

    @property
    def version(self) -> str:
        detail = {"none": "", "prefix": str(self.prefix_len), "ngram": str(self.ngram_n)}[self.subword]
        return f"{TOKENIZER_VERSION}|{self.subword}{detail}"

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for piece in normalize(text).split():
            piece = _DIGIT_GROUP_COMMA.sub("", piece).strip(_EDGE_PUNCT)
            if not piece:
                continue
            if _HAS_DIGIT.search(piece) and _IDENTIFIER_SEPARATOR.search(piece):
                tokens.append(piece)  # whole identifier, e.g. 18/ലെഗ്.സി3/2024/നിയമം
            for word in _TOKEN_RE.findall(piece):
                if word.isdigit():
                    tokens.append(word.lstrip("0") or "0")
                    continue
                if len(word) < 2:
                    continue  # stray OCR characters / single letters
                tokens.append(word)
                if self.subword != "none" and _HAS_MALAYALAM.search(word):
                    tokens.extend(self._subwords(word))
        return tokens

    def _subwords(self, word: str) -> List[str]:
        if self.subword == "prefix":
            if len(word) >= self.prefix_len:
                return [PREFIX_MARK + word[: self.prefix_len]]
            return []
        n = self.ngram_n
        if len(word) <= n:
            return []
        return [NGRAM_MARK + word[i: i + n] for i in range(len(word) - n + 1)]
