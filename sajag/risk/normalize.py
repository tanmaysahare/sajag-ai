"""Text normalisation for mixed English / Hindi / Hinglish input.

ASR and OCR output is noisy: casing, punctuation, stray symbols and
Devanagari nukta variants all differ between engines. We normalise before
pattern matching and before hashing features for the classifier.
"""

from __future__ import annotations

import re
import unicodedata

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_PUNCT = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)
_SPACES = re.compile(r"\s+")

# Common OCR / ASR confusions and spelling variants seen in scam scripts.
_REPLACEMENTS = {
    "o.t.p": "otp",
    "o t p": "otp",
    "any desk": "anydesk",
    "team viewer": "teamviewer",
    "k.y.c": "kyc",
    "c.b.i": "cbi",
    "e.d.": "ed",
    "adhar": "aadhaar",
    "aadhar": "aadhaar",
    "adhaar": "aadhaar",
    "whatsapp's": "whatsapp",
    "‍": "",  # zero-width joiner
    "‌": "",  # zero-width non-joiner
}


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def normalize(text: str) -> str:
    """Lower-case, NFKC-normalise, drop punctuation, collapse whitespace."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    for src, dst in _REPLACEMENTS.items():
        t = t.replace(src, dst)
    t = _PUNCT.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    return t


def phrase_in(phrase: str, text_norm: str) -> bool:
    """Match a normalised phrase inside normalised text.

    Latin-script phrases must sit on word boundaries (so "ed" does not match
    "need"); Devanagari phrases are matched as substrings because Hindi
    postpositions often attach without spaces in ASR output.
    """
    p = normalize(phrase)
    if not p:
        return False
    if has_devanagari(p):
        return p in text_norm
    return re.search(r"(?<![\w])" + re.escape(p) + r"(?![\w])", text_norm) is not None
