"""Minimal multilingual Whisper tokenizer built on tiktoken.

Only what Sajag needs: special-token ids for the decoder prompt
(<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>) and decoding of
generated ids back to text. The BPE ranks file (``multilingual.tiktoken``) is
downloaded once by ``python -m sajag fetch-models`` from the openai/whisper repo.
"""

from __future__ import annotations

import base64
from pathlib import Path

LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi",
    "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy",
    "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
]
TIKTOKEN_URL = (
    "https://raw.githubusercontent.com/openai/whisper/"
    "839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/multilingual.tiktoken"
)


class WhisperTokenizer:
    def __init__(self, ranks_path: str | Path, num_languages: int = 99):
        import tiktoken

        ranks = {}
        for line in Path(ranks_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                tok, rank = line.split()
                ranks[base64.b64decode(tok)] = int(rank)
        n = len(ranks)
        specials = ["<|endoftext|>", "<|startoftranscript|>"]
        specials += [f"<|{lang}|>" for lang in LANGUAGES[:num_languages]]
        specials += ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>",
                     "<|notimestamps|>"]
        specials += [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
        self.special = {tok: n + i for i, tok in enumerate(specials)}
        self.enc = tiktoken.Encoding(
            name="whisper-multilingual",
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens=self.special,
        )
        self.eot = self.special["<|endoftext|>"]
        self.sot = self.special["<|startoftranscript|>"]

    def prompt(self, language: str | None = None, task: str = "transcribe") -> list[int]:
        ids = [self.sot]
        if language:
            ids.append(self.special[f"<|{language}|>"])
        ids.append(self.special[f"<|{task}|>"])
        ids.append(self.special["<|notimestamps|>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode([i for i in ids if i < self.eot]).strip()
