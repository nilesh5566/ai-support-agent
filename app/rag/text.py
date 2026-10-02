"""Shared text utilities (tokenisation used by the hashing embedder and the mock LLM)."""
import re

STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from has have how i if in into is it its
    me my of on or our please so that the their them then there these this to us was we what when
    where which who why will with would you your yours i'm im it's can't cannot don't dont about
    any been get got just like need want hi hello hey thanks thank""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            base = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            if suffix in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiouls":
                base = base[:-1]  # resetting -> reset, shipped -> ship
            return base
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
