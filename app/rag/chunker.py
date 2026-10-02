"""Heading-aware chunking: keeps the markdown section path with every chunk."""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    index: int
    section: str
    text: str


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(h for _, h in stack), body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return sections


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out, cur = [], ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        while len(sentence) > max_chars:  # pathological sentence: hard split
            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if cur and len(cur) + len(sentence) + 1 > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces: list[str] = []
        for p in paragraphs:
            pieces.extend(_split_long(p, max_chars))
        cur = ""
        for piece in pieces:
            if cur and len(cur) + len(piece) + 2 > max_chars:
                chunks.append(Chunk(len(chunks), section, cur))
                tail = cur[-overlap:] if overlap else ""
                if " " in tail:
                    tail = tail.split(" ", 1)[1]
                cur = f"{tail}\n\n{piece}".strip() if tail else piece
            else:
                cur = f"{cur}\n\n{piece}".strip()
        if cur:
            chunks.append(Chunk(len(chunks), section, cur))
    return chunks
