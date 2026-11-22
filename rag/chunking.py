"""Paragraph-aware chunking for Wikipedia text, ported from the structural splitter of the earlier RAG project
(sections -> paragraphs -> merged to [min, max] words, the previous paragraph repeated as overlap). Word counts
instead of tokens; the bounds are smaller because the agent's context window is 2048 tokens."""
import re

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
MIN_WORDS, MAX_WORDS = 40, 150


def is_heading(line):
    """wikimedia/wikipedia keeps section titles as short lines without end punctuation."""
    s = line.strip()
    return 0 < len(s.split()) <= 8 and not s.endswith((".", "!", "?", ":", ";", ","))


def _split_long(text, max_words):
    out, buf = [], []
    for s in SENT.split(text):
        if buf and len(" ".join(buf + [s]).split()) > max_words:
            out.append(" ".join(buf)); buf = []
        buf.append(s)
    if buf:
        if out and len(" ".join(buf).split()) < MIN_WORDS // 2:   # a short tail joins the previous piece
            out[-1] += " " + " ".join(buf)
        else:
            out.append(" ".join(buf))
    return out


def chunk_article(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """-> list of (section, chunk_text). Paragraphs are blank-line separated; a heading starts a new section."""
    sections, cur, title = [], [], ""
    for block in re.split(r"\n\s*\n", text or ""):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if len(lines) == 1 and is_heading(lines[0]):
            if cur:
                sections.append((title, cur))
            title, cur = lines[0].strip(), []
            continue
        if is_heading(lines[0]):                       # heading glued to its first paragraph
            if cur:
                sections.append((title, cur))
            title, cur, lines = lines[0].strip(), [], lines[1:]
        cur.append(re.sub(r"\s+", " ", " ".join(lines)).strip())
    if cur:
        sections.append((title, cur))

    chunks = []
    for title, paras in sections:
        buf, prev = [], None
        for p in paras:
            n = len(p.split())
            if n > max_words:                          # overflow: sentence-split, no overlap
                if buf:
                    chunks.append((title, " ".join(buf))); buf = []
                chunks += [(title, c) for c in _split_long(p, max_words)]
                prev = None
            elif len(" ".join(buf).split()) + n < min_words or not buf:
                buf.append(p); prev = p
            else:
                chunks.append((title, " ".join(buf)))
                buf = ([prev] if prev and len(prev.split()) + n <= max_words else []) + [p]
                prev = p
        if buf:
            chunks.append((title, " ".join(buf)))
    return [(t, c) for t, c in chunks if len(c.split()) >= 8]
