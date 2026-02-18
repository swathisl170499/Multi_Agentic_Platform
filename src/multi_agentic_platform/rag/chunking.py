from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = end - chunk_overlap
    return chunks
