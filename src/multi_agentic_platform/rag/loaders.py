from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF ingestion") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, ensure_ascii=False, indent=2)


def _load_csv(path: Path) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(lines)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_document(path_str: str) -> tuple[str, str]:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content = _load_pdf(path)
    elif suffix == ".json":
        content = _load_json(path)
    elif suffix == ".csv":
        content = _load_csv(path)
    else:
        content = _load_text(path)

    return str(path), content
