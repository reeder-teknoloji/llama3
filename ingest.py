from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".json",
    ".jsonl",
    ".html",
    ".htm",
    ".pdf",
}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def find_source_files(knowledge_dir: Path) -> Iterable[Path]:
    for path in knowledge_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF okumak icin 'pypdf' gerekli. Kurulum: python -m pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_file_as_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_text(path)

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".json", ".jsonl"}:
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return raw
    if suffix in {".html", ".htm"}:
        # Basic cleanup; keep dependencies minimal.
        no_tags = re.sub(r"<[^>]+>", " ", raw)
        return no_tags
    return raw


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunk_size = max(50, chunk_size)
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step
    return chunks


def build_index(
    knowledge_dir: Optional[Path],
    output_path: Path,
    chunk_size: int,
    overlap: int,
    input_files: Optional[List[Path]] = None,
) -> None:
    chunks: List[Chunk] = []
    files: List[Path] = []
    if input_files:
        files.extend(input_files)
    elif knowledge_dir is not None:
        files.extend(find_source_files(knowledge_dir))

    for source_file in files:
        rel_source = str(source_file.name if knowledge_dir is None else source_file.relative_to(knowledge_dir))
        text = normalize_text(read_file_as_text(source_file))
        if not text:
            continue
        for i, chunk_text in enumerate(split_into_chunks(text, chunk_size, overlap), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{rel_source}:{i}",
                    source=rel_source,
                    text=chunk_text,
                )
            )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_dir": str(knowledge_dir) if knowledge_dir is not None else None,
        "input_files": [str(f) for f in input_files] if input_files else None,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(chunks)} chunks -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local knowledge index for RAG.")
    parser.add_argument("--knowledge_dir", default=None, help="Folder that contains docs.")
    parser.add_argument(
        "--input_files",
        nargs="+",
        default=None,
        help="Optional explicit file paths. If provided, only these files are indexed.",
    )
    parser.add_argument("--output_path", default="knowledge/index.json", help="Output index path.")
    parser.add_argument("--chunk_size", type=int, default=220, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=40, help="Word overlap between chunks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    knowledge_dir = Path(args.knowledge_dir) if args.knowledge_dir else None
    input_files = [Path(p) for p in args.input_files] if args.input_files else None
    if knowledge_dir is None and not input_files:
        raise SystemExit("Hata: --knowledge_dir veya --input_files verilmelidir.")
    build_index(
        knowledge_dir=knowledge_dir,
        output_path=Path(args.output_path),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        input_files=input_files,
    )
