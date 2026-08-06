"""Parse raw documentation and produce normalized documents and chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "manifest" / "source_catalog.csv"
DEFAULT_REPORT = ROOT / "data" / "manifest" / "download_report.csv"
DEFAULT_DOCUMENTS = ROOT / "data" / "processed" / "documents.jsonl"
DEFAULT_FIXED = ROOT / "data" / "processed" / "chunks_fixed.jsonl"
DEFAULT_SECTION = ROOT / "data" / "processed" / "chunks_section.jsonl"
DEFAULT_STATS = ROOT / "data" / "manifest" / "parse_stats.json"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
BLOCK_TAGS = {"p", "li", "pre", "blockquote", "dt", "dd", "td", "th"}
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "svg", "noscript"}


@dataclass
class ParsedDocument:
    """A normalized document and its ordered sections."""

    metadata: dict[str, Any]
    sections: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--fixed-output", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--section-output", type=Path, default=DEFAULT_SECTION)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--chunk-chars", type=int, default=1600)
    parser.add_argument("--overlap-chars", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Normalize Unicode, remove controls, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = "".join(
        char for char in text if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", clean_text(text)).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def append_block(
    groups: list[dict[str, Any]], heading_path: list[str], text: str, page: int | None
) -> None:
    cleaned = clean_text(text)
    if len(cleaned) < 2:
        return
    key = (tuple(heading_path), page)
    if groups and (tuple(groups[-1]["heading_path"]), groups[-1]["page"]) == key:
        groups[-1]["text"] = f"{groups[-1]['text']}\n\n{cleaned}"
        return
    groups.append({"heading_path": list(heading_path), "page": page, "text": cleaned})


def parse_html(raw: bytes) -> tuple[str, list[dict[str, Any]]]:
    soup = BeautifulSoup(raw, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()
    root = soup.body or soup
    groups: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    for element in root.find_all(list(BLOCK_TAGS) + [f"h{i}" for i in range(1, 7)]):
        if element.name and element.name.startswith("h") and element.name[1:].isdigit():
            level = int(element.name[1:])
            heading = clean_text(element.get_text(" ", strip=True))
            if not heading:
                continue
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            continue
        append_block(groups, heading_stack, element.get_text(" ", strip=True), None)
    return title, groups


def parse_markdown(raw: bytes) -> tuple[str, list[dict[str, Any]]]:
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    title = ""
    groups: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    block: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal block
        if block:
            append_block(groups, heading_stack, "\n".join(block), None)
            block = []

    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            block.append(line)
            continue
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = clean_text(match.group(2))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            if not title and level == 1:
                title = heading
        elif line.strip():
            block.append(line)
        else:
            flush()
    flush()
    return title, groups


def parse_pdf(path: Path) -> tuple[str, list[dict[str, Any]]]:
    reader = PdfReader(str(path))
    groups: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        append_block(groups, [f"page {page_number}"], page.extract_text() or "", page_number)
    return path.stem, groups


def deduplicate_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        digest = text_hash(section["text"])
        if not digest or digest in seen:
            continue
        seen.add(digest)
        unique.append(
            {
                "section_id": f"section-{index:04d}",
                "heading_path": section["heading_path"],
                "page": section["page"],
                "text": section["text"],
                "text_sha256": digest,
            }
        )
    return unique


def load_rows(catalog_path: Path, report_path: Path) -> list[dict[str, str]]:
    with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        catalog = {row["document_id"]: row for row in csv.DictReader(handle)}
    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        report = list(csv.DictReader(handle))
    rows: list[dict[str, str]] = []
    for item in report:
        row = {**catalog[item["document_id"]], **item}
        if item.get("error"):
            continue
        rows.append(row)
    return rows


def parse_one(row: dict[str, str]) -> ParsedDocument:
    local_path = ROOT / row["local_path"]
    raw = local_path.read_bytes()
    suffix = local_path.suffix.lower()
    if suffix == ".pdf":
        detected_title, raw_sections = parse_pdf(local_path)
    elif suffix in {".md", ".markdown"}:
        detected_title, raw_sections = parse_markdown(raw)
    elif suffix in {".html", ".htm", ".txt"}:
        detected_title, raw_sections = (
            parse_html(raw)
            if suffix != ".txt"
            else (
                "",
                [{"heading_path": [], "page": None, "text": raw.decode("utf-8", errors="replace")}],
            )
        )
    else:
        raise ValueError(f"Unsupported document extension: {suffix}")

    sections = deduplicate_sections(raw_sections)
    full_text = "\n\n".join(section["text"] for section in sections)
    metadata = {
        "document_id": row["document_id"],
        "product": row["product"],
        "topic": row["topic"],
        "title": detected_title or row["title"],
        "source_url": row["source_url"],
        "source_kind": row["source_kind"],
        "language": row["language"],
        "license_id": row["license_id"],
        "retrieved_at": row.get("retrieved_at", ""),
        "source_sha256": row.get("sha256", ""),
        "content_sha256": text_hash(full_text),
        "source_path": row["local_path"],
        "char_count": len(full_text),
        "section_count": len(sections),
        "parse_status": "ok" if sections else "empty",
    }
    return ParsedDocument(metadata=metadata, sections=sections)


def document_record(document: ParsedDocument) -> dict[str, Any]:
    return {**document.metadata, "sections": document.sections}


def section_segments(document: ParsedDocument) -> tuple[str, list[tuple[int, int, dict[str, Any]]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for section in document.sections:
        heading = " > ".join(section["heading_path"])
        segment = f"{heading}\n{section['text']}" if heading else section["text"]
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(segment)
        cursor += len(segment)
        spans.append((start, cursor, section))
    return "".join(parts), spans


def overlapping_sections(
    spans: list[tuple[int, int, dict[str, Any]]], start: int, end: int
) -> list[dict[str, Any]]:
    return [section for left, right, section in spans if left < end and right > start]


def make_chunk(
    document: ParsedDocument,
    strategy: str,
    index: int,
    text: str,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    first = sections[0] if sections else {"heading_path": [], "page": None}
    return {
        "chunk_id": f"{document.metadata['document_id']}:{strategy}:{index:04d}",
        "document_id": document.metadata["document_id"],
        "product": document.metadata["product"],
        "source_url": document.metadata["source_url"],
        "source_sha256": document.metadata["source_sha256"],
        "section_path": first["heading_path"],
        "section_paths": [section["heading_path"] for section in sections],
        "page": first["page"],
        "pages": sorted({section["page"] for section in sections if section["page"] is not None}),
        "strategy": strategy,
        "chunk_index": index,
        "text": clean_text(text),
        "char_count": len(clean_text(text)),
        "text_sha256": text_hash(text),
    }


def fixed_chunks(document: ParsedDocument, target: int, overlap: int) -> list[dict[str, Any]]:
    text, spans = section_segments(document)
    if not text:
        return []
    step = max(1, target - overlap)
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(text), step)):
        end = min(start + target, len(text))
        chunk = make_chunk(
            document, "fixed", index, text[start:end], overlapping_sections(spans, start, end)
        )
        if chunk["text"]:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def section_chunks(document: ParsedDocument, target: int, overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = max(1, target - overlap)
    index = 0
    for section in document.sections:
        text = section["text"]
        for start in range(0, len(text), step):
            end = min(start + target, len(text))
            chunks.append(make_chunk(document, "section", index, text[start:end], [section]))
            index += 1
            if end >= len(text):
                break
    return [chunk for chunk in chunks if chunk["text"]]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def summarize(
    documents: list[ParsedDocument],
    fixed: list[dict[str, Any]],
    section: list[dict[str, Any]],
    failures: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lengths = [chunk["char_count"] for chunk in fixed + section]
    by_product: dict[str, dict[str, int]] = {}
    for document in documents:
        product = document.metadata["product"]
        stats = by_product.setdefault(
            product, {"documents": 0, "sections": 0, "fixed_chunks": 0, "section_chunks": 0}
        )
        stats["documents"] += 1
        stats["sections"] += document.metadata["section_count"]
        stats["fixed_chunks"] += sum(
            1 for chunk in fixed if chunk["document_id"] == document.metadata["document_id"]
        )
        stats["section_chunks"] += sum(
            1 for chunk in section if chunk["document_id"] == document.metadata["document_id"]
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "chunk_chars": args.chunk_chars,
        "overlap_chars": args.overlap_chars,
        "documents_ok": len(documents),
        "documents_failed": len(failures),
        "sections": sum(document.metadata["section_count"] for document in documents),
        "fixed_chunks": len(fixed),
        "section_chunks": len(section),
        "chunk_char_min": min(lengths) if lengths else 0,
        "chunk_char_max": max(lengths) if lengths else 0,
        "chunk_char_median": statistics.median(lengths) if lengths else 0,
        "products": dict(sorted(by_product.items())),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    rows = load_rows(args.catalog, args.report)
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.documents.exists():
        with args.documents.open(encoding="utf-8") as handle:
            existing = {
                record["document_id"]: record for line in handle if (record := json.loads(line))
            }

    documents: list[ParsedDocument] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        try:
            parsed = parse_one(row)
            if args.resume and row["document_id"] in existing:
                cached = existing[row["document_id"]]
                if cached.get("source_sha256") == parsed.metadata["source_sha256"]:
                    parsed = ParsedDocument(
                        metadata={key: cached[key] for key in parsed.metadata},
                        sections=cached["sections"],
                    )
            documents.append(parsed)
        except Exception as exc:  # noqa: BLE001 - record per-document failures
            failures.append(
                {
                    "document_id": row["document_id"],
                    "source_path": row.get("local_path", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    documents.sort(key=lambda document: document.metadata["document_id"])
    fixed = [
        chunk
        for document in documents
        for chunk in fixed_chunks(document, args.chunk_chars, args.overlap_chars)
    ]
    section = [
        chunk
        for document in documents
        for chunk in section_chunks(document, args.chunk_chars, args.overlap_chars)
    ]
    write_jsonl(args.documents, [document_record(document) for document in documents])
    write_jsonl(args.fixed_output, fixed)
    write_jsonl(args.section_output, section)
    stats = summarize(documents, fixed, section, failures, args)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
