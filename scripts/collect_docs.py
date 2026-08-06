"""Collect curated official documentation and write a reproducible audit report."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "manifest" / "source_catalog.csv"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_REPORT = ROOT / "data" / "manifest" / "download_report.csv"
DEFAULT_SUMMARY = ROOT / "data" / "manifest" / "collection_summary.json"

REPORT_FIELDS = [
    "document_id",
    "product",
    "topic",
    "source_url",
    "final_url",
    "http_status",
    "content_type",
    "retrieved_at",
    "bytes",
    "sha256",
    "text_sha256",
    "encoding",
    "replacement_char_count",
    "detected_title",
    "exact_duplicate_of",
    "text_duplicate_of",
    "license_id",
    "redistribution_status",
    "suspected_secret_pattern_count",
    "local_path",
    "error",
]

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
]


class VisibleTextParser(HTMLParser):
    """Extract visible text and the first HTML title without external packages."""

    hidden_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self.hidden_tags:
            self.hidden_depth += 1
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self.hidden_tags and self.hidden_depth:
            self.hidden_depth -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if not self.hidden_depth:
            self.text_parts.append(data)


@dataclass
class CollectedDocument:
    """Result of collecting one catalog entry."""

    catalog: dict[str, str]
    result: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"document_id", "product", "source_url", "license_id", "enabled"}
    missing_columns = required - set(rows[0] if rows else {})
    if missing_columns:
        raise ValueError(f"Catalog is missing columns: {sorted(missing_columns)}")

    enabled = [row for row in rows if row["enabled"].strip().lower() == "true"]
    ids = [row["document_id"] for row in enabled]
    urls = [row["source_url"] for row in enabled]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog contains duplicate document_id values")
    if len(urls) != len(set(urls)):
        raise ValueError("Catalog contains duplicate source_url values")
    return enabled


def decode_content(content: bytes, response: httpx.Response) -> tuple[str, str, int]:
    encoding = response.encoding or "utf-8"
    try:
        text = content.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        encoding = "utf-8"
        text = content.decode(encoding, errors="replace")
    return text, encoding, text.count("\ufffd")


def normalize_visible_text(html: str) -> tuple[str, str]:
    parser = VisibleTextParser()
    parser.feed(html)
    text = unicodedata.normalize("NFKC", " ".join(parser.text_parts))
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
    return normalized, title


def count_suspected_secret_patterns(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in SECRET_PATTERNS)


async def fetch_with_retries(
    client: httpx.AsyncClient,
    url: str,
    retries: int,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.get(url)
            if response.status_code < 500:
                return response
            response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


async def collect_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, str],
    raw_dir: Path,
    retries: int,
) -> CollectedDocument:
    collected = CollectedDocument(catalog=row)
    retrieved_at = datetime.now(UTC).isoformat()
    result: dict[str, Any] = {
        "document_id": row["document_id"],
        "product": row["product"],
        "topic": row["topic"],
        "source_url": row["source_url"],
        "retrieved_at": retrieved_at,
        "license_id": row["license_id"],
        "redistribution_status": row["redistribution_status"],
        "error": "",
    }
    try:
        async with semaphore:
            response = await fetch_with_retries(client, row["source_url"], retries)

        result["http_status"] = response.status_code
        result["final_url"] = str(response.url)
        result["content_type"] = response.headers.get("content-type", "")
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
            collected.result = result
            return collected

        content = response.content
        text, encoding, replacement_count = decode_content(content, response)
        normalized_text, detected_title = normalize_visible_text(text)
        sha256 = hashlib.sha256(content).hexdigest()
        text_sha256 = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

        product_dir = raw_dir / row["product"]
        product_dir.mkdir(parents=True, exist_ok=True)
        content_type = result["content_type"].lower()
        suffix = (
            ".html" if "html" in content_type else ".md" if "markdown" in content_type else ".txt"
        )
        output_path = product_dir / f"{row['document_id']}{suffix}"
        output_path.write_bytes(content)

        result.update(
            {
                "bytes": len(content),
                "sha256": sha256,
                "text_sha256": text_sha256,
                "encoding": encoding,
                "replacement_char_count": replacement_count,
                "detected_title": detected_title,
                "suspected_secret_pattern_count": count_suspected_secret_patterns(text),
                "local_path": output_path.relative_to(ROOT).as_posix(),
            }
        )
        if len(content) < 500:
            result["error"] = "Content is unexpectedly small"
    except Exception as exc:  # noqa: BLE001 - report every per-document failure
        result["error"] = f"{type(exc).__name__}: {exc}"
    collected.result = result
    return collected


def mark_duplicates(documents: list[CollectedDocument]) -> None:
    exact_seen: dict[str, str] = {}
    text_seen: dict[str, str] = {}
    for document in documents:
        result = document.result
        result["exact_duplicate_of"] = ""
        result["text_duplicate_of"] = ""
        if not result.get("sha256"):
            continue
        document_id = str(result["document_id"])
        sha256 = str(result["sha256"])
        text_sha256 = str(result["text_sha256"])
        if sha256 in exact_seen:
            result["exact_duplicate_of"] = exact_seen[sha256]
        else:
            exact_seen[sha256] = document_id
        if text_sha256 in text_seen:
            result["text_duplicate_of"] = text_seen[text_sha256]
        else:
            text_seen[text_sha256] = document_id


def write_report(path: Path, documents: list[CollectedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(document.result for document in documents)


def build_summary(documents: list[CollectedDocument]) -> dict[str, Any]:
    results = [document.result for document in documents]
    successful = [row for row in results if row.get("http_status") == 200 and not row["error"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "catalog_entries": len(results),
        "successful": len(successful),
        "failed_or_flagged": len(results) - len(successful),
        "products": dict(sorted(Counter(row["product"] for row in results).items())),
        "http_statuses": dict(
            sorted(Counter(str(row.get("http_status", "error")) for row in results).items())
        ),
        "exact_duplicates": sum(bool(row.get("exact_duplicate_of")) for row in results),
        "text_duplicates": sum(bool(row.get("text_duplicate_of")) for row in results),
        "encoding_replacement_characters": sum(
            int(row.get("replacement_char_count", 0) or 0) for row in results
        ),
        "suspected_secret_pattern_hits": sum(
            int(row.get("suspected_secret_pattern_count", 0) or 0) for row in results
        ),
        "review_required_licenses": sum(
            row.get("license_id") == "REVIEW_REQUIRED" for row in results
        ),
        "failed_document_ids": [row["document_id"] for row in results if row["error"]],
    }


async def async_main(args: argparse.Namespace) -> int:
    catalog = read_catalog(args.catalog)
    headers = {
        "User-Agent": "OpsPilotResearchBot/0.1 (+https://github.com/YEYEPrincess/OpsPilot-)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
    }
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=3)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=timeout,
        limits=limits,
    ) as client:
        tasks = [collect_one(client, semaphore, row, args.raw_dir, args.retries) for row in catalog]
        documents = await asyncio.gather(*tasks)

    mark_duplicates(documents)
    documents.sort(key=lambda document: document.catalog["document_id"])
    write_report(args.report, documents)
    summary = build_summary(documents)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_or_flagged"] == 0 else 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
