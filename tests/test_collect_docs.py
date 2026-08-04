"""Tests for the documentation collection and governance helpers."""

from pathlib import Path

from scripts.collect_docs import normalize_visible_text, read_catalog


def test_catalog_has_curated_unique_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = read_catalog(root / "data" / "manifest" / "source_catalog.csv")
    assert 30 <= len(rows) <= 100
    assert len({row["document_id"] for row in rows}) == len(rows)
    assert len({row["source_url"] for row in rows}) == len(rows)
    assert all(row["source_url"].startswith("https://") for row in rows)


def test_visible_text_normalization_ignores_scripts() -> None:
    html = "<html><title> Demo </title><body>Hello   World<script>secret()</script></body></html>"
    text, title = normalize_visible_text(html)
    assert text == "demo hello world"
    assert title == "Demo"
