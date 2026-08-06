"""Unit tests for normalization and chunk metadata."""

from scripts.parse_docs import clean_text, fixed_chunks, section_chunks


def test_clean_text_normalizes_unicode_and_whitespace() -> None:
    assert clean_text("Ａ  B\n\n C\u200b") == "A B\nC"


def test_chunk_strategies_keep_source_and_section_metadata() -> None:
    document = type(
        "Document",
        (),
        {
            "metadata": {
                "document_id": "demo-001",
                "product": "demo",
                "source_url": "https://example.com/demo",
                "source_sha256": "source-hash",
            },
            "sections": [
                {"heading_path": ["A"], "page": None, "text": "alpha " * 100},
                {"heading_path": ["B"], "page": None, "text": "beta " * 100},
            ],
        },
    )()
    fixed = fixed_chunks(document, target=100, overlap=20)
    section = section_chunks(document, target=100, overlap=20)
    assert fixed and section
    assert all(chunk["source_sha256"] == "source-hash" for chunk in fixed + section)
    assert all(chunk["text_sha256"] for chunk in fixed + section)
    assert any(chunk["section_path"] == ["A"] for chunk in section)
    assert any(chunk["section_path"] == ["B"] for chunk in section)
