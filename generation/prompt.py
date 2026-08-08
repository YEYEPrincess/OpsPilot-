"""Versioned prompts and context formatting for the basic RAG loop."""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "rag-v1"


def format_context(hits: list[dict[str, Any]]) -> str:
    """Format retrieved payloads as numbered, traceable evidence blocks."""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        source = hit.get("source_url", "unknown")
        section = " > ".join(hit.get("section_path", [])) or "未标注章节"
        page = hit.get("page") or "未标注页码"
        blocks.append(
            f"[S{index}] score={hit.get('score', 0):.4f}\n"
            f"source={source}\nsection={section}\npage={page}\n"
            f"text={hit.get('text', '')}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    """Build a strict JSON-output prompt for an OpenAI-compatible model."""
    context = format_context(hits)
    return f"""你是 OpsPilot 故障诊断助手。只能依据下面的证据回答，不要编造证据。
如果证据不足，answer 中明确说明“证据不足”，并在 clarification 中提出需要补充的信息。
输出必须是合法 JSON，字段不能缺失：
{{"answer":"...","possible_causes":["..."],"steps":["..."],"risks":["..."],"citations":["S1"],"clarification":""}}

用户问题：{question}

证据：
{context}
"""
