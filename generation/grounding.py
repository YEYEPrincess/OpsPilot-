"""Day 10 grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CITATION_RE = re.compile(r"\[(S\d+)]")
SENTENCE_RE = re.compile(
    r"[^。！？!?\n]+(?:[。！？!?](?:\s*\[S\d+])*)?"
)


class AnswerAction(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"


@dataclass(frozen=True)
class AnswerabilityDecision:
    action: AnswerAction
    reason: str
    top_score: float
    evidence_count: int
    clarification: str = ""


@dataclass(frozen=True)
class CitationValidation:
    """Structural citation check result."""

    valid: bool
    cited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    uncited_claims: tuple[str, ...]


@dataclass
class AnswerabilityPolicy:
    """Conservative rules executed before calling the LLM.

    Score scales differ across cosine, BM25 and reranker models. The threshold
    is configurable and must be calibrated on a validation split.
    """

    min_top_score: float = 0.35
    min_evidence_count: int = 1
    vague_question_max_chars: int = 12

    _vague_markers = (
        "怎么办",
        "怎么回事",
        "有问题",
        "报错了",
        "不能用",
        "帮我看看",
    )

    def decide(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> AnswerabilityDecision:
        """Choose answer, clarification or refusal without hidden labels."""
        clean_question = question.strip()
        if not clean_question:
            return AnswerabilityDecision(
                AnswerAction.CLARIFY,
                "empty_question",
                0.0,
                0,
                "请提供错误现象、运行命令、软件版本和关键日志。",
            )
        if len(clean_question) <= self.vague_question_max_chars and any(
            marker in clean_question for marker in self._vague_markers
        ):
            return AnswerabilityDecision(
                AnswerAction.CLARIFY,
                "question_is_underspecified",
                _top_score(evidence),
                len(evidence),
                build_clarification(clean_question),
            )
        top_score = _top_score(evidence)
        if len(evidence) < self.min_evidence_count:
            return AnswerabilityDecision(
                AnswerAction.REFUSE,
                "no_evidence",
                top_score,
                len(evidence),
                build_clarification(clean_question),
            )
        if top_score < self.min_top_score:
            return AnswerabilityDecision(
                AnswerAction.REFUSE,
                "retrieval_confidence_below_threshold",
                top_score,
                len(evidence),
                build_clarification(clean_question),
            )
        return AnswerabilityDecision(
            AnswerAction.ANSWER,
            "sufficient_retrieval_evidence",
            top_score,
            len(evidence),
        )


def _top_score(evidence: list[dict[str, Any]]) -> float:
    return max((float(item.get("score", 0.0)) for item in evidence), default=0.0)


def build_clarification(question: str) -> str:
    """Generate a deterministic follow-up without another LLM call."""
    missing: list[str] = []
    lowered = question.lower()
    if not re.search(r"\d+\.\d+|version|版本", lowered):
        missing.append("软件与 CUDA/驱动版本")
    if not re.search(r"error|exception|traceback|报错|日志", lowered):
        missing.append("完整错误日志")
    if not re.search(r"docker|python|pip|uv|curl|命令|运行", lowered):
        missing.append("实际运行命令")
    if not missing:
        missing.append("可复现步骤和期望结果")
    return "为了可靠诊断，请补充：" + "、".join(missing) + "。"


def validate_sentence_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> CitationValidation:
    """Require every factual sentence to cite one returned source ID.

    This checks structure, not semantic entailment. A heavier NLI verifier may
    later check whether each cited source actually supports its claim.
    """
    available = {
        str(source.get("citation_id", f"S{index}"))
        for index, source in enumerate(sources, start=1)
    }
    cited = set(CITATION_RE.findall(answer))
    unknown = cited - available
    uncited: list[str] = []
    for match in SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        if not sentence or _is_non_factual(sentence):
            continue
        if not CITATION_RE.search(sentence):
            uncited.append(sentence)
    return CitationValidation(
        valid=not unknown and not uncited and bool(cited),
        cited_ids=tuple(sorted(cited)),
        unknown_ids=tuple(sorted(unknown)),
        uncited_claims=tuple(uncited),
    )


def _is_non_factual(sentence: str) -> bool:
    stripped = sentence.strip(" ：:")
    if len(stripped) <= 3:
        return True
    return stripped.startswith(("证据不足", "请补充", "需要确认", "无法判断"))


def guarded_response(
    question: str,
    evidence: list[dict[str, Any]],
    policy: AnswerabilityPolicy,
) -> dict[str, Any]:
    """Create a safe response when generation should not run."""
    decision = policy.decide(question, evidence)
    if decision.action == AnswerAction.ANSWER:
        raise ValueError("guarded_response is only for clarify/refuse decisions")
    message = (
        "当前问题信息不足，暂不执行故障结论生成。"
        if decision.action == AnswerAction.CLARIFY
        else "当前知识库证据不足，无法给出可靠结论。"
    )
    return {
        "status": decision.action.value,
        "answer": message,
        "claims": [],
        "citations": [],
        "clarification": decision.clarification,
        "guard": {
            "reason": decision.reason,
            "top_score": decision.top_score,
            "evidence_count": decision.evidence_count,
        },
        "sources": evidence,
    }
