"""Tests for Day 10 answerability and citation safeguards."""

from generation.grounding import (
    AnswerabilityPolicy,
    AnswerAction,
    build_clarification,
    validate_sentence_citations,
)


def evidence(score: float = 0.8) -> list[dict[str, object]]:
    return [{"citation_id": "S1", "score": score, "text": "official evidence"}]


def test_answer_when_evidence_is_strong() -> None:
    decision = AnswerabilityPolicy().decide(
        "Docker 容器退出后如何查看日志？", evidence()
    )
    assert decision.action == AnswerAction.ANSWER


def test_clarify_vague_question_before_generation() -> None:
    decision = AnswerabilityPolicy().decide("报错了怎么办", evidence())
    assert decision.action == AnswerAction.CLARIFY
    assert decision.clarification


def test_refuse_when_retrieval_confidence_is_low() -> None:
    decision = AnswerabilityPolicy().decide(
        "未收录框架如何配置未知加速卡？", evidence(0.1)
    )
    assert decision.action == AnswerAction.REFUSE


def test_valid_sentence_level_citations() -> None:
    result = validate_sentence_citations(
        "先使用 docker ps -a 查看状态。[S1] 再读取容器日志。[S1]",
        evidence(),
    )
    assert result.valid


def test_unknown_and_missing_citations_are_rejected() -> None:
    result = validate_sentence_citations(
        "先检查日志。这个结论来自另一个来源。[S9]", evidence()
    )
    assert not result.valid
    assert result.unknown_ids == ("S9",)
    assert result.uncited_claims


def test_clarification_names_missing_context() -> None:
    clarification = build_clarification("服务启动失败")
    assert "版本" in clarification
    assert "日志" in clarification
    assert "运行命令" in clarification
