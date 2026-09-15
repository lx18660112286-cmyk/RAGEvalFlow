"""忠实度 / 接地性简单指标（确定性，不依赖 LLM judge）。

- answer_context_overlap_score：answer 与全部 context 文本的字符 2-gram 重合比例；
- citation_doc_coverage：answer 中明确出现（引用）的、且由 contexts 提供的期望文档比例。

空 expected_docs / 空 contexts → 相关指标返回 0.0（测试中固定）。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def _character_ngrams(text: str, n: int = 2) -> set[str]:
    """去空白后的字符 n-gram 集合（对中文/英文均适用）。"""
    cleaned = re.sub(r"\s+", "", text.lower())
    if len(cleaned) <= n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def _answer_context_overlap_score(answer: str, context_texts: list[str]) -> float:
    answer_grams = _character_ngrams(answer)
    if not answer_grams:
        return 0.0
    context_grams: set[str] = set()
    for text in context_texts:
        context_grams |= _character_ngrams(text)
    if not context_grams:
        return 0.0
    hit = sum(1 for g in answer_grams if g in context_grams)
    return hit / len(answer_grams)


def compute_faithfulness_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """至少包含 answer_context_overlap_score / citation_doc_coverage。"""
    answer = output.answer
    contexts = output.contexts

    overlap = _answer_context_overlap_score(answer, [c.text for c in contexts])

    expected = set(case.expected_docs)
    if not expected or not answer:
        citation_coverage = 0.0
    else:
        # 被 contexts 提供、且在 answer 中出现（引用）的期望文档
        provided = {c.doc_id for c in contexts}
        cited = {d for d in expected if d in answer and d in provided}
        citation_coverage = len(cited) / len(expected)

    return {
        "answer_context_overlap_score": round(overlap, 6),
        "citation_doc_coverage": round(citation_coverage, 6),
    }