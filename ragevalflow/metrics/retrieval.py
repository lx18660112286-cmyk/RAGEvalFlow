"""检索质量指标（确定性）。

统一策略：expected_docs 为空时相关指标一律返回 0.0（测试中固定）。
contexts 顺序 = 按 rank 升序排序（rank 全为 0 时保持列表原顺序）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def _ordered_contexts(output: "RAGOutput") -> list:
    """按 rank 升序返回检索结果（rank 相同保持原顺序）。"""
    return sorted(output.contexts, key=lambda c: c.rank)


def compute_retrieval_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """至少包含 hit_at_1/3/5、mrr、expected_doc_coverage、context_precision_simple。"""
    expected = set(case.expected_docs)
    if not expected:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "expected_doc_coverage": 0.0,
            "context_precision_simple": 0.0,
        }

    ordered = _ordered_contexts(output)

    # hit_at_k：top-k 排名内是否出现任一期望文档
    def hit_at(k: int) -> float:
        return 1.0 if any(c.doc_id in expected for c in ordered[:k]) else 0.0

    # mrr：第一个期望文档命中的倒数排名
    mrr = 0.0
    hit_ids: set[str] = set()
    for pos, ctx in enumerate(ordered, start=1):
        if ctx.doc_id in expected:
            if mrr == 0.0:
                mrr = 1.0 / pos
            hit_ids.add(ctx.doc_id)

    expected_doc_coverage = len(hit_ids) / len(expected)
    context_precision_simple = (len(hit_ids) / len(ordered)) if ordered else 0.0

    return {
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "mrr": round(mrr, 6),
        "expected_doc_coverage": round(expected_doc_coverage, 6),
        "context_precision_simple": round(context_precision_simple, 6),
    }