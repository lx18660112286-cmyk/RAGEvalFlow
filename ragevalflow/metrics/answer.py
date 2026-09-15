"""答案质量指标（确定性）。

统一策略：必须/禁止关键词列表为空时，对应比例指标一律返回 0.0（测试中固定）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def compute_answer_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """至少包含 must_include_coverage / forbidden_claim_rate / answer_length / exact_keyword_coverage。"""
    answer = output.answer

    must_list = case.must_include
    must_hits = sum(1 for kw in must_list if kw in answer)
    must_include_coverage = (must_hits / len(must_list)) if must_list else 0.0

    forbid_list = case.must_not_include
    forbid_hits = sum(1 for kw in forbid_list if kw in answer)
    forbidden_claim_rate = (forbid_hits / len(forbid_list)) if forbid_list else 0.0

    return {
        "must_include_coverage": round(must_include_coverage, 6),
        "forbidden_claim_rate": round(forbidden_claim_rate, 6),
        "answer_length": float(len(answer)),
        # 阶段 2 先等同于 must_include_coverage，后续再扩展更精细的关键词匹配
        "exact_keyword_coverage": round(must_include_coverage, 6),
    }