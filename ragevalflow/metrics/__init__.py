"""确定性评测指标包。

所有指标均为纯规则计算，不调用 LLM judge、不联网，结果可复现。
阶段 2 只做指标计算，不做失败归因（阶段 3）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragevalflow.metrics.retrieval import compute_retrieval_metrics
from ragevalflow.metrics.answer import compute_answer_metrics
from ragevalflow.metrics.faithfulness import compute_faithfulness_metrics
from ragevalflow.metrics.trace import compute_trace_metrics
from ragevalflow.metrics.cost import compute_cost_metrics

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def compute_all_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """计算全部确定性指标，返回扁平 dict（各模块 key 无冲突）。

    按 检索 → 答案 → 忠实度 → trace → 成本 顺序合并。
    """
    merged: dict[str, float] = {}
    for compute in (
        compute_retrieval_metrics,
        compute_answer_metrics,
        compute_faithfulness_metrics,
        compute_trace_metrics,
        compute_cost_metrics,
    ):
        merged.update(compute(case, output))
    return merged


__all__ = [
    "compute_retrieval_metrics",
    "compute_answer_metrics",
    "compute_faithfulness_metrics",
    "compute_trace_metrics",
    "compute_cost_metrics",
    "compute_all_metrics",
]