"""Agentic-RAG trace 指标（确定性）。

规则：
- expected_behavior.should_multi_hop=True 且 retrieval_rounds<=1 → missing_multi_hop=1.0；
- expected_behavior.max_retrieval_rounds 非空且实际轮数超限 → over_retrieval=1.0；
- expected_tools 非空 → expected_tool_coverage = tools_used 覆盖比例；
- forbidden_tools 中任意工具出现在 tools_used 或 trace.steps 的 tool 字段 → forbidden_tool_rate>0。

空约束列表 → 对应比例指标返回 0.0（测试中固定）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def _tools_from_steps(steps: list[dict]) -> set[str]:
    """从 trace.steps 中提取所有 tool 名称。"""
    tools: set[str] = set()
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("tool"), str):
            tools.add(step["tool"])
    return tools


def compute_trace_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """至少包含 retrieval_rounds / rewrite_used / reranker_used / reflection_used /
    over_retrieval / missing_multi_hop / expected_tool_coverage / forbidden_tool_rate。"""
    trace = output.trace
    behavior = case.expected_behavior

    # 覆盖率：tools_used 中命中 expected_tools 的比例
    expected_tools = list(behavior.expected_tools) if behavior else []
    used_tools = set(trace.tools_used)
    expected_hits = sum(1 for t in expected_tools if t in used_tools)
    expected_tool_coverage = (expected_hits / len(expected_tools)) if expected_tools else 0.0

    # 禁止工具比例：tools_used 与 steps 中出现的 forbidden_tools 比例
    forbidden_tools = list(behavior.forbidden_tools) if behavior else []
    used_with_steps = used_tools | _tools_from_steps(trace.steps)
    forbidden_hits = sum(1 for t in forbidden_tools if t in used_with_steps)
    forbidden_tool_rate = (forbidden_hits / len(forbidden_tools)) if forbidden_tools else 0.0

    return {
        "retrieval_rounds": float(trace.retrieval_rounds),
        "rewrite_used": 1.0 if trace.query_rewrite else 0.0,
        "reranker_used": 1.0 if trace.used_reranker else 0.0,
        "reflection_used": 1.0 if trace.used_reflection else 0.0,
        "over_retrieval": _over_retrieval(behavior, trace.retrieval_rounds),
        "missing_multi_hop": _missing_multi_hop(behavior, trace.retrieval_rounds),
        "expected_tool_coverage": round(expected_tool_coverage, 6),
        "forbidden_tool_rate": round(forbidden_tool_rate, 6),
    }


def _over_retrieval(behavior, actual_rounds: int) -> float:
    if behavior is None or behavior.max_retrieval_rounds is None:
        return 0.0
    return 1.0 if actual_rounds > behavior.max_retrieval_rounds else 0.0


def _missing_multi_hop(behavior, actual_rounds: int) -> float:
    if behavior is None or not behavior.should_multi_hop:
        return 0.0
    return 1.0 if actual_rounds <= 1 else 0.0