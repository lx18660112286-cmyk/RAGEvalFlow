"""确定性失败归因（阶段 3）。

对单个样例，根据 EvalCase 约束、RAGOutput 输出与已计算的确定性指标，
按规则判定 0..N 个 FailureFinding。所有评判完全确定、可复现：

- 不调用 LLM、不联网、不依赖 Ragas / DeepEval；
- 复用 schemas.failure 中既有的 13 类 FailureType（规则与枚举名不一致处已适配）；
- 约束为空（expected_docs / must_include / must_not_include / expected_tools / forbidden_tools 等）
  时，对应失败一律不触发 —— 空约束不是失败。

FailureType 适配说明（规则 → 既有枚举名）：
- retrieval miss        -> retrieval_miss
- partial retrieval     -> bad_ranking（期望文档未全部命中/排序不佳）
- low context precision -> bad_ranking
- missing must facts    -> incomplete_answer
- low faithfulness      -> incomplete_answer（message 明示"低忠实度/不支持风险"）
- missing rewrite       -> query_rewrite_missing
- forbidden claim       -> forbidden_claim
- forbidden tool        -> forbidden_tool_called
- missing expected tool -> expected_tool_missing
- high latency          -> latency_regression
- high cost             -> cost_regression
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragevalflow.schemas.failure import FailureFinding, FailureType

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput

# 可配置阈值（默认值；analyze_failures 提供同名参数可以覆盖）
DEFAULT_LATENCY_THRESHOLD_MS = 3000.0
DEFAULT_COST_THRESHOLD = 0.02

# 语义近似映射：user 建议规则中未直接对应枚举名的名称 -> 既有 FailureType
_PARTIAL_RETRIEVAL_TYPE: FailureType = "bad_ranking"
_LOW_FAITHFULNESS_TYPE: FailureType = "incomplete_answer"

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def analyze_failures(
    case: "EvalCase",
    output: "RAGOutput",
    metrics: dict[str, float],
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    cost_threshold: float = DEFAULT_COST_THRESHOLD,
) -> list[FailureFinding]:
    """对单个样例做确定性失败归因，返回 0..N 条 FailureFinding。

    同一 failure_type 出现多条判定时合并为一条：保留最高严重度、追加说明。
    """
    findings: dict[FailureType, FailureFinding] = {}

    def _add(
        failure_type: FailureType,
        severity: str,
        reason: str,
        evidence: str,
        metric_values: dict[str, float],
    ) -> None:
        existing = findings.get(failure_type)
        if existing is None:
            findings[failure_type] = FailureFinding(
                case_id=case.id,
                failure_type=failure_type,
                severity=severity,  # type: ignore[arg-type]
                message=reason,
                evidence=evidence,
                metric_values=metric_values,
            )
            return
        # 合并：取更严重级别，说明追加
        if _SEVERITY_RANK[severity] > _SEVERITY_RANK[existing.severity]:
            existing.severity = severity  # type: ignore[assignment]
        existing.message = f"{existing.message}；{reason}"
        existing.evidence = f"{existing.evidence}；{evidence}"
        existing.metric_values.update(metric_values)

    def _m(key: str, default: float = 0.0) -> float:
        return metrics.get(key, default)

    behavior = case.expected_behavior

    # ---- 检索类 ----
    if case.expected_docs:
        hit3 = _m("hit_at_3")
        coverage = _m("expected_doc_coverage")
        if hit3 == 0.0:
            _add(
                "retrieval_miss",
                "high",
                "未检索到任何期望文档",
                f"expected_docs={case.expected_docs} 均未命中（hit_at_3=0）",
                {"hit_at_3": hit3, "expected_doc_coverage": coverage},
            )
        elif coverage < 1.0:
            _add(
                _PARTIAL_RETRIEVAL_TYPE,
                "medium",
                "部分期望文档未被检索命中（partial retrieval）",
                f"expected_doc_coverage={coverage:.4f} < 1.0",
                {"hit_at_3": hit3, "expected_doc_coverage": coverage},
            )

    # 低上下文精度（answer 满足 must_include 时降为 low）
    if output.contexts and _m("context_precision_simple") < 0.5:
        must_ok = _m("must_include_coverage", 1.0) >= 1.0
        _add(
            _PARTIAL_RETRIEVAL_TYPE,
            "low" if must_ok else "medium",
            "检索上下文精度较低" + ("，但已满足 must_include" if must_ok else ""),
            f"context_precision_simple={_m('context_precision_simple'):.4f} < 0.5，contexts={len(output.contexts)} 条",
            {"context_precision_simple": _m("context_precision_simple"), "must_include_coverage": _m("must_include_coverage")},
        )

    # ---- 答案类 ----
    if case.must_include:
        must_cov = _m("must_include_coverage")
        if must_cov < 1.0:
            _add(
                "incomplete_answer",
                "high",
                "答案缺少期望包含的必需内容（must_include 未完全覆盖）",
                f"must_include={case.must_include}，must_include_coverage={must_cov:.4f}",
                {"must_include_coverage": must_cov},
            )

    if case.must_not_include and _m("forbidden_claim_rate") > 0.0:
        _add(
            "forbidden_claim",
            "critical",
            "答案包含禁止出现的内容",
            f"must_not_include={case.must_not_include}，forbidden_claim_rate={_m('forbidden_claim_rate'):.4f}",
            {"forbidden_claim_rate": _m("forbidden_claim_rate")},
        )

    # 低忠实度 / 答案不支持风险（contexts 非空、answer 非空、重叠度过低）
    if (
        output.contexts
        and _m("answer_length") > 0
        and _m("answer_context_overlap_score") < 0.1
    ):
        _add(
            _LOW_FAITHFULNESS_TYPE,
            "medium",
            "答案与检索上下文重叠度过低，存在低忠实度/不支持（幻觉）风险",
            f"answer_context_overlap_score={_m('answer_context_overlap_score'):.4f} < 0.1",
            {"answer_context_overlap_score": _m("answer_context_overlap_score")},
        )

    # ---- Agentic trace 类 ----
    if behavior is not None:
        if behavior.should_rewrite and _m("rewrite_used") == 0.0:
            _add(
                "query_rewrite_missing",
                "medium",
                "应重写查询但系统未进行查询重写",
                f"expected should_rewrite=True，rewrite_used={_m('rewrite_used'):.0f}",
                {"rewrite_used": _m("rewrite_used")},
            )
        if behavior.should_multi_hop and _m("missing_multi_hop") == 1.0:
            _add(
                "missing_multi_hop",
                "high",
                "应为多跳检索但实际未执行多跳",
                f"expected should_multi_hop=True，retrieval_rounds={int(_m('retrieval_rounds'))}",
                {"missing_multi_hop": _m("missing_multi_hop"), "retrieval_rounds": _m("retrieval_rounds")},
            )
        if _m("over_retrieval") == 1.0:
            _add(
                "over_retrieval",
                "medium",
                "检索轮数超过允许上限",
                f"retrieval_rounds={int(_m('retrieval_rounds'))}，max_retrieval_rounds={behavior.max_retrieval_rounds}",
                {"over_retrieval": _m("over_retrieval"), "retrieval_rounds": _m("retrieval_rounds")},
            )
        if behavior.expected_tools and _m("expected_tool_coverage") < 1.0:
            _add(
                "expected_tool_missing",
                "high",
                "期望工具未被调用",
                f"expected_tools={behavior.expected_tools}，expected_tool_coverage={_m('expected_tool_coverage'):.4f}",
                {"expected_tool_coverage": _m("expected_tool_coverage")},
            )
        if _m("forbidden_tool_rate") > 0.0:
            _add(
                "forbidden_tool_called",
                "critical",
                "调用了禁止使用的工具",
                f"forbidden_tools={behavior.forbidden_tools}，forbidden_tool_rate={_m('forbidden_tool_rate'):.4f}",
                {"forbidden_tool_rate": _m("forbidden_tool_rate")},
            )

    # ---- 成本 / 延迟类 ----
    if _m("latency_ms") > latency_threshold_ms:
        _add(
            "latency_regression",
            "medium",
            f"延迟超过阈值 {latency_threshold_ms:g}ms",
            f"latency_ms={_m('latency_ms'):.2f} > {latency_threshold_ms:g}",
            {"latency_ms": _m("latency_ms")},
        )
    if _m("estimated_cost") > cost_threshold:
        _add(
            "cost_regression",
            "medium",
            f"估算成本超过阈值 {cost_threshold:g}",
            f"estimated_cost={_m('estimated_cost'):.4f} > {cost_threshold:g}",
            {"estimated_cost": _m("estimated_cost")},
        )

    return list(findings.values())


__all__ = ["analyze_failures", "DEFAULT_LATENCY_THRESHOLD_MS", "DEFAULT_COST_THRESHOLD"]