"""baseline 与 candidate 实验的确定性回归检测（阶段 3）。

- 只比较共同 case_id；
- baseline 独有 case -> missing_candidate_case；candidate 独有 case -> new_candidate_case；
- 对共同 case 逐指标按方向与阈值判定 improved / regressed / unchanged；
- 指标默认方向与默认阈值遵循阶段 3 规格（可通过 RegressionThresholds 覆盖）。

不调用 LLM、不联网、结果可复现。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragevalflow.schemas.experiment_result import CaseResult

# 指标默认方向
HIGHER_IS_BETTER = {
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr",
    "expected_doc_coverage",
    "context_precision_simple",
    "must_include_coverage",
    "exact_keyword_coverage",
    "answer_context_overlap_score",
    "citation_doc_coverage",
    "expected_tool_coverage",
}
LOWER_IS_BETTER = {
    "forbidden_claim_rate",
    "latency_ms",
    "estimated_cost",
    "total_tokens",
    "over_retrieval",
    "missing_multi_hop",
    "forbidden_tool_rate",
}
METRIC_DIRECTIONS: dict[str, Literal["higher_is_better", "lower_is_better"]] = {
    name: "higher_is_better" for name in HIGHER_IS_BETTER
}
METRIC_DIRECTIONS.update({name: "lower_is_better" for name in LOWER_IS_BETTER})

_EPS = 1e-9


class ComparisonError(Exception):
    """对比前置条件不满足（如 baseline/candidate 无共同 case_id）。"""


class RegressionThresholds(BaseModel):
    """回归判定阈值（均为默认值，可覆盖）。"""

    absolute_drop: float = Field(default=0.05, description="绝对下降阈值（0~1 比例类指标）")
    relative_drop: float = Field(default=0.10, description="相对下降阈值（10%）")
    latency_increase: float = Field(default=0.20, description="延迟增幅阈值（20%）")
    cost_increase: float = Field(default=0.20, description="成本增幅阈值（20%）")


class MetricDelta(BaseModel):
    """单个指标在共同 case 上的平均变化。"""

    metric_name: str
    baseline_avg: float
    candidate_avg: float
    delta: float
    direction: Literal["higher_is_better", "lower_is_better"]
    status: Literal["improved", "regressed", "unchanged"]


class CaseDelta(BaseModel):
    """单个共同 case 的变化。"""

    case_id: str
    status: Literal["improved", "regressed", "unchanged", "missing_candidate_case", "new_candidate_case"]
    changed_metrics: dict[str, float] = Field(default_factory=dict, description="metric -> delta(candidate-baseline)")
    new_failure_types: list[str] = Field(default_factory=list, description="candidate 新增的失败类型")
    resolved_failure_types: list[str] = Field(default_factory=list, description="candidate 修复的失败类型")


class RegressionSummary(BaseModel):
    """对比汇总。"""

    baseline_experiment: str
    candidate_experiment: str
    common_cases: int
    improved_cases: int = 0
    regressed_cases: int = 0
    unchanged_cases: int = 0
    overall_status: Literal["improved", "regressed", "mixed", "unchanged"] = "unchanged"


class RegressionReport(BaseModel):
    """完整对比结果。"""

    summary: RegressionSummary
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    case_deltas: list[CaseDelta] = Field(default_factory=list)


def _metric_status(
    name: str, baseline: float, candidate: float, thresholds: RegressionThresholds
) -> Literal["improved", "regressed", "unchanged"] | None:
    """单指标判定。无默认方向（未在方向表内）返回 None。"""
    direction = METRIC_DIRECTIONS.get(name)
    if direction is None:
        return None
    delta = candidate - baseline
    if abs(delta) < _EPS:
        return "unchanged"

    if direction == "higher_is_better":
        if delta > 0:
            rel = delta / baseline if baseline > _EPS else 1.0
            return "improved" if delta >= thresholds.absolute_drop or rel >= thresholds.relative_drop else "unchanged"
        drop = -delta
        rel = drop / baseline if baseline > _EPS else 0.0
        return "regressed" if drop >= thresholds.absolute_drop or rel >= thresholds.relative_drop else "unchanged"

    # lower_is_better：candidate 更大更差；增幅阈值对 latency/cost 单独配置
    inc_ratio = (
        thresholds.latency_increase
        if name == "latency_ms"
        else (thresholds.cost_increase if name == "estimated_cost" else thresholds.relative_drop)
    )
    if delta > 0:
        rel = delta / baseline if baseline > _EPS else 1.0
        worsened = rel >= inc_ratio if baseline > _EPS else delta >= thresholds.absolute_drop
        return "regressed" if worsened else "unchanged"
    gain = -delta
    rel = gain / baseline if baseline > _EPS else 0.0
    return "improved" if rel >= thresholds.relative_drop or gain >= thresholds.absolute_drop else "unchanged"


def compare_experiments(
    baseline_results: list[CaseResult],
    candidate_results: list[CaseResult],
    thresholds: RegressionThresholds | None = None,
) -> RegressionReport:
    """对比 baseline 与 candidate，返回 RegressionReport。"""
    thresholds = thresholds or RegressionThresholds()

    base_map = {r.case_id: r for r in baseline_results}
    cand_map = {r.case_id: r for r in candidate_results}
    common_ids = sorted(set(base_map) & set(cand_map))
    if not common_ids:
        raise ComparisonError("baseline 与 candidate 实验没有共同 case_id，无法对比")

    case_deltas: list[CaseDelta] = []
    for case_id in common_ids:
        base, cand = base_map[case_id], cand_map[case_id]
        changed: dict[str, float] = {}
        for metric in METRIC_DIRECTIONS:
            b, c = base.metrics.get(metric, 0.0), cand.metrics.get(metric, 0.0)
            if abs(c - b) >= _EPS:
                changed[metric] = round(c - b, 6)
        per_metric = {
            metric: _metric_status(metric, base.metrics.get(metric, 0.0), cand.metrics.get(metric, 0.0), thresholds)
            for metric in changed
        }
        regressed_metrics = [m for m, s in per_metric.items() if s == "regressed"]
        improved_metrics = [m for m, s in per_metric.items() if s == "improved"]

        base_types = set(base.failure_types)
        cand_types = set(cand.failure_types)
        new_failures = sorted(cand_types - base_types)
        resolved_failures = sorted(base_types - cand_types)

        if new_failures or regressed_metrics:
            status: str = "regressed"
        elif resolved_failures or improved_metrics:
            status = "improved"
        else:
            status = "unchanged"
        case_deltas.append(
            CaseDelta(
                case_id=case_id,
                status=status,  # type: ignore[arg-type]
                changed_metrics=changed,
                new_failure_types=new_failures,
                resolved_failure_types=resolved_failures,
            )
        )

    # 仅 baseline 有 / 仅 candidate 有
    for case_id in sorted(set(base_map) - set(cand_map)):
        case_deltas.append(
            CaseDelta(
                case_id=case_id,
                status="missing_candidate_case",
                new_failure_types=[],
                resolved_failure_types=[],
            )
        )
    for case_id in sorted(set(cand_map) - set(base_map)):
        case_deltas.append(
            CaseDelta(
                case_id=case_id,
                status="new_candidate_case",
                new_failure_types=sorted(set(cand_map[case_id].failure_types)),
                resolved_failure_types=[],
            )
        )

    # 指标平均变化（仅共同 case 参与聚合）
    metric_deltas: list[MetricDelta] = []
    all_results = [*baseline_results, *candidate_results]
    metric_names = [m for m in METRIC_DIRECTIONS if any(m in r.metrics for r in all_results)]
    for metric in metric_names:
        base_vals = [base_map[c].metrics.get(metric, 0.0) for c in common_ids]
        cand_vals = [cand_map[c].metrics.get(metric, 0.0) for c in common_ids]
        b_avg = sum(base_vals) / len(common_ids)
        c_avg = sum(cand_vals) / len(common_ids)
        status = _metric_status(metric, b_avg, c_avg, thresholds)
        if status is None:
            continue
        metric_deltas.append(
            MetricDelta(
                metric_name=metric,
                baseline_avg=round(b_avg, 6),
                candidate_avg=round(c_avg, 6),
                delta=round(c_avg - b_avg, 6),
                direction=METRIC_DIRECTIONS[metric],
                status=status,
            )
        )

    improved = sum(1 for d in case_deltas if d.status == "improved")
    regressed = sum(1 for d in case_deltas if d.status == "regressed")
    unchanged = sum(1 for d in case_deltas if d.status == "unchanged")
    if regressed == 0 and improved > 0:
        overall: str = "improved"
    elif improved == 0 and regressed > 0:
        overall = "regressed"
    elif improved == 0 and regressed == 0:
        overall = "unchanged"
    else:
        overall = "mixed"

    baseline_exp = baseline_results[0].experiment_id if baseline_results else ""
    candidate_exp = candidate_results[0].experiment_id if candidate_results else ""

    return RegressionReport(
        summary=RegressionSummary(
            baseline_experiment=baseline_exp,
            candidate_experiment=candidate_exp,
            common_cases=len(common_ids),
            improved_cases=improved,
            regressed_cases=regressed,
            unchanged_cases=unchanged,
            overall_status=overall,  # type: ignore[arg-type]
        ),
        metric_deltas=metric_deltas,
        case_deltas=case_deltas,
    )


__all__ = [
    "compare_experiments",
    "RegressionReport",
    "RegressionThresholds",
    "ComparisonError",
    "METRIC_DIRECTIONS",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
]