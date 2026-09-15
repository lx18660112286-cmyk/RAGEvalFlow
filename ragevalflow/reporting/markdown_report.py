"""Markdown 报告生成器（阶段 3）。

- generate_experiment_report：单实验报告（Summary / 聚合指标 / 失败汇总 / Case 表 / 建议 / 限制）；
- generate_comparison_report：baseline vs candidate 对比报告。

全部基于 deterministic 数据（指标 + 失败归因），不含 LLM judge 结论。
空约束说明：约束列表为空时对应比例为 0.0 代表"无约束"，报告以 N/A 注明，不视为失败。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragevalflow.analysis.regression import METRIC_DIRECTIONS, RegressionReport

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.experiment_result import CaseResult

# 聚合指标分区（key -> 中文标签）
METRIC_GROUPS: dict[str, list[str]] = {
    "检索指标 (retrieval)": ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "expected_doc_coverage", "context_precision_simple"],
    "答案指标 (answer)": ["must_include_coverage", "forbidden_claim_rate", "answer_length", "exact_keyword_coverage"],
    "忠实度指标 (faithfulness)": ["answer_context_overlap_score", "citation_doc_coverage"],
    "Trace 指标 (agentic)": ["retrieval_rounds", "rewrite_used", "reranker_used", "reflection_used", "over_retrieval", "missing_multi_hop", "expected_tool_coverage", "forbidden_tool_rate"],
    "成本/运行时指标 (cost)": ["latency_ms", "input_tokens", "output_tokens", "estimated_cost", "total_tokens"],
}

# Case 表的关键指标列
CASE_TABLE_METRICS = [
    "hit_at_3",
    "expected_doc_coverage",
    "must_include_coverage",
    "forbidden_claim_rate",
    "answer_context_overlap_score",
    "latency_ms",
    "estimated_cost",
]

_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _avg_metrics(results: list["CaseResult"]) -> dict[str, float]:
    if not results:
        return {}
    keys = set().union(*(r.metrics.keys() for r in results))
    return {k: sum(r.metrics.get(k, 0.0) for r in results) / len(results) for k in keys}


def _iter_findings(results: list["CaseResult"]):
    """遍历全部失败归因（findings 缺失时由 failure_types 回退为 only-type 伪条目）。"""
    for result in results:
        if result.failure_findings:
            for f in result.failure_findings:
                yield result.case_id, f.failure_type, f.severity, f.message
        else:
            for t in result.failure_types:
                yield result.case_id, t, "unknown", ""


def _case_status(case_result: "CaseResult") -> str:
    if case_result.failure_findings:
        worst = max((_SEV_RANK[f.severity] for f in case_result.failure_findings), default=0)
        mapping = {3: "FAIL", 2: "FAIL", 1: "WARN", 0: "LOW"}
        return mapping.get(worst, "OK")
    if case_result.failure_types:
        return "FAIL"
    return "OK"


def _aggregate_failure_stats(results: list["CaseResult"]) -> dict[str, dict]:
    """返回 {type: {"count", "sev_critical"...}} 与 severity 分布。"""
    type_counter: dict[str, int] = {}
    type_sev: dict[str, str] = {}
    sev_counter: dict[str, int] = {}
    for _, ftype, sev, _message in _iter_findings(results):
        type_counter[ftype] = type_counter.get(ftype, 0) + 1
        if sev != "unknown" and _SEV_RANK.get(sev, -1) > _SEV_RANK.get(type_sev.get(ftype, ""), -1):
            type_sev[ftype] = sev
        if sev != "unknown":
            sev_counter[sev] = sev_counter.get(sev, 0) + 1
    return {
        "by_type": type_counter,
        "type_severity": type_sev,
        "by_severity": sev_counter,
    }


def _recommendations(stats: dict) -> list[str]:
    recs: list[str] = []
    by_type = stats["by_type"]
    if by_type.get("retrieval_miss", 0):
        recs.append(f"存在 {by_type['retrieval_miss']} 个 retrieval miss → 建议改善检索器 / 索引 / 查询重写。")
    if by_type.get("bad_ranking", 0):
        recs.append(f"存在 {by_type['bad_ranking']} 个 partial retrieval / 低上下文精度 → 建议优化检索召回与排序。")
    if "incomplete_answer" in by_type:
        recs.append(f"存在 {by_type['incomplete_answer']} 个 incomplete_answer（must_include 缺失或低忠实度）→ 建议改善答案合成 prompt，并加强基于检索上下文的引用 grounding。")
    if by_type.get("missing_multi_hop", 0):
        recs.append(f"存在 {by_type['missing_multi_hop']} 个 missing multi-hop → 建议改善 planner / 查询分解。")
    if by_type.get("query_rewrite_missing", 0):
        recs.append(f"存在 {by_type['query_rewrite_missing']} 个 query rewrite 缺失 → 建议提升查询理解与改写能力。")
    if by_type.get("expected_tool_missing", 0):
        recs.append(f"存在 {by_type['expected_tool_missing']} 个期望工具未调用 → 建议优化工具调用策略。")
    if by_type.get("forbidden_claim", 0) or by_type.get("forbidden_tool_called", 0):
        recs.append("检测到 forbidden claim / forbidden tool → 建议加强内容与工具调用护栏。")
    if by_type.get("latency_regression", 0):
        recs.append(f"存在 {by_type['latency_regression']} 个高延迟 → 建议优化检索轮数 / 上下文长度 / 链路延迟。")
    if by_type.get("cost_regression", 0):
        recs.append(f"存在 {by_type['cost_regression']} 个高成本 → 建议降低冗余检索与上下文输入。")
    if not recs:
        recs.append("未发现显著失败，当前配置表现稳定（可维持）。")
    return recs


# ---------------------------------------------------------------------------
# 单实验报告
# ---------------------------------------------------------------------------

def generate_experiment_report(
    *,
    experiment_id: str,
    name: str,
    rag_version: str,
    created_at: str,
    case_results: list["CaseResult"],
    cases: dict[str, "EvalCase"] | None = None,
) -> str:
    """生成单实验 Markdown 报告。"""
    cases = cases or {}
    avg = _avg_metrics(case_results)
    stats = _aggregate_failure_stats(case_results)

    lines: list[str] = ["# Experiment Report", ""]
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"- experiment_id: `{experiment_id}`")
    lines.append(f"- name: `{name}`")
    lines.append(f"- rag_version: `{rag_version}`")
    lines.append(f"- case count: {len(case_results)}")
    lines.append(f"- created_at: {created_at}")
    lines.append("")

    lines.append("## 2. Aggregate Metrics")
    lines.append("")
    lines.append("> 约束为空时对应比例指标为 0.0 表示“无约束”(N/A)，不代表失败。")
    lines.append("")
    for group, keys in METRIC_GROUPS.items():
        entries = [(k, avg[k]) for k in keys if k in avg]
        if not entries:
            continue
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| 指标 | 平均值 |")
        lines.append("|---|---|")
        for key, value in entries:
            lines.append(f"| `{key}` | {_fmt(value)} |")
        lines.append("")

    lines.append("## 3. Failure Summary")
    lines.append("")
    if not stats["by_type"]:
        lines.append("> 未检测到失败归因（或所有相关约束为空，视为 N/A）。")
        lines.append("")
    else:
        lines.append("### 失败类型分布")
        lines.append("")
        lines.append("| 失败类型 | 数量 | 最高严重度 |")
        lines.append("|---|---|---|")
        for ftype, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{ftype}` | {count} | {stats['type_severity'].get(ftype, '-')} |")
        lines.append("")
        lines.append("### 严重度分布")
        lines.append("")
        lines.append("| 严重度 | 数量 |")
        lines.append("|---|---|")
        for sev in ("critical", "high", "medium", "low"):
            if sev in stats["by_severity"]:
                lines.append(f"| {sev} | {stats['by_severity'][sev]} |")
        lines.append("")
        lines.append("### Top 失败样例")
        lines.append("")
        lines.append("| case_id | 失败类型 | 原因 |")
        lines.append("|---|---|---|")
        failed = sorted(
            (r for r in case_results if r.failure_findings),
            key=lambda r: -max((_SEV_RANK[f.severity] for f in r.failure_findings), default=0),
        )
        for r in failed[:5]:
            types = "、".join(r.failure_types) or "-"
            first_msg = r.failure_findings[0].message if r.failure_findings else "-"
            lines.append(f"| `{r.case_id}` | {types} | {first_msg} |")
        lines.append("")

    lines.append("## 4. Case Results")
    lines.append("")
    header = "| case_id | category | " + " | ".join(f"`{m}`" for m in CASE_TABLE_METRICS) + " | failure_types | status |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CASE_TABLE_METRICS) + 4))
    for r in sorted(case_results, key=lambda x: x.case_id):
        category = cases.get(r.case_id).category if cases.get(r.case_id) else "-"
        cells = " | ".join(_fmt(r.metrics.get(m, 0.0)) for m in CASE_TABLE_METRICS)
        ftypes = "、".join(r.failure_types) or "无"
        lines.append(f"| `{r.case_id}` | {category} | {cells} | {ftypes} | {_case_status(r)} |")
    lines.append("")

    lines.append("## 5. Recommendations")
    lines.append("")
    for rec in _recommendations(stats):
        lines.append(f"- {rec}")
    lines.append("")

    lines.append("## 6. Limitations")
    lines.append("")
    lines.append("- 当前为 deterministic metrics，全部结果可复现，不依赖任何模型判定。")
    lines.append("- 不含 LLM judge（未对答案做模型打分/偏好评估）。")
    lines.append("- 不含 Ragas / DeepEval 集成。")
    lines.append("- MockRAGClient 仅用于本地评测流程验证，不等价于真实 RAG 系统表现。")
    lines.append("- 失败归因基于规则阈值（如 overlap < 0.1、latency > 3000ms 等），可能出现边界误判。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 对比报告
# ---------------------------------------------------------------------------

_OVERALL_LABEL = {
    "improved": "整体改善",
    "regressed": "整体回退",
    "mixed": "混合（部分改善、部分回退）",
    "unchanged": "无明显变化",
}
_STATUS_LABEL = {
    "improved": "改善",
    "regressed": "回退",
    "unchanged": "无变化",
    "missing_candidate_case": "candidate 缺失该 case",
    "new_candidate_case": "candidate 新增 case",
}


def generate_comparison_report(report: RegressionReport) -> str:
    """生成 baseline vs candidate 对比 Markdown 报告。"""
    summary = report.summary
    lines: list[str] = ["# Experiment Comparison Report", ""]
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"- baseline experiment: `{summary.baseline_experiment}`")
    lines.append(f"- candidate experiment: `{summary.candidate_experiment}`")
    lines.append(f"- common cases: {summary.common_cases}")
    lines.append(f"- overall status: **{summary.overall_status}**（{_OVERALL_LABEL.get(summary.overall_status, summary.overall_status)}）")
    lines.append(f"- improved cases: {summary.improved_cases} / regressed cases: {summary.regressed_cases} / unchanged: {summary.unchanged_cases}")
    lines.append("")

    lines.append("## 2. Metric Deltas")
    lines.append("")
    if not report.metric_deltas:
        lines.append("> 无共同可判定指标变化。")
        lines.append("")
    else:
        lines.append("| 指标 | 方向 | baseline_avg | candidate_avg | delta | status |")
        lines.append("|---|---|---|---|---|---|")
        for d in report.metric_deltas:
            arrow = "↑" if d.direction == "higher_is_better" else "↓"
            lines.append(
                f"| `{d.metric_name}` | {arrow} | {_fmt(d.baseline_avg)} | {_fmt(d.candidate_avg)} | {d.delta:+.4f} | {d.status} |"
            )
        lines.append("")

    regressed = [d for d in report.case_deltas if d.status == "regressed"]
    improved = [d for d in report.case_deltas if d.status == "improved"]

    lines.append("## 3. Regressed Cases")
    lines.append("")
    if not regressed:
        lines.append("> 无退化 case。")
        lines.append("")
    else:
        lines.append("| case_id | 退化指标 | 新增失败类型 |")
        lines.append("|---|---|---|")
        for d in regressed:
            changed = "、".join(f"{m}:{v:+.3f}" for m, v in d.changed_metrics.items() if v < 0 and METRIC_DIRECTIONS.get(m) == "higher_is_better")
            changed_more = "、".join(f"{m}:{v:+.3f}" for m, v in d.changed_metrics.items() if v > 0 and METRIC_DIRECTIONS.get(m) == "lower_is_better")
            degraded = "、".join(x for x in (changed, changed_more) if x) or "-"
            new_f = "、".join(d.new_failure_types) or "-"
            lines.append(f"| `{d.case_id}` | {degraded} | {new_f} |")
        lines.append("")

    lines.append("## 4. Improved Cases")
    lines.append("")
    if not improved:
        lines.append("> 无改善 case。")
        lines.append("")
    else:
        lines.append("| case_id | 改善指标 | 解决的失败类型 |")
        lines.append("|---|---|---|")
        for d in improved:
            gained = "、".join(f"{m}:{v:+.3f}" for m, v in d.changed_metrics.items() if v > 0 and METRIC_DIRECTIONS.get(m) == "higher_is_better")
            gained_more = "、".join(f"{m}:{v:+.3f}" for m, v in d.changed_metrics.items() if v < 0 and METRIC_DIRECTIONS.get(m) == "lower_is_better")
            improved_m = "、".join(x for x in (gained, gained_more) if x) or "-"
            resolved = "、".join(d.resolved_failure_types) or "-"
            lines.append(f"| `{d.case_id}` | {improved_m} | {resolved} |")
        lines.append("")

    lines.append("## 5. Recommendation")
    lines.append("")
    if summary.overall_status == "improved" and summary.regressed_cases == 0:
        lines.append("**建议升级 candidate**：整体指标改善且无退化 case。")
    elif summary.overall_status == "improved":
        lines.append(f"**谨慎升级 candidate**：整体改善，但仍有 {summary.regressed_cases} 个 case 退化，需结合退化原因判断。")
    elif summary.overall_status == "mixed":
        lines.append(f"**谨慎升级 candidate**：改善与退化并存（改善 {summary.improved_cases} / 退化 {summary.regressed_cases}），建议先定位退化根因。")
    elif summary.overall_status == "unchanged":
        lines.append("**暂不建议升级**：candidate 与 baseline 无明显差异。")
    else:
        lines.append("**不建议升级 candidate**：存在退化指标或新增失败类型，建议先定位回归根因。")
    lines.append("")
    return "\n".join(lines)


__all__ = ["generate_experiment_report", "generate_comparison_report"]