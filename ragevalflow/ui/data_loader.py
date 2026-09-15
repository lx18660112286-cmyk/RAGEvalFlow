"""Dashboard 数据层（阶段 4）。

从 SQLite 读取数据并聚合，供 Streamlit 页面展示。本模块**不 import streamlit**，
保持纯逻辑、可单测。

所有公开函数都不抛异常：数据库不存在 / 未初始化 / 实验不存在时返回带
``error`` 字段的字典或空列表，由页面层渲染中文提示，避免 Python stack trace。

复用现有能力：
- 读取：``ragevalflow.core.storage.Storage``；
- 对比：``ragevalflow.analysis.regression.compare_experiments``（不重写回归逻辑）；
- 报告：``ragevalflow.reporting.markdown_report``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ragevalflow.analysis.regression import ComparisonError, RegressionReport, compare_experiments
from ragevalflow.core.storage import Storage
from ragevalflow.reporting.markdown_report import (
    generate_comparison_report as _generate_comparison_md,
    generate_experiment_report as _generate_experiment_md,
)

DEFAULT_DB = Path("data") / "ragevalflow.db"
_REQUIRED_TABLES = {"eval_cases", "experiments", "case_results"}

# 实验列表展示所需的聚合指标（对应 runner.ExperimentRunSummary）
_LIST_METRICS = (
    "hit_at_3",
    "expected_doc_coverage",
    "must_include_coverage",
    "answer_context_overlap_score",
    "latency_ms",
)

# Case 明细表字段（与阶段 3 报告 CASE_TABLE_METRICS 对齐）
CASE_TABLE_METRICS = (
    "hit_at_3",
    "expected_doc_coverage",
    "must_include_coverage",
    "answer_context_overlap_score",
    "latency_ms",
)


def resolve_db_path(db_input: str | None = None) -> Path:
    """数据库路径优先级：sidebar 传入 > 环境变量 RAGEFLOW_DB > 默认 ./data/ragevalflow.db。"""
    if db_input:
        return Path(db_input)
    env_db = os.environ.get("RAGEFLOW_DB")
    if env_db:
        return Path(env_db)
    return DEFAULT_DB


def _check_ready(db_path: Path) -> Storage:
    """校验数据库存在且已初始化，否则抛出带中文信息的 DashboardError。"""
    if not Path(db_path).exists():
        raise DashboardError("数据库不存在，请先执行 python -m ragevalflow.main init-db")
    storage = Storage(db_path)
    tables = storage.user_table_names()
    if not _REQUIRED_TABLES.issubset(tables):
        raise DashboardError("数据库未初始化（缺少所需用户表），请先执行 python -m ragevalflow.main init-db")
    return storage


class DashboardError(Exception):
    """数据库/查询前置条件不满足，消息为中文、可直接展示。"""


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def load_overview(db_path: Path | str) -> dict[str, Any]:
    """Overview 页数据；数据库不可用时返回带 ``error`` 的字典。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError as exc:
        return {
            "error": str(exc),
            "db_path": str(db_path),
            "eval_cases": 0,
            "experiments": 0,
            "case_results": 0,
            "latest_experiment": None,
            "report_hints": [],
        }

    experiments = storage.list_experiments()
    total_results = 0
    latest = None
    if experiments:
        latest_row = experiments[0]
        latest = {
            "experiment_id": latest_row.experiment_id,
            "name": latest_row.name,
            "rag_version": latest_row.rag_version,
            "created_at": latest_row.created_at,
        }
        for row in experiments:
            total_results += storage.count_case_results(row.experiment_id)

    report_hints = _report_hints()
    return {
        "db_path": str(db_path),
        "eval_cases": storage.count_cases(),
        "experiments": len(experiments),
        "case_results": total_results,
        "latest_experiment": latest,
        "report_hints": report_hints,
    }


def _report_hints() -> list[str]:
    """提示已存在的 Markdown 报告文件，便于演示快速跳转。"""
    hints = []
    for name in ("baseline", "agentic_v1", "compare"):
        p = Path("reports") / f"{name}.md"
        if p.exists():
            hints.append(f"✅ 报告已生成：`{p}`")
        else:
            hints.append(f"⚠️ 报告未生成：`{p}`（可在对应页签点击生成按钮）")
    return hints


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def load_experiments(db_path: Path | str) -> list[dict[str, Any]]:
    """实验列表（含聚合指标）。数据库不可用/无实验时返回空列表，不崩溃。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError:
        return []

    rows = storage.list_experiments()
    items: list[dict[str, Any]] = []
    for row in rows:
        results = storage.list_case_results(row.experiment_id)
        n = len(results)
        avg = _avg(results, _LIST_METRICS)
        items.append(
            {
                "experiment_id": row.experiment_id,
                "name": row.name,
                "rag_version": row.rag_version,
                "created_at": row.created_at,
                "case_results": n,
                **{f"avg_{k}": avg.get(k, 0.0) for k in _LIST_METRICS},
                "failure_total": sum(len(r.failure_types) for r in results),
            }
        )
    return items


def _avg(results, keys: tuple[str, ...]) -> dict[str, float]:
    if not results:
        return {}
    return {k: sum(r.metrics.get(k, 0.0) for r in results) / len(results) for k in keys}


def list_case_ids(db_path: Path | str, experiment_id: str) -> list[str]:
    """返回某实验的全部 case_id（用于 Case Detail 下拉框）。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError:
        return []
    return [r.case_id for r in storage.list_case_results(experiment_id)]


# ---------------------------------------------------------------------------
# Experiment Detail
# ---------------------------------------------------------------------------

def load_experiment_detail(db_path: Path | str, experiment_id: str) -> dict[str, Any]:
    """单个实验的聚合指标 / 失败分布 / case 明细表。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError as exc:
        return {"error": str(exc)}

    row = storage.get_experiment_by_id_or_name(experiment_id)
    if row is None:
        return {"error": f"未找到实验 '{experiment_id}'（请确认 experiment_id 或实验名）"}
    results = storage.list_case_results(row.experiment_id)
    if not results:
        return {"error": f"实验 '{row.name}' 没有 case_results，请先执行 run 生成结果"}

    cases = {c.id: c for c in storage.list_cases()}
    return {
        "experiment_id": row.experiment_id,
        "name": row.name,
        "rag_version": row.rag_version,
        "created_at": row.created_at,
        "aggregate_metrics": _avg(results, tuple(results[0].metrics.keys())),
        "failure_type_dist": _failure_type_dist(results),
        "severity_dist": _severity_dist(results),
        "case_results": [_case_row(r, cases.get(r.case_id)) for r in results],
    }


def _failure_type_dist(results) -> list[dict[str, Any]]:
    counter: dict[str, int] = {}
    for r in results:
        for t in r.failure_types:
            counter[t] = counter.get(t, 0) + 1
    return [{"failure_type": k, "count": v} for k, v in sorted(counter.items(), key=lambda x: -x[1])]


def _severity_dist(results) -> list[dict[str, Any]]:
    counter: dict[str, int] = {}
    for r in results:
        for f in r.failure_findings:
            counter[f.severity] = counter.get(f.severity, 0) + 1
    return [{"severity": k, "count": v} for k, v in counter.items()]


def _case_row(r, case) -> dict[str, Any]:
    return {
        "case_id": r.case_id,
        "category": getattr(case, "category", "-") if case else "-",
        **{m: r.metrics.get(m, 0.0) for m in CASE_TABLE_METRICS},
        "failure_types": "、".join(r.failure_types) or "无",
        "status": _status(r),
    }


def _status(r) -> str:
    if not r.failure_findings and not r.failure_types:
        return "OK"
    if not r.failure_findings:
        return "FAIL"
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    worst = max((rank[f.severity] for f in r.failure_findings), default=0)
    return {3: "FAIL", 2: "FAIL", 1: "WARN", 0: "LOW"}.get(worst, "OK")


# ---------------------------------------------------------------------------
# Case Detail
# ---------------------------------------------------------------------------

def load_case_detail(db_path: Path | str, experiment_id: str, case_id: str) -> dict[str, Any]:
    """单个 case 的完整明细（面试演示核心区域）。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError as exc:
        return {"error": str(exc)}

    paired = storage.get_case_result_with_case(experiment_id, case_id)
    if paired is None:
        return {"error": f"未找到实验 '{experiment_id}' 下的 case '{case_id}'"}
    result, case = paired
    output = result.rag_output
    return {
        "case_id": case.id,
        "category": case.category,
        "question": case.question,
        "reference_answer": case.reference_answer,
        "expected_docs": case.expected_docs,
        "must_include": case.must_include,
        "must_not_include": case.must_not_include,
        "answer": output.answer,
        "contexts": [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "rank": c.rank, "text": c.text} for c in output.contexts],
        "trace": output.trace.model_dump(),
        "runtime": output.runtime.model_dump(),
        "metrics": result.metrics,
        "findings": [f.model_dump() for f in result.failure_findings],
        "failure_types": result.failure_types,
    }


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def run_comparison(db_path: Path | str, baseline: str, candidate: str) -> dict[str, Any]:
    """复用 regression.compare_experiments 做 baseline vs candidate 对比。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError as exc:
        return {"error": str(exc)}

    b_row = storage.get_experiment_by_id_or_name(baseline)
    if b_row is None:
        return {"error": f"未找到 baseline 实验 '{baseline}'"}
    c_row = storage.get_experiment_by_id_or_name(candidate)
    if c_row is None:
        return {"error": f"未找到 candidate 实验 '{candidate}'"}

    b_results = storage.list_case_results(b_row.experiment_id)
    c_results = storage.list_case_results(c_row.experiment_id)
    if not b_results:
        return {"error": f"baseline 实验 '{b_row.name}' 没有 case_results"}
    if not c_results:
        return {"error": f"candidate 实验 '{c_row.name}' 没有 case_results"}

    try:
        report: RegressionReport = compare_experiments(b_results, c_results)
    except ComparisonError as exc:
        return {"error": str(exc)}

    return {"report": report}


def comparison_to_dicts(report: RegressionReport) -> dict[str, Any]:
    """把 RegressionReport 转成可供 st.dataframe / st.bar_chart 展示的 dict。

    ``improved_cases`` / ``regressed_cases`` / ``unchanged_cases`` 为计数（供 st.metric），
    具体的 case 列表放在 ``improved_case_rows`` / ``regressed_case_rows``。
    """
    s = report.summary
    return {
        "overall_status": s.overall_status,
        "improved_cases": s.improved_cases,
        "regressed_cases": s.regressed_cases,
        "unchanged_cases": s.unchanged_cases,
        "common_cases": s.common_cases,
        "baseline_experiment": s.baseline_experiment,
        "candidate_experiment": s.candidate_experiment,
        "metric_deltas": [
            {
                "metric_name": m.metric_name,
                "baseline_avg": m.baseline_avg,
                "candidate_avg": m.candidate_avg,
                "delta": m.delta,
                "status": m.status,
            }
            for m in report.metric_deltas
        ],
        "regressed_case_rows": [
            {
                "case_id": d.case_id,
                "changed_metrics": _fmt_changes(d.changed_metrics),
                "new_failure_types": "、".join(d.new_failure_types) or "-",
            }
            for d in report.case_deltas
            if d.status == "regressed"
        ],
        "improved_case_rows": [
            {
                "case_id": d.case_id,
                "changed_metrics": _fmt_changes(d.changed_metrics),
                "resolved_failure_types": "、".join(d.resolved_failure_types) or "-",
            }
            for d in report.case_deltas
            if d.status == "improved"
        ],
    }


def _fmt_changes(changed: dict[str, float]) -> str:
    if not changed:
        return "-"
    return "、".join(f"{k}:{v:+.3f}" for k, v in sorted(changed.items()))


# ---------------------------------------------------------------------------
# 报告生成（调用现有 markdown_report 生成器）
# ---------------------------------------------------------------------------

def generate_experiment_report(db_path: Path | str, experiment_id: str, output: str | None = None) -> dict[str, Any]:
    """生成单实验 Markdown 报告，返回 {"path": ...}。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError as exc:
        return {"error": str(exc)}

    row = storage.get_experiment_by_id_or_name(experiment_id)
    if row is None:
        return {"error": f"未找到实验 '{experiment_id}'"}
    results = storage.list_case_results(row.experiment_id)
    if not results:
        return {"error": f"实验 '{row.name}' 没有 case_results"}

    cases = {c.id: c for c in storage.list_cases()}
    md = _generate_experiment_md(
        experiment_id=row.experiment_id,
        name=row.name,
        rag_version=row.rag_version,
        created_at=row.created_at,
        case_results=results,
        cases=cases,
    )
    out = Path(output) if output else Path("reports") / f"{row.name}_{row.experiment_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return {"path": str(out)}


def generate_comparison_report(db_path: Path | str, baseline: str, candidate: str, output: str | None = None) -> dict[str, Any]:
    """生成对比 Markdown 报告，返回 {"path": ...}。"""
    db_path = Path(db_path)
    try:
        storage = _check_ready(db_path)
    except DashboardError as exc:
        return {"error": str(exc)}

    b_row = storage.get_experiment_by_id_or_name(baseline)
    if b_row is None:
        return {"error": f"未找到 baseline 实验 '{baseline}'"}
    c_row = storage.get_experiment_by_id_or_name(candidate)
    if c_row is None:
        return {"error": f"未找到 candidate 实验 '{candidate}'"}

    b_results = storage.list_case_results(b_row.experiment_id)
    c_results = storage.list_case_results(c_row.experiment_id)
    if not b_results or not c_results:
        return {"error": "baseline 或 candidate 实验没有 case_results"}

    try:
        report = compare_experiments(b_results, c_results)
    except ComparisonError as exc:
        return {"error": str(exc)}

    md = _generate_comparison_md(report)
    out = (
        Path(output)
        if output
        else Path("reports") / f"compare_{b_row.name}_{c_row.name}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return {"path": str(out)}


__all__ = [
    "resolve_db_path",
    "DEFAULT_DB",
    "DashboardError",
    "load_overview",
    "load_experiments",
    "list_case_ids",
    "load_experiment_detail",
    "load_case_detail",
    "run_comparison",
    "comparison_to_dicts",
    "generate_experiment_report",
    "generate_comparison_report",
]
