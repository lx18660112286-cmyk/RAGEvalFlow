"""Markdown 报告生成器测试。"""

from __future__ import annotations

from ragevalflow.analysis.regression import RegressionReport, RegressionSummary
from ragevalflow.reporting.markdown_report import generate_comparison_report, generate_experiment_report
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.experiment_result import CaseResult
from ragevalflow.schemas.failure import FailureFinding
from ragevalflow.schemas.rag_output import RAGOutput


def _finding(case_id, ftype, severity, message="失败原因"):
    return FailureFinding(case_id=case_id, failure_type=ftype, severity=severity, message=message)


def _result(case_id: str, metrics: dict, types: list[str], findings: list[FailureFinding]) -> CaseResult:
    return CaseResult(
        experiment_id="exp_1",
        case_id=case_id,
        rag_output=RAGOutput(question="q", answer="a"),
        metrics=metrics,
        failure_types=types,
        failure_findings=findings,
    )


RESULTS = [
    _result(
        "c1",
        {"hit_at_3": 1.0, "expected_doc_coverage": 0.5, "must_include_coverage": 1.0, "forbidden_claim_rate": 0.0, "answer_context_overlap_score": 0.4, "latency_ms": 200.0, "estimated_cost": 0.004},
        ["bad_ranking"],
        [_finding("c1", "bad_ranking", "medium")],
    ),
    _result(
        "c2",
        {"hit_at_3": 0.0, "expected_doc_coverage": 0.0, "must_include_coverage": 0.0, "answer_context_overlap_score": 0.0, "latency_ms": 4000.0, "estimated_cost": 0.03},
        ["retrieval_miss", "incomplete_answer", "latency_regression"],
        [
            _finding("c2", "retrieval_miss", "high", "未检索到期望文档"),
            _finding("c2", "incomplete_answer", "high", "缺少 must_include"),
            _finding("c2", "latency_regression", "medium"),
        ],
    ),
    _result(
        "c3",
        {"hit_at_3": 1.0, "expected_doc_coverage": 1.0, "must_include_coverage": 1.0, "answer_context_overlap_score": 0.8, "latency_ms": 100.0, "estimated_cost": 0.002},
        [],
        [],
    ),
]
CASES = {
    "c1": EvalCase(id="c1", category="IT", question="q"),
    "c2": EvalCase(id="c2", category="HR", question="q"),
    "c3": EvalCase(id="c3", category="产品", question="q"),
}


def test_experiment_report_sections_and_content():
    md = generate_experiment_report(
        experiment_id="exp_1",
        name="baseline",
        rag_version="mock-baseline-v0",
        created_at="2026-09-15T12:00:00",
        case_results=RESULTS,
        cases=CASES,
    )
    assert "# Experiment Report" in md
    for section in ("## 1. Summary", "## 2. Aggregate Metrics", "## 3. Failure Summary", "## 4. Case Results", "## 5. Recommendations", "## 6. Limitations"):
        assert section in md
    # Summary 字段
    assert "exp_1" in md and "mock-baseline-v0" in md and "case count: 3" in md
    # 聚合指标分组出现
    assert "检索指标" in md and "忠实度指标" in md
    # 失败分布：bad_ranking / retrieval_miss / latency_regression
    assert "bad_ranking" in md and "retrieval_miss" in md
    # 严重度分布
    assert "| high | 2 |" in md or "| high |" in md
    # Case 表
    assert "| `c2` |" in md and "| FAIL |" in md
    # 推荐（retrieval miss -> 检索建议；高延迟 -> 延迟建议）
    assert "检索器" in md and "延迟" in md
    # 限制说明
    assert "deterministic" in md and "LLM judge" in md and "Ragas" in md


def test_experiment_report_empty_constraint_note():
    """空约束时报告以 N/A 注明，不解释为失败。"""
    md = generate_experiment_report(
        experiment_id="exp_2",
        name="no_constraint",
        rag_version="",
        created_at="2026-09-15T12:00:00",
        case_results=[
            _result("c1", {"hit_at_3": 0.0, "must_include_coverage": 0.0, "latency_ms": 10.0}, [], []),
        ],
    )
    assert "N/A" in md or "无约束" in md
    assert "未检测到失败归因" in md


def test_comparison_report_sections_and_status():
    report = RegressionReport(
        summary=RegressionSummary(
            baseline_experiment="exp_base",
            candidate_experiment="exp_cand",
            common_cases=2,
            improved_cases=1,
            regressed_cases=1,
            unchanged_cases=0,
            overall_status="mixed",
        ),
    )
    md = generate_comparison_report(report)
    assert "# Experiment Comparison Report" in md
    for section in ("## 1. Summary", "## 2. Metric Deltas", "## 3. Regressed Cases", "## 4. Improved Cases", "## 5. Recommendation"):
        assert section in md
    assert "exp_base" in md and "exp_cand" in md
    assert "mixed" in md
    assert "谨慎升级 candidate" in md