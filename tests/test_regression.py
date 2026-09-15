"""回归检测（regression.compare_experiments）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragevalflow.analysis.failure_analyzer import analyze_failures
from ragevalflow.analysis.regression import ComparisonError, compare_experiments
from ragevalflow.integrations.rag_client import MockRAGClient
from ragevalflow.metrics import compute_all_metrics
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.experiment_result import CaseResult
from ragevalflow.schemas.rag_output import RAGOutput


def _result(experiment: str, case_id: str, metrics: dict, failure_types: list[str] | None = None) -> CaseResult:
    return CaseResult(
        experiment_id=experiment,
        case_id=case_id,
        rag_output=RAGOutput(question="q", answer="a"),
        metrics=metrics,
        failure_types=list(failure_types or []),
    )


def _baseline_results() -> list[CaseResult]:
    return [
        _result(
            "exp_base",
            "c1",
            {
                "hit_at_3": 1.0,
                "expected_doc_coverage": 0.5,
                "must_include_coverage": 0.5,
                "answer_context_overlap_score": 0.05,
                "forbidden_claim_rate": 0.0,
                "latency_ms": 300.0,
                "estimated_cost": 0.004,
            },
            failure_types=["incomplete_answer", "bad_ranking"],
        ),
        _result(
            "exp_base",
            "c2",
            {"hit_at_3": 1.0, "expected_doc_coverage": 1.0, "must_include_coverage": 1.0, "latency_ms": 300.0},
            failure_types=[],
        ),
    ]


def _improved_candidate() -> list[CaseResult]:
    return [
        _result(
            "exp_cand",
            "c1",
            {
                "hit_at_3": 1.0,
                "expected_doc_coverage": 1.0,
                "must_include_coverage": 1.0,
                "answer_context_overlap_score": 0.5,
                "forbidden_claim_rate": 0.0,
                "latency_ms": 200.0,
                "estimated_cost": 0.004,
            },
            failure_types=[],
        ),
        _result(
            "exp_cand",
            "c2",
            {"hit_at_3": 1.0, "expected_doc_coverage": 1.0, "must_include_coverage": 1.0, "latency_ms": 300.0},
            failure_types=[],
        ),
    ]


def test_compare_identifies_improvement():
    """candidate 指标全面改善且修复失败 → improved。"""
    report = compare_experiments(_baseline_results(), _improved_candidate())
    assert report.summary.common_cases == 2
    assert report.summary.improved_cases == 1
    assert report.summary.regressed_cases == 0
    assert report.summary.unchanged_cases == 1
    assert report.summary.overall_status == "improved"

    by_case = {d.case_id: d for d in report.case_deltas}
    assert by_case["c1"].status == "improved"
    assert by_case["c1"].resolved_failure_types == ["bad_ranking", "incomplete_answer"]

    by_metric = {m.metric_name: m for m in report.metric_deltas}
    assert by_metric["must_include_coverage"].status == "improved"
    assert by_metric["latency_ms"].direction == "lower_is_better"
    assert by_metric["latency_ms"].status == "improved"  # -33% >= 20%


def test_compare_identifies_constructed_regression():
    """人为构造退化：指标下降 + 新增失败 → regressed。"""
    candidate = _improved_candidate()
    candidate[0] = _result(
        "exp_cand",
        "c1",
        {
            "hit_at_3": 1.0,
            "expected_doc_coverage": 0.0,
            "must_include_coverage": 0.0,
            "forbidden_claim_rate": 0.5,
            "latency_ms": 600.0,
            "estimated_cost": 0.004,
        },
        failure_types=["forbidden_claim"],
    )
    report = compare_experiments(_baseline_results(), candidate)
    assert report.summary.regressed_cases == 1
    assert report.summary.overall_status == "regressed"

    c1 = next(d for d in report.case_deltas if d.case_id == "c1")
    assert c1.status == "regressed"
    assert c1.new_failure_types == ["forbidden_claim"]
    # 退化指标应包含 must_include_coverage
    assert "must_include_coverage" in c1.changed_metrics


def test_no_common_case_raises():
    with pytest.raises(ComparisonError):
        compare_experiments(
            [_result("base", "c1", {"latency_ms": 1.0})],
            [_result("cand", "c9", {"latency_ms": 2.0})],
        )


def test_missing_and_new_candidate_case_marking():
    """baseline 独有 / candidate 独有 case 被标记。"""
    base = [_result("base", "c1", {"latency_ms": 1.0}), _result("base", "c2", {"latency_ms": 1.0})]
    cand = [_result("cand", "c1", {"latency_ms": 1.0}), _result("cand", "c3", {"latency_ms": 1.0})]
    report = compare_experiments(base, cand)
    statuses = {d.case_id: d.status for d in report.case_deltas}
    assert statuses["c1"] == "unchanged"
    assert statuses["c2"] == "missing_candidate_case"
    assert statuses["c3"] == "new_candidate_case"


def test_new_failure_type_flags_regression():
    """指标不变但 candidate 新增失败类型 → regressed。"""
    base = [_result("base", "c1", {"must_include_coverage": 1.0}, failure_types=[])]
    cand = [_result("cand", "c1", {"must_include_coverage": 1.0}, failure_types=["forbidden_claim"])]
    report = compare_experiments(base, cand)
    assert report.case_deltas[0].status == "regressed"
    assert report.case_deltas[0].new_failure_types == ["forbidden_claim"]


def test_metric_delta_aggregation_and_direction():
    """metric_deltas 聚合平均值与方向正确。"""
    report = compare_experiments(_baseline_results(), _improved_candidate())
    by_metric = {m.metric_name: m for m in report.metric_deltas}
    assert abs(by_metric["must_include_coverage"].delta - 0.25) < 1e-6  # avg 0.75 - 0.5
    assert by_metric["expected_doc_coverage"].direction == "higher_is_better"
    assert by_metric["latency_ms"].delta < 0
    # 未配置方向的指标不出现在 metric_deltas
    assert "answer_length" not in by_metric


# ---------------------------------------------------------------------------
# 阶段 3.1：全数据集回归测试（修复 case_it_002 forbidden_claim）
# ---------------------------------------------------------------------------

def _run_mock_experiment(mode: str, cases: list[EvalCase], top_k: int) -> list[CaseResult]:
    """用 MockRAGClient 跑完整个数据集，生成与 runner 同构的 CaseResult 列表。"""
    client = MockRAGClient(mode, top_k=top_k)
    results = []
    for case in cases:
        output = client.answer(case)
        metrics = compute_all_metrics(case, output)
        findings = analyze_failures(case, output, metrics)
        results.append(
            CaseResult(
                experiment_id=f"exp_{mode}",
                case_id=case.id,
                rag_output=output,
                metrics=metrics,
                failure_types=[f.failure_type for f in findings],
                failure_findings=findings,
            )
        )
    return results


def _real_dataset_cases() -> list[EvalCase]:
    p = Path("datasets/eval_cases.jsonl")
    if not p.exists():  # 兜底：找不到数据集文件则跳过该测试
        pytest.skip("datasets/eval_cases.jsonl 不存在")
    return [
        EvalCase.model_validate(json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_full_dataset_agentic_no_forbidden_claim():
    """修复后，agentic_v1 在真实数据集上 forbidden_claim_rate 恒为 0、无 forbidden_claim 失败。"""
    cases = _real_dataset_cases()
    agentic = _run_mock_experiment("agentic_v1", cases, 5)
    assert agentic  # 非空
    for r in agentic:
        assert r.metrics.get("forbidden_claim_rate", 0.0) == 0.0
        assert "forbidden_claim" not in r.failure_types


def test_full_dataset_compare_no_critical_regression_and_improved():
    """阶段 3.1：baseline vs agentic_v1 全数据集对比不再因 case_it_002 退化，overall_status 应为 improved。"""
    cases = _real_dataset_cases()
    baseline = _run_mock_experiment("baseline", cases, 3)
    agentic = _run_mock_experiment("agentic_v1", cases, 5)

    report = compare_experiments(baseline, agentic)
    # case_it_002 不再退化，且被判定为改善
    it002 = next(d for d in report.case_deltas if d.case_id == "case_it_002")
    assert it002.status == "improved"
    assert "forbidden_claim" not in it002.new_failure_types

    # 无退化 case，整体 improved
    assert report.summary.regressed_cases == 0
    assert report.summary.improved_cases > 0
    assert report.summary.overall_status == "improved"

    # metric_deltas 中 forbidden_claim_rate 不应再为 regressed
    by_metric = {m.metric_name: m for m in report.metric_deltas}
    assert by_metric["forbidden_claim_rate"].status != "regressed"
    assert abs(by_metric["forbidden_claim_rate"].candidate_avg) < 1e-9