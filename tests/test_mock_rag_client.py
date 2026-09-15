"""MockRAGClient 测试：两种模式差异、确定性、边界输入。"""

from __future__ import annotations

import pytest

from ragevalflow.analysis.failure_analyzer import analyze_failures
from ragevalflow.integrations.rag_client import MockRAGClient
from ragevalflow.metrics.answer import compute_answer_metrics
from ragevalflow.schemas.eval_case import EvalCase, ExpectedBehavior


def _agentic_case() -> EvalCase:
    return EvalCase(
        id="case_multi",
        question="外包员工请假后年假如何折算？",
        reference_answer="按实际服务月份折算，并完成请假登记。",
        expected_docs=["HR-LEAVE-003", "HR-OUT-002"],
        must_include=["折算", "登记"],
        must_not_include=["立即"],
        expected_behavior=ExpectedBehavior(
            should_rewrite=True,
            should_multi_hop=True,
            max_retrieval_rounds=2,
            expected_tools=["hr_policy_db"],
        ),
    )


def test_baseline_vs_agentic_outputs_differ():
    """baseline 与 agentic_v1 输出不同，且 agentic 明显更强。"""
    case = _agentic_case()
    baseline = MockRAGClient("baseline", top_k=3).answer(case)
    agentic = MockRAGClient("agentic_v1", top_k=3).answer(case)

    # 检索：baseline 只命中一半期望文档，agentic 全部命中
    assert len(agentic.contexts) == 2
    assert len(baseline.contexts) == 1
    assert baseline.contexts[0].doc_id != agentic.contexts[1].doc_id

    # 答案：agentic 包含全部必须关键词，baseline 只包含一半（[::2]）
    assert all(kw in agentic.answer for kw in case.must_include)
    assert "折算" in baseline.answer  # 偶数索引命中
    assert "登记" not in baseline.answer  # 缺失

    # trace：agentic 具备 rewrite / multi-hop / reranker / reflection
    assert agentic.trace.query_rewrite is not None
    assert baseline.trace.query_rewrite is None
    assert agentic.trace.retrieval_rounds == 2
    assert baseline.trace.retrieval_rounds == 1
    assert agentic.trace.used_reranker and agentic.trace.used_reflection
    assert not baseline.trace.used_reranker and not baseline.trace.used_reflection
    assert "hr_policy_db" in agentic.trace.tools_used

    # 运行时：agentic 更快、成本更低
    assert agentic.runtime.latency_ms < baseline.runtime.latency_ms


def test_deterministic_output():
    """同一输入多次调用输出完全一致（无可复现性问题）。"""
    case = _agentic_case()
    first = MockRAGClient("agentic_v1", top_k=5).answer(case).model_dump()
    second = MockRAGClient("agentic_v1", top_k=5).answer(case).model_dump()
    assert first == second


def test_empty_expected_docs_no_error():
    case = EvalCase(id="plain", question="没有期望文档的问题")
    baseline = MockRAGClient("baseline").answer(case)
    agentic = MockRAGClient("agentic_v1").answer(case)
    assert baseline.contexts == []
    assert agentic.contexts == []
    assert agentic.trace.retrieval_rounds == 1


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        MockRAGClient("unknown_mode")


def test_mock_config_in_output():
    case = EvalCase(id="c", question="q")
    output = MockRAGClient("baseline", top_k=3).answer(case)
    assert output.config["client"] == "mock"
    assert output.config["mock_mode"] == "baseline"


def test_agentic_excludes_must_not_include_from_reference_answer():
    """阶段 3.1 修复：参考答案含 must_not_include 时，agentic_v1 必须剥离禁止内容。

    复现 case_it_002：reference_answer 为“…无需管理员权限”，must_not_include=["管理员权限"]。
    """
    case = EvalCase(
        id="case_it_002",
        question="忘记密码如何重置？",
        reference_answer="通过 IT 帮助台自助服务页面重置，无需管理员权限。",
        expected_docs=["IT-AUTH-001"],
        must_include=["帮助台"],
        must_not_include=["管理员权限"],
    )
    output = MockRAGClient("agentic_v1", top_k=5).answer(case)
    # 禁止短语绝不出现在输出答案中
    assert "管理员权限" not in output.answer
    # 必须内容仍被保留
    assert "帮助台" in output.answer

    metrics = compute_answer_metrics(case, output)
    assert metrics["forbidden_claim_rate"] == 0.0
    assert metrics["must_include_coverage"] == 1.0

    findings = analyze_failures(case, output, metrics)
    assert not any(f.failure_type == "forbidden_claim" for f in findings)


def test_agentic_forbidden_claim_rate_zero_across_any_case():
    """对任意含 must_not_include 的样例，agentic_v1 的 forbidden_claim_rate 恒为 0。"""
    case = EvalCase(
        id="c_forbid",
        question="如何处理漏洞信息？",
        reference_answer="应直接查询内部漏洞库，不要使用谷歌或百度搜索。",
        expected_docs=["SEC-VULN-001"],
        must_include=["内部漏洞库"],
        must_not_include=["谷歌", "百度"],
    )
    output = MockRAGClient("agentic_v1", top_k=3).answer(case)
    metrics = compute_answer_metrics(case, output)
    assert metrics["forbidden_claim_rate"] == 0.0
    for kw in case.must_not_include:
        assert kw not in output.answer