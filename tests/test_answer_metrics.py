"""答案质量指标测试。"""

from __future__ import annotations

from ragevalflow.metrics.answer import compute_answer_metrics
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.rag_output import RAGOutput


def test_must_include_full_coverage():
    case = EvalCase(id="c1", question="q", must_include=["工单系统", "主管审批"])
    output = RAGOutput(question="q", answer="需要提交工单系统，并经过主管审批。")
    metrics = compute_answer_metrics(case, output)
    assert metrics["must_include_coverage"] == 1.0


def test_must_include_partial_coverage():
    case = EvalCase(id="c1", question="q", must_include=["A", "B", "C"])
    metrics = compute_answer_metrics(case, RAGOutput(question="q", answer="答案是 A。"))
    assert metrics["must_include_coverage"] == round(1 / 3, 6)


def test_forbidden_claim_rate():
    case = EvalCase(id="c1", question="q", must_not_include=["立即到账"])
    metrics = compute_answer_metrics(case, RAGOutput(question="q", answer="退款立即到账，无需等待。"))
    assert metrics["forbidden_claim_rate"] == 1.0


def test_empty_lists_return_zero():
    """must_include / must_not_include 为空 → 比例指标 0.0，不报错。"""
    case = EvalCase(id="c1", question="q")
    metrics = compute_answer_metrics(case, RAGOutput(question="q", answer="答案内容"))
    assert metrics["must_include_coverage"] == 0.0
    assert metrics["forbidden_claim_rate"] == 0.0
    assert metrics["answer_length"] == float(len("答案内容"))


def test_answer_length_and_exact_keyword():
    case = EvalCase(id="c1", question="q", must_include=["关键词"])
    metrics = compute_answer_metrics(case, RAGOutput(question="q", answer="包含关键词的答案 text"))
    assert metrics["answer_length"] == float(len("包含关键词的答案 text"))
    # 阶段 2 exact_keyword_coverage 等同于 must_include_coverage
    assert metrics["exact_keyword_coverage"] == metrics["must_include_coverage"] == 1.0