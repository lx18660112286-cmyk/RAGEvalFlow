"""Pydantic schema 校验测试（阶段 1）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragevalflow.schemas import (
    CaseResult,
    Context,
    EvalCase,
    ExpectedBehavior,
    FailureFinding,
    RAGOutput,
    RuntimeInfo,
    Trace,
)
from ragevalflow.schemas.failure import FailureType


def test_eval_case_minimal():
    """最简样例：只提供 id 与 question，其余字段使用默认值。"""
    case = EvalCase(id="c1", question="问题")
    assert case.category == "general"
    assert case.reference_answer is None
    assert case.expected_docs == []
    assert case.must_include == []
    assert case.must_not_include == []
    assert case.expected_behavior is None


def test_eval_case_full():
    """完整样例：包含全部字段与 expected_behavior，且可序列化往返。"""
    data = {
        "id": "c_full",
        "category": "安全",
        "question": "数据如何加密？",
        "reference_answer": "AES-256",
        "expected_docs": ["DOC-1"],
        "must_include": ["AES-256"],
        "must_not_include": ["明文"],
        "expected_behavior": {
            "should_rewrite": True,
            "should_multi_hop": True,
            "max_retrieval_rounds": 2,
            "expected_tools": ["kb"],
            "forbidden_tools": ["web_search"],
        },
    }
    case = EvalCase.model_validate(data)
    assert case.expected_behavior is not None
    assert case.expected_behavior.should_rewrite is True
    assert case.expected_behavior.max_retrieval_rounds == 2
    # 往返：dump 后再校验仍然一致
    again = EvalCase.model_validate(case.model_dump())
    assert again.model_dump() == case.model_dump()


def test_eval_case_missing_question_fails():
    """缺少必填字段 question 时校验失败。"""
    with pytest.raises(ValidationError):
        EvalCase(id="c_bad")  # type: ignore[call-arg]


def test_expected_behavior_defaults():
    behavior = ExpectedBehavior()
    assert behavior.should_rewrite is False
    assert behavior.should_multi_hop is False
    assert behavior.max_retrieval_rounds is None
    assert behavior.expected_tools == []
    assert behavior.forbidden_tools == []


def test_rag_output_many_contexts_no_limit():
    """数据模型层不限制 contexts 条数（50 条通过校验）。"""
    contexts = [
        Context(doc_id=f"d{i}", chunk_id=f"d{i}-c0", text=f"片段 {i}", score=1.0 - i / 100, rank=i + 1)
        for i in range(50)
    ]
    output = RAGOutput(question="q", answer="a", contexts=contexts)
    assert len(output.contexts) == 50


def test_rag_output_defaults():
    """RAGOutput 空 contexts / 默认 trace / runtime。"""
    output = RAGOutput(question="q", answer="a")
    assert output.contexts == []
    assert output.trace.retrieval_rounds == 1
    assert output.trace.tools_used == []
    assert output.runtime.estimated_cost == 0.0
    assert output.config == {}


def test_context_round_trip():
    context = Context(doc_id="d1", chunk_id="c1", text="hello", score=0.9, rank=1)
    assert Context.model_validate(context.model_dump()) == context


def test_trace_and_runtime_defaults():
    trace = Trace(query_rewrite="重写", retrieval_rounds=3, used_reranker=True)
    assert trace.steps == []
    runtime = RuntimeInfo(latency_ms=123.5, input_tokens=10, output_tokens=20, estimated_cost=0.01)
    assert runtime.input_tokens == 10
    # 默认值不可变（不共享可变对象）
    assert trace.tools_used is not Trace().tools_used


def test_failure_finding_valid_and_invalid():
    finding = FailureFinding(case_id="c1", failure_type="retrieval_miss", evidence="未命中 DOC-1", severity="high")
    assert finding.failure_type == "retrieval_miss"
    assert finding.severity == "high"
    with pytest.raises(ValidationError):
        FailureFinding(case_id="c1", failure_type="unknown_type")  # type: ignore[arg-type]


def test_failure_type_literal_members():
    """13 类失败归因类型可被 Literacy 校验。"""
    expected = {
        "retrieval_miss", "bad_ranking", "incomplete_answer", "forbidden_claim",
        "missing_refusal", "wrong_refusal", "over_retrieval", "missing_multi_hop",
        "query_rewrite_missing", "forbidden_tool_called", "expected_tool_missing",
        "cost_regression", "latency_regression",
    }
    assert set(FailureType.__args__) == expected


def test_case_result_defaults():
    """CaseResult 阶段 1：metrics / failure_types 默认空。"""
    result = CaseResult(experiment_id="e1", case_id="c1", rag_output=RAGOutput(question="q", answer="a"))
    assert result.metrics == {}
    assert result.failure_types == []