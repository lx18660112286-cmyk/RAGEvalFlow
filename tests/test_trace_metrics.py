"""Agentic-RAG trace 指标测试。"""

from __future__ import annotations

from ragevalflow.metrics.trace import compute_trace_metrics
from ragevalflow.schemas.eval_case import EvalCase, ExpectedBehavior
from ragevalflow.schemas.rag_output import RAGOutput, Trace


def test_missing_multi_hop_when_rounds_too_low():
    case = EvalCase(id="c1", question="q", expected_behavior=ExpectedBehavior(should_multi_hop=True))
    output = RAGOutput(question="q", answer="a", trace=Trace(retrieval_rounds=1))
    metrics = compute_trace_metrics(case, output)
    assert metrics["missing_multi_hop"] == 1.0


def test_no_missing_multi_hop_when_two_rounds():
    case = EvalCase(id="c1", question="q", expected_behavior=ExpectedBehavior(should_multi_hop=True))
    output = RAGOutput(question="q", answer="a", trace=Trace(retrieval_rounds=2))
    metrics = compute_trace_metrics(case, output)
    assert metrics["missing_multi_hop"] == 0.0


def test_over_retrieval_when_exceeding_limit():
    case = EvalCase(id="c1", question="q", expected_behavior=ExpectedBehavior(max_retrieval_rounds=2))
    output = RAGOutput(question="q", answer="a", trace=Trace(retrieval_rounds=3))
    metrics = compute_trace_metrics(case, output)
    assert metrics["over_retrieval"] == 1.0


def test_flags_from_trace():
    case = EvalCase(id="c1", question="q")
    output = RAGOutput(
        question="q",
        answer="a",
        trace=Trace(query_rewrite="改写", retrieval_rounds=2, used_reranker=True, used_reflection=True),
    )
    metrics = compute_trace_metrics(case, output)
    assert metrics["rewrite_used"] == 1.0
    assert metrics["reranker_used"] == 1.0
    assert metrics["reflection_used"] == 1.0
    assert metrics["retrieval_rounds"] == 2.0


def test_expected_tool_coverage():
    case = EvalCase(id="c1", question="q", expected_behavior=ExpectedBehavior(expected_tools=["kb", "db"]))
    output = RAGOutput(question="q", answer="a", trace=Trace(tools_used=["kb"]))
    metrics = compute_trace_metrics(case, output)
    assert metrics["expected_tool_coverage"] == 0.5


def test_forbidden_tool_rate_from_tools_and_steps():
    case = EvalCase(id="c1", question="q", expected_behavior=ExpectedBehavior(forbidden_tools=["web_search"]))
    # tools_used 中出现
    out1 = RAGOutput(question="q", answer="a", trace=Trace(tools_used=["retriever", "web_search"]))
    assert compute_trace_metrics(case, out1)["forbidden_tool_rate"] == 1.0
    # steps 的 tool 字段中出现
    out2 = RAGOutput(question="q", answer="a", trace=Trace(tools_used=["retriever"], steps=[{"tool": "web_search"}]))
    assert compute_trace_metrics(case, out2)["forbidden_tool_rate"] == 1.0
    # 均未出现 → 0.0
    out3 = RAGOutput(question="q", answer="a", trace=Trace(tools_used=["retriever"]))
    assert compute_trace_metrics(case, out3)["forbidden_tool_rate"] == 0.0


def test_empty_behavior_defaults():
    """无 expected_behavior → 无多跳/超限/工具约束，全部 0；retrieval_rounds 取 trace 值。"""
    case = EvalCase(id="c1", question="q")
    metrics = compute_trace_metrics(case, RAGOutput(question="q", answer="a", trace=Trace(retrieval_rounds=1)))
    assert metrics["retrieval_rounds"] == 1.0
    assert metrics["missing_multi_hop"] == 0.0
    assert metrics["over_retrieval"] == 0.0
    assert metrics["expected_tool_coverage"] == 0.0
    assert metrics["forbidden_tool_rate"] == 0.0