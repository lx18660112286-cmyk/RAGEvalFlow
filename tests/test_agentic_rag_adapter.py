"""AgenticRAGClient 单元测试：真实输出 → RAGOutput 映射、trace/runtime 真实性、防泄密。

不使用真实 DEEPSEEK / LightRAG（无网络），而是注入轻量 fake 组件，验证 Adapter 的逻辑。
fake 仅模拟被测系统的"形状"（AgentResult / KnowledgeSearchResult / TraceEvent），
不引入 polaris，保证测试在任意 Python 环境可跑。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragevalflow.integrations.agentic_rag_client import (
    AgenticRAGClient,
    build_rag_output,
)
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.rag_output import RAGOutput, Trace


# ---------------------------------------------------------------------------
# 轻量 fake（模拟被测系统的输出形状）
# ---------------------------------------------------------------------------

def _status(name="SUCCESS"):
    return SimpleNamespace(value=name)


def _agent_result(
    *,
    answer: str = "这是一段答案。",
    routing_steps=None,
    tool_calls=None,
    status="SUCCESS",
):
    return SimpleNamespace(
        status=_status(status),
        answer=answer,
        citations=[],
        tool_calls=tool_calls or [],
        steps=1,
        error=None,
        trace_id="abc",
        original_query="原问题",
        routing_steps=routing_steps or [],
    )


def _tool_record(name="search_dev_knowledge", status="SUCCESS", routing=None, ms=12.3):
    return SimpleNamespace(
        name=name,
        arguments={"query": "q"},
        result_status=status,
        error=None,
        duration_ms=ms,
        sources=[],
        routing=routing,
    )


def _routing_step(tool_query="原问题", intent="factual", strategy="focused"):
    return SimpleNamespace(
        step_index=0,
        tool_call_id="t1",
        original_user_query="原问题",
        tool_query=tool_query,
        intent=SimpleNamespace(value=intent),
        strategy=SimpleNamespace(value=strategy),
        reason="r",
        fallback_used=False,
    )


def _chunk(source_name, chunk_id, content="正文", reference_id="r"):
    return SimpleNamespace(
        chunk_id=chunk_id,
        content=content,
        reference_id=reference_id,
        source_name=source_name,
        source_path=None,
    )


def _recorded(*results):
    return [SimpleNamespace(evidence=SimpleNamespace(chunks=list(cs))) for cs in results]


def _usage_event(input_tokens=None, output_tokens=None):
    return SimpleNamespace(
        event_type=SimpleNamespace(value="MODEL_CALL_COMPLETED"),
        attributes={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )


# ---------------------------------------------------------------------------
# 纯映射函数测试
# ---------------------------------------------------------------------------

def test_build_rag_output_maps_answer_and_contexts():
    out = build_rag_output(
        question="原问题",
        original_query="原问题",
        agent_result=_agent_result(answer="正确答案"),
        recorded_results=_recorded(
            [_chunk("api_auth.md", "c1", "内容A", "r1")],
            [_chunk("sub/dir/deployment.md", "c2", "内容B", "r2")],
        ),
        trace_events=[],
        model="deepseek-chat",
        latency_ms=123.456,
        inner_config={"client": "agentic_rag"},
    )
    assert out.answer == "正确答案"
    assert [c.doc_id for c in out.contexts] == ["api_auth.md", "deployment.md"]
    assert [c.rank for c in out.contexts] == [1, 2]
    assert [c.text for c in out.contexts] == ["内容A", "内容B"]
    assert [c.chunk_id for c in out.contexts] == ["c1", "c2"]
    assert all(c.score is None for c in out.contexts)  # 未提供 score → None，非 0
    assert out.runtime.latency_ms == pytest.approx(123.456, abs=1e-3)
    assert out.config["client"] == "agentic_rag"


def test_build_rag_output_normalizes_doc_id_to_basename():
    out = build_rag_output(
        question="q", original_query="q",
        agent_result=_agent_result(),
        recorded_results=_recorded([_chunk("docs/api_auth.md", "c1", "c", "r1")]),
        trace_events=[], model="", latency_ms=1.0,
    )
    assert out.contexts[0].doc_id == "api_auth.md"


def test_build_rag_output_dedupes_same_chunk_across_rounds():
    out = build_rag_output(
        question="q", original_query="q",
        agent_result=_agent_result(),
        recorded_results=_recorded([_chunk("api_auth.md", "c1", "第一次", "r1")], [_chunk("api_auth.md", "c1", "第二次", "r1")]),
        trace_events=[], model="", latency_ms=1.0,
    )
    assert len(out.contexts) == 1
    assert out.contexts[0].text == "第一次"  # 保留首次出现


def test_build_rag_output_trace_from_real_routing():
    routing = [_routing_step(tool_query="原问题", strategy="focused")]
    tool = _tool_record(routing=_routing_step(tool_query="原问题"))
    out = build_rag_output(
        question="原问题", original_query="原问题",
        agent_result=_agent_result(routing_steps=routing, tool_calls=[tool]),
        recorded_results=[], trace_events=[], model="", latency_ms=1.0,
    )
    assert out.trace.retrieval_rounds == 1
    assert out.trace.tools_used == ["search_dev_knowledge"]  # 真实工具名
    assert out.trace.used_reranker is False   # 系统确未实现 → false，不伪造
    assert out.trace.used_reflection is False
    # tool_query == 原文 → 未改写
    assert out.trace.query_rewrite is None


def test_build_rag_output_rewrite_detected_when_tool_query_differs():
    routing = [_routing_step(tool_query="Mercury access token TTL 有效期", strategy="focused")]
    out = build_rag_output(
        question="原问题", original_query="原问题",
        agent_result=_agent_result(routing_steps=routing),
        recorded_results=[], trace_events=[], model="", latency_ms=1.0,
    )
    assert out.trace.query_rewrite == "Mercury access token TTL 有效期"


def test_build_rag_output_tokens_from_real_usage_events():
    events = [
        _usage_event(input_tokens=100, output_tokens=50),
        _usage_event(input_tokens=30, output_tokens=20),
    ]
    out = build_rag_output(
        question="q", original_query="q",
        agent_result=_agent_result(), recorded_results=[],
        trace_events=events, model="deepseek-chat", latency_ms=1.0,
    )
    assert out.runtime.input_tokens == 130
    assert out.runtime.output_tokens == 70


def test_build_rag_output_unavailable_tokens_are_none():
    # 无任何 usage 事件 → tokens 保持 None，不给 0
    out = build_rag_output(
        question="q", original_query="q",
        agent_result=_agent_result(), recorded_results=[],
        trace_events=[SimpleNamespace(event_type=SimpleNamespace(value="AGENT_COMPLETED"), attributes={})],
        model="deepseek-chat", latency_ms=1.0,
    )
    assert out.runtime.input_tokens is None
    assert out.runtime.output_tokens is None


def test_build_rag_output_estimated_cost_unavailable_without_pricing():
    events = [_usage_event(input_tokens=100, output_tokens=50)]
    out_no_pricing = build_rag_output(
        question="q", original_query="q",
        agent_result=_agent_result(), recorded_results=[], trace_events=events,
        model="deepseek-chat", latency_ms=1.0, pricing=None,
    )
    assert out_no_pricing.runtime.estimated_cost is None  # 不伪造成本

    out_priced = build_rag_output(
        question="q", original_query="q",
        agent_result=_agent_result(), recorded_results=[], trace_events=events,
        model="deepseek-chat", latency_ms=1.0,
        pricing={"input_per_1m_tokens": 1.0, "output_per_1m_tokens": 2.0},
    )
    assert out_priced.runtime.estimated_cost == pytest.approx((100 * 1.0 + 50 * 2.0) / 1e6)


# ---------------------------------------------------------------------------
# AgenticRAGClient（注入 fake builder）端到端 + 防泄密
# ---------------------------------------------------------------------------

class _FakeAdapter:
    initialized = False

    async def initialize(self):
        self.initialized = True

    async def close(self):
        pass


class _FakeOrchestrator:
    def __init__(self, result_builder):
        self.calls: list[str] = []
        self._result_builder = result_builder

    async def run(self, message: str):
        self.calls.append(message)
        return self._result_builder()


class _FakeBuilt:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        #: 记录传入 orchestrator 的原文，供防泄密断言
        self._orchestrator = orchestrator
        self.adapter = _FakeAdapter()
        self.tracer = None


def _make_client(agent_result_builder):
    def builder(settings, working_dir, recorder):
        return _FakeBuilt(_FakeOrchestrator(agent_result_builder))

    return AgenticRAGClient(builder=builder, model="deepseek-chat"), builder


def test_client_answer_only_passes_question_to_sut():
    """防泄密：被测系统输入里绝不能出现任何 ground-truth 字段。"""
    sent = []

    def result_builder():
        return _agent_result()

    def builder(settings, working_dir, recorder):
        orch = _FakeOrchestrator(result_builder)
        sent.append(orch)
        return _FakeBuilt(orch)

    client = AgenticRAGClient(builder=builder, model="deepseek-chat")
    case = EvalCase(
        id="case_001",
        question="Mercury access token 有效期是多少？",
        reference_answer="这是参考答案。",
        expected_docs=["api_auth.md"],
        must_include=["30 分钟"],
        must_not_include=["15 分钟"],
    )
    out = client.answer(case)
    client.close()

    orch = sent[0]
    assert len(orch.calls) == 1
    payload = orch.calls[0]
    # 传给 SUT 的只有 question
    assert payload == case.question
    for secret in ["这是参考答案", "api_auth.md", "30 分钟", "15 分钟"]:
        assert secret not in payload
    # 返回的 RAGOutput 也不携带 ground-truth
    assert "答案" not in " ".join(f"{c.doc_id} {c.text}" for c in out.contexts)  # 空 context 说明无泄密
    assert out.question == case.question


def test_client_answer_roundtrip_maps_output_and_runtime():
    def result_builder():
        return _agent_result(answer="真实答案", tool_calls=[_tool_record()])

    client = _make_client(result_builder)[0]
    case = EvalCase(id="c1", question="q")
    out = client.answer(case)
    client.close()

    assert isinstance(out, RAGOutput)
    assert out.answer == "真实答案"
    assert out.trace.tools_used == ["search_dev_knowledge"]
    assert out.trace.used_reranker is False
    assert out.runtime.latency_ms >= 0.0


def test_client_answer_no_routing_keeps_rounds_unchanged():
    def result_builder():
        return _agent_result()  # 无 tool_calls / routing_steps

    client = _make_client(result_builder)[0]
    out = client.answer(EvalCase(id="c1", question="q"))
    client.close()
    assert out.trace.retrieval_rounds == 1
    assert out.trace.tools_used == []


def test_client_answer_status_mapped_into_config():
    def result_builder():
        return _agent_result(status="MAX_STEPS_EXCEEDED")

    client = _make_client(result_builder)[0]
    out = client.answer(EvalCase(id="c1", question="q"))
    client.close()
    assert out.config.get("status") == "MAX_STEPS_EXCEEDED"