"""doc_id / chunk_id normalization 专项测试。

Agentic-RAG 检索是 chunk-level，expected_docs 是 document-level。
规则：doc_id = basename(source_name)（已是稳定文件名语义）；成对规范化 + 去重。
"""

from __future__ import annotations

from ragevalflow.integrations.agentic_rag_client import build_rag_output


class _Chunk:
    def __init__(self, source_name, chunk_id, content="正文", reference_id=None):
        self.source_name = source_name
        self.chunk_id = chunk_id
        self.content = content
        self.reference_id = reference_id


class _E:
    def __init__(self, chunks):
        self.chunks = chunks


class _Res:
    def __init__(self, chunks):
        self.evidence = _E(chunks)


class _AgentResult:
    status = type("S", (), {"value": "SUCCESS"})()
    answer = "答案"
    citations = []
    tool_calls = []
    steps = 1
    error = None
    trace_id = None
    original_query = "q"
    routing_steps = []
    routing = None


def _map(chunks_per_round):
    return build_rag_output(
        question="q",
        original_query="q",
        agent_result=_AgentResult(),
        recorded_results=[_Res(cs) for cs in chunks_per_round],
        trace_events=[],
        model="",
        latency_ms=1.0,
    )


def test_full_path_source_normalizes_to_basename():
    out = _map([[_Chunk("repos/mercury/docs/api_auth.md", "c1")]])
    assert out.contexts[0].doc_id == "api_auth.md"


def test_windows_path_source_normalizes_to_basename():
    out = _map([[_Chunk("C:\\docs\\deployment\\deployment.md", "c1")]])
    assert out.contexts[0].doc_id == "deployment.md"


def test_missing_source_falls_back_to_chunk_id():
    out = _map([[_Chunk(None, "chunk-99")]])
    assert out.contexts[0].doc_id == "chunk-99"
    assert out.contexts[0].chunk_id == "chunk-99"


def test_missing_source_and_chunk_falls_back_to_rank():
    out = _map([[_Chunk(None, None)]])
    assert out.contexts[0].doc_id == "1"
    assert out.contexts[0].chunk_id is None


def test_duplicate_chunk_across_rounds_is_deduped():
    out = _map(
        [
            [_Chunk("api_auth.md", "c1", "第一轮"), _Chunk("api_auth.md", "c2")],
            [_Chunk("api_auth.md", "c1", "第二轮重复")],
        ]
    )
    assert len(out.contexts) == 2
    assert out.contexts[0].doc_id == "api_auth.md"
    # 保留首次出现，且 rank 连续
    assert out.contexts[0].text == "第一轮"
    assert [c.rank for c in out.contexts] == [1, 2]


def test_score_is_none_not_zero_when_unavailable():
    """Agentic-RAG 不提供 score → score 应为 None，而不是被伪造为 0。"""
    out = _map([[_Chunk("api_auth.md", "c1")]])
    assert out.contexts[0].score is None


def test_chunk_ids_preserved_for_stable_matching():
    out = _map([[_Chunk("incident_runbook.md", "chunk_x1"), _Chunk("deployment.md", "chunk_y2")]])
    assert [c.chunk_id for c in out.contexts] == ["chunk_x1", "chunk_y2"]