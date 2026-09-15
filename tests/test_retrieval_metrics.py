"""检索质量指标测试。"""

from __future__ import annotations

from ragevalflow.metrics.retrieval import compute_retrieval_metrics
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.rag_output import Context, RAGOutput


def _ctx(doc_id: str, rank: int) -> Context:
    return Context(doc_id=doc_id, chunk_id=f"{doc_id}-c0", text="内容", score=1.0, rank=rank)


def test_hit_at_k_and_mrr_with_second_rank_hit():
    """期望文档在 rank 2：hit_at_1=0、hit_at_3=1、mrr=0.5。"""
    case = EvalCase(id="c1", question="q", expected_docs=["D1"])
    output = RAGOutput(question="q", answer="a", contexts=[_ctx("X1", 1), _ctx("D1", 2)])
    metrics = compute_retrieval_metrics(case, output)
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 0.5


def test_full_hit_metrics():
    case = EvalCase(id="c1", question="q", expected_docs=["D1", "D2"])
    output = RAGOutput(question="q", answer="a", contexts=[_ctx("D1", 1), _ctx("D2", 2)])
    metrics = compute_retrieval_metrics(case, output)
    assert metrics["hit_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["expected_doc_coverage"] == 1.0
    assert metrics["context_precision_simple"] == 1.0


def test_miss_all_zero():
    case = EvalCase(id="c1", question="q", expected_docs=["D1"])
    output = RAGOutput(question="q", answer="a", contexts=[_ctx("X1", 1), _ctx("X2", 2)])
    metrics = compute_retrieval_metrics(case, output)
    assert metrics["hit_at_3"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["expected_doc_coverage"] == 0.0
    assert metrics["context_precision_simple"] == 0.0


def test_empty_expected_docs_returns_zero():
    """expected_docs 为空 → 全部指标 0.0，不报错。"""
    case = EvalCase(id="c1", question="q")
    output = RAGOutput(question="q", answer="a", contexts=[_ctx("X1", 1)])
    metrics = compute_retrieval_metrics(case, output)
    assert all(value == 0.0 for value in metrics.values())


def test_no_contexts_returns_zero():
    case = EvalCase(id="c1", question="q", expected_docs=["D1"])
    metrics = compute_retrieval_metrics(case, RAGOutput(question="q", answer="a"))
    assert metrics["context_precision_simple"] == 0.0
    assert metrics["expected_doc_coverage"] == 0.0


def test_many_contexts_still_computable():
    """contexts 很多（50 条）仍可计算；期望文档在 rank 25。"""
    case = EvalCase(id="c1", question="q", expected_docs=["D1"])
    contexts = [_ctx(f"X{i}", i + 1) for i in range(50)]
    contexts[24] = _ctx("D1", 25)
    metrics = compute_retrieval_metrics(
        case, RAGOutput(question="q", answer="a", contexts=contexts)
    )
    assert metrics["hit_at_1"] == 0.0 and metrics["hit_at_5"] == 0.0
    assert metrics["mrr"] == round(1 / 25, 6)
    assert metrics["expected_doc_coverage"] == 1.0
    assert metrics["context_precision_simple"] == round(1 / 50, 6)