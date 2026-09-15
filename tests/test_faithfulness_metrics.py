"""忠实度 / 接地性简单指标测试。"""

from __future__ import annotations

from ragevalflow.metrics.faithfulness import compute_faithfulness_metrics
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.rag_output import Context, RAGOutput


def test_overlap_full_when_answer_from_context():
    text = "AES-256 加密用于静态数据保护，密钥由 KMS 托管。"
    case = EvalCase(id="c1", question="q")
    output = RAGOutput(question="q", answer=text, contexts=[Context(doc_id="D1", chunk_id="c", text=text)])
    metrics = compute_faithfulness_metrics(case, output)
    assert metrics["answer_context_overlap_score"] == 1.0


def test_overlap_zero_when_unrelated():
    case = EvalCase(id="c1", question="q")
    output = RAGOutput(
        question="q",
        answer="完全无关的内容abcxyz",
        contexts=[Context(doc_id="D1", chunk_id="c", text="今天天气晴朗阳光明媚")],
    )
    metrics = compute_faithfulness_metrics(case, output)
    assert metrics["answer_context_overlap_score"] == 0.0


def test_overlap_zero_without_contexts():
    case = EvalCase(id="c1", question="q")
    metrics = compute_faithfulness_metrics(case, RAGOutput(question="q", answer="答案内容"))
    assert metrics["answer_context_overlap_score"] == 0.0


def test_citation_doc_coverage():
    case = EvalCase(id="c1", question="q", expected_docs=["D1", "D2"])
    output = RAGOutput(
        question="q",
        answer="详见 D1。",
        contexts=[
            Context(doc_id="D1", chunk_id="c1", text="内容1"),
            Context(doc_id="D2", chunk_id="c2", text="内容2"),
        ],
    )
    metrics = compute_faithfulness_metrics(case, output)
    assert metrics["citation_doc_coverage"] == 0.5  # D1 被引用，D2 未出现


def test_citation_zero_without_cited_doc_ids():
    case = EvalCase(id="c1", question="q", expected_docs=["D1"])
    output = RAGOutput(
        question="q",
        answer="答案未标注引用来源",
        contexts=[Context(doc_id="D1", chunk_id="c", text="内容")],
    )
    metrics = compute_faithfulness_metrics(case, output)
    assert metrics["citation_doc_coverage"] == 0.0


def test_empty_expected_docs_returns_zero():
    case = EvalCase(id="c1", question="q")
    metrics = compute_faithfulness_metrics(case, RAGOutput(question="q", answer="答案"))
    assert metrics["citation_doc_coverage"] == 0.0