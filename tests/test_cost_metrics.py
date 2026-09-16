"""成本 / 运行时指标测试。"""

from __future__ import annotations

from ragevalflow.metrics.cost import compute_cost_metrics
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.rag_output import RAGOutput, RuntimeInfo


def test_cost_metrics_values():
    case = EvalCase(id="c1", question="q")
    output = RAGOutput(
        question="q",
        answer="a",
        runtime=RuntimeInfo(latency_ms=120.5, input_tokens=100, output_tokens=50, estimated_cost=0.0015),
    )
    metrics = compute_cost_metrics(case, output)
    assert metrics == {
        "latency_ms": 120.5,
        "input_tokens": 100.0,
        "output_tokens": 50.0,
        "estimated_cost": 0.0015,
        "total_tokens": 150.0,
    }


def test_cost_metrics_defaults():
    """token / cost 不可观测（None）时，对应指标键不写入，避免把 unavailable 伪装成 0。"""
    case = EvalCase(id="c1", question="q")
    metrics = compute_cost_metrics(case, RAGOutput(question="q", answer="a"))
    assert metrics["latency_ms"] == 0.0     # latency 始终写入
    assert "total_tokens" not in metrics
    assert "estimated_cost" not in metrics