"""成本 / 运行时指标（确定性）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def compute_cost_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """至少包含 latency_ms / input_tokens / output_tokens / estimated_cost / total_tokens。"""
    runtime = output.runtime
    return {
        "latency_ms": float(runtime.latency_ms),
        "input_tokens": float(runtime.input_tokens),
        "output_tokens": float(runtime.output_tokens),
        "estimated_cost": float(runtime.estimated_cost),
        "total_tokens": float(runtime.input_tokens + runtime.output_tokens),
    }