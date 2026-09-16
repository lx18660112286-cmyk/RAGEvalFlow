"""成本 / 运行时指标（确定性）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragevalflow.schemas.eval_case import EvalCase
    from ragevalflow.schemas.rag_output import RAGOutput


def compute_cost_metrics(case: "EvalCase", output: "RAGOutput") -> dict[str, float]:
    """至少包含 latency_ms / input_tokens / output_tokens / estimated_cost / total_tokens。

    token 与 cost 字段在 runtime 中为 None（不可观测）时，对应指标键不写入，
    避免把 unavailable 伪装成 0。latency_ms 始终写入。
    """
    runtime = output.runtime
    metrics: dict[str, float] = {"latency_ms": float(runtime.latency_ms)}
    input_tokens = runtime.input_tokens
    output_tokens = runtime.output_tokens
    if input_tokens is not None:
        metrics["input_tokens"] = float(input_tokens)
    if output_tokens is not None:
        metrics["output_tokens"] = float(output_tokens)
    if input_tokens is not None or output_tokens is not None:
        metrics["total_tokens"] = float((input_tokens or 0) + (output_tokens or 0))
    if runtime.estimated_cost is not None:
        metrics["estimated_cost"] = float(runtime.estimated_cost)
    return metrics