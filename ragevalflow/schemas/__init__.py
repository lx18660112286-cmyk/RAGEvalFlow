"""RAGEvalFlow 核心数据模型（Pydantic v2）。

阶段 1 仅定义结构，不包含任何指标 / runner / 归因逻辑。
"""

from ragevalflow.schemas.eval_case import EvalCase, ExpectedBehavior
from ragevalflow.schemas.rag_output import Context, Trace, RuntimeInfo, RAGOutput
from ragevalflow.schemas.experiment_result import Experiment, CaseResult
from ragevalflow.schemas.failure import FailureType, FailureFinding

__all__ = [
    "EvalCase",
    "ExpectedBehavior",
    "Context",
    "Trace",
    "RuntimeInfo",
    "RAGOutput",
    "Experiment",
    "CaseResult",
    "FailureType",
    "FailureFinding",
]