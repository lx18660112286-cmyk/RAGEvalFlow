"""失败归因与回归检测（阶段 3）。

全部为 deterministic 规则：
- 不调用 LLM judge、不联网、不依赖 Ragas / DeepEval；
- 失败归因复用 schemas.failure 中的 13 类 FailureType。
"""

from __future__ import annotations

from ragevalflow.analysis.failure_analyzer import analyze_failures
from ragevalflow.analysis.regression import (
    ComparisonError,
    RegressionReport,
    RegressionThresholds,
    compare_experiments,
)

__all__ = [
    "analyze_failures",
    "compare_experiments",
    "RegressionReport",
    "RegressionThresholds",
    "ComparisonError",
]