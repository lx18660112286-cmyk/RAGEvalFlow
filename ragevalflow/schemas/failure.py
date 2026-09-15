"""失败归因类型与归因条目（阶段 3 实现判定逻辑，本阶段仅定义结构）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 13 类失败归因类型（failure taxonomy）
FailureType = Literal[
    "retrieval_miss",          # 检索未命中期望文档
    "bad_ranking",             # 期望文档被检索到但排序靠后
    "incomplete_answer",       # 答案不完整（缺少必须内容）
    "forbidden_claim",         # 答案包含禁止内容
    "missing_refusal",         # 应拒绝回答却未拒绝
    "wrong_refusal",           # 不应拒绝却被拒绝
    "over_retrieval",          # 检索轮数超过上限
    "missing_multi_hop",       # 应为多跳却未多跳
    "query_rewrite_missing",   # 应重写查询却未重写
    "forbidden_tool_called",   # 调用了禁止工具
    "expected_tool_missing",   # 期望工具未被调用
    "cost_regression",         # 成本回退
    "latency_regression",      # 延迟回退
]


class FailureFinding(BaseModel):
    """单条失败归因结论（阶段 3 由 deterministic 规则生成）。"""

    case_id: str = Field(description="归属样例 ID")
    failure_type: FailureType = Field(description="失败类型")
    evidence: str = Field(default="", description="证据/说明（规则判定后的可读描述）")
    message: str = Field(default="", description="失败原因的可读说明（reason）")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium", description="严重程度（critical/high/medium/low）"
    )
    metric_values: dict[str, float] = Field(
        default_factory=dict, description="触发该失败的关键指标快照（供报告与调试）"
    )