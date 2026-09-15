"""评测样例（EvalCase）模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExpectedBehavior(BaseModel):
    """Agentic 行为的期望约束（阶段 2/3 的 trace 指标与归因会用到）。"""

    should_rewrite: bool = Field(default=False, description="期望系统对用户查询进行重写")
    should_multi_hop: bool = Field(default=False, description="期望系统进行多跳检索")
    max_retrieval_rounds: int | None = Field(default=None, description="期望的最大检索轮数")
    expected_tools: list[str] = Field(default_factory=list, description="期望调用的工具列表")
    forbidden_tools: list[str] = Field(default_factory=list, description="禁止调用的工具列表")


class EvalCase(BaseModel):
    """单条评测样例。"""

    id: str = Field(description="样例唯一标识（主键）")
    category: str = Field(default="general", description="样例分类，如 IT / HR / 产品 / 安全")
    question: str = Field(description="评测问题")
    reference_answer: str | None = Field(default=None, description="参考答案（可选）")
    expected_docs: list[str] = Field(default_factory=list, description="期望命中的 doc_id 列表")
    must_include: list[str] = Field(default_factory=list, description="答案必须包含的关键词/短语")
    must_not_include: list[str] = Field(default_factory=list, description="答案禁止出现的关键词/短语")
    expected_behavior: ExpectedBehavior | None = Field(default=None, description="Agentic 行为期望（可选）")