"""RAG 系统输出（RAGOutput）相关模型。

数据模型层对 contexts 条数不做任何限制；展示层（阶段 4 Dashboard）可自行截断。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Context(BaseModel):
    """检索命中的单个上下文片段。

    真实系统（如 Agentic-RAG）可能不提供 chunk_id / score 等字段：
    score / chunk_id 允许为 None，以区分"系统确实返回了 0 分"与"未观测到"。
    """

    doc_id: str = Field(description="所属文档 ID")
    chunk_id: str | None = Field(default=None, description="块 ID（未提供则为 None）")
    text: str = Field(default="", description="上下文文本")
    score: float | None = Field(default=None, description="检索得分（系统未提供则为 None）")
    rank: int = Field(default=0, description="在检索结果中的排序（从 1 开始）")


class Trace(BaseModel):
    """Agentic-RAG 执行轨迹（阶段 3 指标的核心输入）。"""

    query_rewrite: str | None = Field(default=None, description="重写后的查询（未重写则为 None）")
    retrieval_rounds: int = Field(default=1, description="检索轮数")
    used_reranker: bool = Field(default=False, description="是否使用了重排序器")
    used_reflection: bool = Field(default=False, description="是否使用了反思/自我校验")
    steps: list[dict] = Field(default_factory=list, description="执行步骤明细（自由结构）")
    tools_used: list[str] = Field(default_factory=list, description="实际调用的工具列表")


class RuntimeInfo(BaseModel):
    """运行时信息（延迟 / token / 成本）。

    input_tokens / output_tokens / estimated_cost 允许为 None：
    - tokens 仅在真实 provider 返回 usage 时才填充（Agentic-RAG 只有 Agent-LLM 的 usage；
      LightRAG 内核的 token 不可观测，故报 None 而非估算）。
    - estimated_cost 仅在 model 已知 + token 已知 + 提供定价时才计算，否则 None（不伪造成本）。
    """

    latency_ms: float = Field(default=0.0, description="端到端延迟（毫秒）")
    input_tokens: int | None = Field(default=None, description="输入 token 数（未观测到则为 None）")
    output_tokens: int | None = Field(default=None, description="输出 token 数（未观测到则为 None）")
    estimated_cost: float | None = Field(default=None, description="估算成本（元，未知则为 None）")


class RAGOutput(BaseModel):
    """一次调用 RAG 系统的完整输出。"""

    question: str = Field(description="原始问题")
    answer: str = Field(description="系统生成的答案")
    contexts: list[Context] = Field(default_factory=list, description="检索上下文，条数不限")
    trace: Trace = Field(default_factory=Trace, description="Agentic 执行轨迹")
    runtime: RuntimeInfo = Field(default_factory=RuntimeInfo, description="运行时信息")
    config: dict = Field(default_factory=dict, description="本次调用使用的系统配置")