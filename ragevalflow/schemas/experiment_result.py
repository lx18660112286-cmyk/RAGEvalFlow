"""实验（Experiment）与单案例结果（CaseResult）模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from ragevalflow.schemas.failure import FailureFinding
from ragevalflow.schemas.rag_output import RAGOutput


class Experiment(BaseModel):
    """一次评测实验。

    config 字段存放 YAML 解析后的配置字典；
    持久化层（experiments.config 列）额外保存完整 YAML 原文。
    """

    experiment_id: str = Field(description="实验 ID，格式 exp_YYYYMMDD_HHMMSS_随机后缀")
    name: str = Field(description="实验名（唯一，重复创建报错）")
    rag_version: str = Field(default="", description="被评测系统/RAG 版本")
    config: dict = Field(default_factory=dict, description="实验配置（解析后的字典）")
    created_at: datetime = Field(description="创建时间")


class CaseResult(BaseModel):
    """单条样例在某个实验下的结果。

    阶段 1 仅定义结构：metrics 与 failure_types 为空，由阶段 2/3 填充。
    """

    experiment_id: str = Field(description="所属实验 ID")
    case_id: str = Field(description="所属样例 ID（对应 eval_cases.id）")
    rag_output: RAGOutput = Field(description="RAG 系统输出")
    metrics: dict[str, float] = Field(default_factory=dict, description="确定性指标（阶段 2 填充）")
    failure_types: list[str] = Field(default_factory=list, description="失败归因类型（阶段 3 填充，仅类型名）")
    failure_findings: list[FailureFinding] = Field(
        default_factory=list, description="失败归因详情（阶段 3 填充，含 evidence/message/severity）"
    )