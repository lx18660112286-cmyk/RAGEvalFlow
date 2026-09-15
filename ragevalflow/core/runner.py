"""实验运行器：dataset → mock RAG → 确定性指标 → 失败归因 → case_results 落库。

阶段 3 起会写入真实失败归因（failure_types / failure_findings）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ragevalflow.analysis.failure_analyzer import analyze_failures
from ragevalflow.core.dataset import load_config_yaml, resolve_experiment_name
from ragevalflow.core.storage import Storage
from ragevalflow.integrations.rag_client import MockRAGClient
from ragevalflow.metrics import compute_all_metrics
from ragevalflow.schemas.experiment_result import CaseResult

if TYPE_CHECKING:
    from ragevalflow.schemas.rag_output import RAGOutput


class RunnerError(Exception):
    """实验运行前置条件不满足（数据库未初始化 / 数据集中无样例等）。"""


@dataclass
class ExperimentRunSummary:
    """一次实验运行的汇总结果。"""

    experiment_id: str
    name: str
    cases: int
    avg_metrics: dict[str, float] = field(default_factory=dict)
    failure_total: int = field(default=0)  # 全部样例累计失败归因条数


# 运行摘要中优先展示的指标顺序
SUMMARY_METRIC_KEYS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr",
    "expected_doc_coverage",
    "context_precision_simple",
    "must_include_coverage",
    "forbidden_claim_rate",
    "answer_context_overlap_score",
    "citation_doc_coverage",
    "missing_multi_hop",
    "over_retrieval",
    "latency_ms",
    "estimated_cost",
)


def run_experiment(
    db_path: str | Path,
    config_path: str | Path,
    experiment_name: str | None = None,
) -> ExperimentRunSummary:
    """运行一次评测实验。

    流程：
    1. 校验数据库已初始化、数据集中存在评测样例（否则抛出中文 RunnerError）；
    2. 读取 YAML 配置，解析实验名与 client 配置，创建 MockRAGClient；
    3. 对每条样例调用 client.answer()，计算全部确定性指标并做失败归因（纯内存）；
    4. 全部成功后创建 experiment，并将 case_results（含 failure_types / failure_findings）事务化写入；
    5. 返回汇总（含各指标平均值）。

    任一样例的指标计算或失败归因抛出异常，整个 run 失败、不写入任何半成品 case_results
    也未创建 experiment；重复 experiment_name 时抛出 DuplicateExperimentError，不会覆盖已有实验。
    """
    storage = Storage(db_path)
    _require_ready(storage)

    data, config_yaml = load_config_yaml(config_path)
    name = resolve_experiment_name(data, experiment_name)
    rag_version = str(data.get("rag_version") or "")

    client_config = data.get("config") or {}
    mock_mode = str(client_config.get("mock_mode") or "baseline")
    top_k = int(client_config.get("top_k") or 3)

    client = MockRAGClient(mode=mock_mode, top_k=top_k)

    # 先在内存中完成全部调用、指标计算与失败归因，实验创建成功前不落任何数据
    computed: list[tuple[str, "RAGOutput", dict[str, float], list]] = []
    for case in storage.list_cases():
        output = client.answer(case)
        metrics = compute_all_metrics(case, output)
        findings = analyze_failures(case, output, metrics)
        computed.append((case.id, output, metrics, findings))

    # 全部成功后再创建实验并写入结果
    experiment_row = storage.create_experiment(name=name, rag_version=rag_version, config_yaml=config_yaml)
    results = [
        CaseResult(
            experiment_id=experiment_row.experiment_id,
            case_id=case_id,
            rag_output=output,
            metrics=metrics,
            failure_types=[f.failure_type for f in findings],
            failure_findings=findings,
        )
        for case_id, output, metrics, findings in computed
    ]
    storage.save_case_results(results)

    return ExperimentRunSummary(
        experiment_id=experiment_row.experiment_id,
        name=name,
        cases=len(results),
        avg_metrics=_average_metrics(results),
        failure_total=sum(len(r.failure_types) for r in results),
    )


def _require_ready(storage: Storage) -> None:
    """运行前置校验，未满足时给出中文错误。"""
    tables = storage.user_table_names()
    required = {"eval_cases", "experiments", "case_results"}
    if not required.issubset(tables):
        raise RunnerError(
            "数据库未初始化（缺少所需用户表），请先执行：python -m ragevalflow.main init-db"
        )
    if storage.count_cases() == 0:
        raise RunnerError(
            "数据集中没有评测样例，请先导入数据：python -m ragevalflow.main dataset import <jsonl 文件>"
        )


def _average_metrics(results: list[CaseResult]) -> dict[str, float]:
    """按指标 key 对全部样例求平均值（含默认缺省 0）。"""
    if not results:
        return {}
    keys = set().union(*(r.metrics.keys() for r in results))
    averages = {key: sum(r.metrics.get(key, 0.0) for r in results) / len(results) for key in keys}
    return {key: round(averages[key], 6) for key in SUMMARY_METRIC_KEYS if key in averages}