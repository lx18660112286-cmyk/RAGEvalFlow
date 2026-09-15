"""实验运行器测试：端到端 run、重复名失败、前置校验失败。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragevalflow.core import dataset as ds_lib
from ragevalflow.core.runner import RunnerError, run_experiment
from ragevalflow.core.storage import DuplicateExperimentError, Storage

RUN_CONFIG = (
    "experiment_name: base_test\n"
    "rag_version: mock-baseline-v0\n"
    "config:\n"
    "  client: mock\n"
    "  mock_mode: baseline\n"
    "  top_k: 3\n"
)

CASES = [
    {"id": "c1", "category": "IT", "question": "如何申请 VPN？", "expected_docs": ["IT-VPN-001"], "must_include": ["工单系统"]},
    {"id": "c2", "category": "产品", "question": "NovaSearch 刷新间隔？", "expected_docs": ["NOVASEARCH-INDEX-002"], "must_include": ["5 分钟"]},
    {"id": "c3", "category": "HR", "question": "外包员工年假折算？", "expected_docs": ["HR-LEAVE-003"], "must_include": ["折算"]},
]


@pytest.fixture()
def env(tmp_path: Path):
    """构造已初始化的库 + 已导入的数据集 + 配置文件。"""
    db_path = tmp_path / "run.db"
    storage = Storage(db_path)
    storage.init_db()
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CASES) + "\n", encoding="utf-8")
    ds_lib.import_cases(storage, jsonl)
    config = tmp_path / "run.yaml"
    config.write_text(RUN_CONFIG, encoding="utf-8")
    return db_path, config


def test_run_experiment_end_to_end(env):
    db_path, config = env
    summary = run_experiment(db_path, config)

    assert summary.name == "base_test"
    assert summary.cases == len(CASES)
    assert summary.experiment_id.startswith("exp_")
    # 平均值应覆盖关键指标
    assert "hit_at_3" in summary.avg_metrics
    assert "must_include_coverage" in summary.avg_metrics
    assert "latency_ms" in summary.avg_metrics

    # case_results 已写入且与 experiment_id 关联；failure_types 为空
    storage = Storage(db_path)
    rows = storage.list_experiments()
    assert len(rows) == 1
    experiment_id = rows[0].experiment_id
    assert storage.count_case_results(experiment_id) == len(CASES)
    results = storage.list_case_results(experiment_id)
    assert sorted(r.case_id for r in results) == ["c1", "c2", "c3"]
    assert all("must_include_coverage" in r.metrics for r in results)
    # 阶段 3：失败归因真实写入 case_results
    assert any(r.failure_types for r in results)
    assert any(r.failure_findings for r in results)


def test_run_experiment_duplicate_name_fails(env):
    """同一配置重复运行 → 同名实验报错，不允许覆盖。"""
    db_path, config = env
    run_experiment(db_path, config)
    with pytest.raises(DuplicateExperimentError):
        run_experiment(db_path, config)
    # 仍只有 1 个实验
    assert len(Storage(db_path).list_experiments()) == 1


def test_run_experiment_name_override(env):
    db_path, config = env
    summary = run_experiment(db_path, config, experiment_name="custom_name")
    assert summary.name == "custom_name"


def test_run_without_dataset_fails_chinese_error(tmp_path: Path):
    """数据集为空 → 中文错误。"""
    db_path = tmp_path / "empty.db"
    storage = Storage(db_path)
    storage.init_db()  # 只初始化，不导入数据
    config = tmp_path / "run.yaml"
    config.write_text(RUN_CONFIG, encoding="utf-8")
    with pytest.raises(RunnerError) as exc_info:
        run_experiment(db_path, config)
    assert "没有评测样例" in str(exc_info.value)


def test_run_without_db_init_fails_chinese_error(tmp_path: Path):
    """数据库文件不存在（未 init-db）→ 中文错误。"""
    db_path = tmp_path / "not_initialized.db"
    config = tmp_path / "run.yaml"
    config.write_text(RUN_CONFIG, encoding="utf-8")
    with pytest.raises(RunnerError) as exc_info:
        run_experiment(db_path, config)
    assert "数据库未初始化" in str(exc_info.value)


def test_run_writes_detected_failure_types(env):
    """阶段 3：failure_types 不再固定为空，写入真实失败归因（低忠实度等）。"""
    db_path, config = env
    summary = run_experiment(db_path, config)
    storage = Storage(db_path)
    results = storage.list_case_results(summary.experiment_id)
    all_types = [t for r in results for t in r.failure_types]
    # mock baseline 答案与检索上下文重叠度过低 → 至少能检测到失败
    assert all_types, "baseline 实验应产生失败归因"
    assert "incomplete_answer" in all_types