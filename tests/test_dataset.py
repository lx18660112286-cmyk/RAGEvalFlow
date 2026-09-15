"""数据集导入导出测试：事务语义 / upsert / round-trip / dry-run / 实验创建。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ragevalflow.core import dataset as ds_lib
from ragevalflow.core.storage import DuplicateExperimentError, Storage
from ragevalflow.schemas.eval_case import EvalCase

SAMPLE_CASES = [
    {"id": "c1", "category": "IT", "question": "如何申请 VPN？"},
    {"id": "c2", "category": "HR", "question": "年假几天？", "must_include": ["10 天"]},
    {
        "id": "c3",
        "category": "产品",
        "question": "NovaSearch 刷新间隔？",
        "expected_behavior": {"should_multi_hop": True},
    },
]


def _write_jsonl(path: Path, cases: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def test_import_valid(storage: Storage, tmp_path: Path):
    path = _write_jsonl(tmp_path / "a.jsonl", SAMPLE_CASES)
    stats = ds_lib.import_cases(storage, path)
    assert stats.total == 3 and stats.inserted == 3 and stats.updated == 0
    assert storage.count_cases() == 3


def test_import_repeat_is_upsert_no_count_growth(storage: Storage, tmp_path: Path):
    """重复导入同一文件：条数不增加，全部变为更新。"""
    path = _write_jsonl(tmp_path / "a.jsonl", SAMPLE_CASES)
    ds_lib.import_cases(storage, path)
    stats = ds_lib.import_cases(storage, path)
    assert stats.total == 3 and stats.inserted == 0 and stats.updated == 3
    assert storage.count_cases() == 3


def test_import_upsert_updates_fields(storage: Storage, tmp_path: Path):
    """同名 id 重复导入时覆盖已有字段。"""
    path = _write_jsonl(tmp_path / "a.jsonl", SAMPLE_CASES)
    ds_lib.import_cases(storage, path)
    changed = [dict(SAMPLE_CASES[0], question="新版问题：VPN 申请流程")]
    path2 = _write_jsonl(tmp_path / "b.jsonl", changed)
    stats = ds_lib.import_cases(storage, path2)
    assert stats.inserted == 0 and stats.updated == 1
    cases = {c.id: c for c in storage.list_cases()}
    assert cases["c1"].question == "新版问题：VPN 申请流程"


def test_import_invalid_aborts_without_writing(storage: Storage, tmp_path: Path):
    """含非法样例时整体失败，数据库零修改（事务语义）。"""
    path = _write_jsonl(tmp_path / "a.jsonl", SAMPLE_CASES)
    ds_lib.import_cases(storage, path)  # 先导入 3 条合法数据

    bad = [
        dict(SAMPLE_CASES[0], question="即使合法样例变了也不应写入"),  # 首条合法但被修改
        {"id": "c_bad"},  # 缺少 question → 非法
    ]
    bad_path = _write_jsonl(tmp_path / "bad.jsonl", bad)
    with pytest.raises(ds_lib.DatasetValidationError) as exc_info:
        ds_lib.import_cases(storage, bad_path)
    message = str(exc_info.value)
    assert "第 2 行" in message and "question" in message
    # 数据库零修改：仍为 3 条，且第一条的 question 未被改动
    cases = {c.id: c for c in storage.list_cases()}
    assert len(cases) == 3
    assert cases["c1"].question == "如何申请 VPN？"


def test_import_missing_file_raises(storage: Storage, tmp_path: Path):
    with pytest.raises(ds_lib.DatasetFileError):
        ds_lib.import_cases(storage, tmp_path / "not_exist.jsonl")


def test_import_dry_run_writes_nothing(storage: Storage, tmp_path: Path):
    path = _write_jsonl(tmp_path / "a.jsonl", SAMPLE_CASES)
    stats = ds_lib.import_cases(storage, path, dry_run=True)
    assert stats.total == 3 and stats.inserted == 0 and stats.updated == 0
    assert storage.count_cases() == 0


# ---------------------------------------------------------------------------
# export / round-trip
# ---------------------------------------------------------------------------

def test_export_round_trip(storage: Storage, tmp_path: Path):
    """导出后再导入：语义等价（校验通过、条数一致、内容一致）。"""
    path = _write_jsonl(tmp_path / "a.jsonl", SAMPLE_CASES)
    ds_lib.import_cases(storage, path)

    out_path = tmp_path / "export.jsonl"
    n = ds_lib.export_cases(storage, out_path)
    assert n == 3

    exported = ds_lib.load_cases_from_jsonl(out_path)  # 重新校验通过
    assert len(exported) == 3
    assert [c.id for c in exported] == ["c1", "c2", "c3"]

    before = {c.id: c.model_dump() for c in storage.list_cases()}
    after = {c.id: c.model_dump() for c in exported}
    assert before == after


def test_serialize_cases_empty():
    assert ds_lib.serialize_cases_to_jsonl([]) == ""


# ---------------------------------------------------------------------------
# experiments create（YAML）
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_resolve_experiment_name_priority():
    data = {"experiment_name": "from_yaml", "name": "from_name_key"}
    assert ds_lib.resolve_experiment_name(data, "from_cli") == "from_cli"
    assert ds_lib.resolve_experiment_name(data, None) == "from_yaml"
    data2 = {"name": "only_name_key"}
    assert ds_lib.resolve_experiment_name(data2, None) == "only_name_key"
    with pytest.raises(ds_lib.ConfigError):
        ds_lib.resolve_experiment_name({"other": 1}, None)


def test_create_experiment_from_config_stores_full_yaml(storage: Storage, tmp_path: Path):
    """experiments.config 列保存完整 YAML 原文；rag_version 取顶层字段。"""
    yaml_text = (
        "experiment_name: baseline\n"
        "rag_version: mock-baseline-v0\n"
        "config:\n"
        "  top_k: 3\n"
    )
    cfg = _write_config(tmp_path, yaml_text)
    exp = ds_lib.create_experiment_from_config(storage, cfg)

    assert exp.name == "baseline"
    assert exp.rag_version == "mock-baseline-v0"
    assert exp.config["config"]["top_k"] == 3
    rows = storage.list_experiments()
    assert rows[0].config == yaml_text  # 完整 YAML 原文
    # experiment_id 格式：exp_YYYYMMDD_HHMMSS_<6位随机后缀>
    import re

    assert re.fullmatch(r"exp_\d{8}_\d{6}_[0-9a-f]{6}", exp.experiment_id)


def test_create_experiment_name_override_and_duplicate(storage: Storage, tmp_path: Path):
    """--name 覆盖 YAML；重复创建同名实验报错且不产生新记录。"""
    yaml_text = "experiment_name: baseline\nrag_version: v1\n"
    cfg = _write_config(tmp_path, yaml_text)

    exp1 = ds_lib.create_experiment_from_config(storage, cfg, name="override_name")
    assert exp1.name == "override_name"

    with pytest.raises(DuplicateExperimentError):
        ds_lib.create_experiment_from_config(storage, cfg, name="override_name")
    assert len(storage.list_experiments()) == 1  # 库内仍只有 1 条

    # 不传 --name 时取名为 YAML experiment_name = baseline（与 override_name 不同，创建成功）
    exp2 = ds_lib.create_experiment_from_config(storage, cfg)
    assert exp2.name == "baseline"
    # 再以 baseline 重复创建 → 报错
    with pytest.raises(DuplicateExperimentError):
        ds_lib.create_experiment_from_config(storage, cfg)
    assert len(storage.list_experiments()) == 2


def test_create_experiment_no_valid_name_raises(storage: Storage, tmp_path: Path):
    cfg = _write_config(tmp_path, "rag_version: v1\nconfig: {}\n")
    with pytest.raises(ds_lib.ConfigError):
        ds_lib.create_experiment_from_config(storage, cfg)


def test_create_experiment_missing_config_file(storage: Storage, tmp_path: Path):
    with pytest.raises(ds_lib.ConfigError):
        ds_lib.create_experiment_from_config(storage, tmp_path / "no_such.yaml")


def test_create_experiment_invalid_yaml(storage: Storage, tmp_path: Path):
    cfg = _write_config(tmp_path, "not: [valid\n")
    with pytest.raises(ds_lib.ConfigError):
        ds_lib.create_experiment_from_config(storage, cfg)