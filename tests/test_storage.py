"""SQLite 存储层测试：建表 / 写入 / 查询 / 实验唯一性。"""

from __future__ import annotations

import re

import pytest
from sqlalchemy import inspect

from ragevalflow.core.storage import DuplicateExperimentError, Storage
from ragevalflow.schemas.eval_case import EvalCase


def test_init_db_creates_only_three_user_tables(storage: Storage):
    """init_db 后恰好 3 张用户表，且不含 sqlite_% 内部表。"""
    tables = storage.user_table_names()
    assert sorted(tables) == ["case_results", "eval_cases", "experiments"]
    # 通过 inspect 直接验证也排除 sqlite_% 内部表
    raw = [t for t in inspect(storage._engine).get_table_names() if not t.startswith("sqlite_")]
    assert set(raw) == {"eval_cases", "experiments", "case_results"}


def test_init_db_idempotent(db_path, storage: Storage):
    storage.init_db()  # 重复初始化不报错
    assert len(storage.user_table_names()) == 3


def test_case_persistence_round_trip(storage: Storage):
    case = EvalCase(
        id="c1",
        category="安全",
        question="数据如何加密？",
        reference_answer="AES-256",
        expected_docs=["SEC-1"],
        must_include=["AES-256"],
        must_not_include=["明文"],
        expected_behavior={"should_rewrite": True},
    )  # type: ignore[arg-type]
    storage.import_cases([case])
    cases = storage.list_cases()
    assert len(cases) == 1
    restored = cases[0]
    assert restored.id == "c1"
    assert restored.expected_docs == ["SEC-1"]
    assert restored.expected_behavior is not None and restored.expected_behavior.should_rewrite is True


def test_case_persistence_without_expected_behavior(storage: Storage):
    case = EvalCase(id="c2", question="无行为约束")
    storage.import_cases([case])
    restored = storage.list_cases()[0]
    assert restored.expected_behavior is None
    assert restored.must_include == []


def test_create_experiment_id_format_and_unique_name(storage: Storage):
    row = storage.create_experiment(name="baseline", rag_version="v1", config_yaml="name: baseline\n")
    assert re.fullmatch(r"exp_\d{8}_\d{6}_[0-9a-f]{6}", row.experiment_id)
    assert row.name == "baseline"
    # 同名重复：报错 + 事务回滚
    with pytest.raises(DuplicateExperimentError):
        storage.create_experiment(name="baseline", rag_version="v2", config_yaml="other")
    assert len(storage.list_experiments()) == 1
    rows = storage.list_experiments()
    assert rows[0].rag_version == "v1"  # 未被子事务污染


def test_list_experiments_newest_first(storage: Storage):
    storage.create_experiment(name="a", rag_version="", config_yaml="")
    storage.create_experiment(name="b", rag_version="", config_yaml="")
    names = [r.name for r in storage.list_experiments()]
    assert names == ["b", "a"]  # 按 created_at 倒序


def test_count_case_results_empty(storage: Storage):
    row = storage.create_experiment(name="base", rag_version="", config_yaml="")
    assert storage.count_case_results(row.experiment_id) == 0