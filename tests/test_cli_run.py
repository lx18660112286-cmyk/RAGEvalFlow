"""CLI run 命令测试（直接调用 main()）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragevalflow.core.storage import Storage
from ragevalflow.main import main


@pytest.fixture()
def cli_env(tmp_path: Path):
    """返回 (db 路径, jsonl 路径, config 路径)。"""
    db = tmp_path / "cli.db"
    cases = [
        {"id": "c1", "question": "如何申请 VPN？", "expected_docs": ["IT-VPN-001"], "must_include": ["工单系统"]},
        {"id": "c2", "question": "NovaSearch 刷新间隔？", "must_include": ["5 分钟"]},
    ]
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    config = tmp_path / "run.yaml"
    config.write_text(
        "experiment_name: cli_test\nrag_version: v1\nconfig:\n  client: mock\n  mock_mode: agentic_v1\n  top_k: 3\n",
        encoding="utf-8",
    )
    return db, jsonl, config


def test_cli_run_full_flow(capsys, cli_env):
    """init-db → import → run → 实验创建且 case_results 落库。"""
    db, jsonl, config = cli_env
    assert main(["init-db", "--db", str(db)]) == 0
    assert main(["dataset", "import", str(jsonl), "--db", str(db)]) == 0
    assert main(["run", "--config", str(config), "--db", str(db)]) == 0

    captured = capsys.readouterr().out
    assert "实验运行完成" in captured
    assert "experiment_id: exp_" in captured
    assert "cases: 2" in captured
    assert "avg_hit_at_3:" in captured
    assert "avg_must_include_coverage:" in captured

    storage = Storage(db)
    experiments = storage.list_experiments()
    assert len(experiments) == 1
    assert storage.count_case_results(experiments[0].experiment_id) == 2


def test_cli_run_duplicate_name_fails(capsys, cli_env):
    db, jsonl, config = cli_env
    main(["init-db", "--db", str(db)])
    main(["dataset", "import", str(jsonl), "--db", str(db)])
    assert main(["run", "--config", str(config), "--db", str(db)]) == 0
    assert main(["run", "--config", str(config), "--db", str(db)]) == 1
    assert "已存在" in capsys.readouterr().err


def test_cli_run_without_dataset_fails(capsys, cli_env):
    db, _, config = cli_env
    main(["init-db", "--db", str(db)])
    assert main(["run", "--config", str(config), "--db", str(db)]) == 1
    assert "没有评测样例" in capsys.readouterr().err


def test_cli_run_name_override(capsys, cli_env):
    db, jsonl, config = cli_env
    main(["init-db", "--db", str(db)])
    main(["dataset", "import", str(jsonl), "--db", str(db)])
    assert main(["run", "--config", str(config), "--name", "custom_run", "--db", str(db)]) == 0
    captured = capsys.readouterr().out
    assert "name: custom_run" in captured