"""CLI report 命令测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragevalflow.core.storage import Storage
from ragevalflow.main import main


@pytest.fixture()
def cli_env(tmp_path: Path):
    db = tmp_path / "cli.db"
    cases = [
        {"id": "c1", "question": "如何申请 VPN？", "expected_docs": ["IT-VPN-001"], "must_include": ["工单系统"]},
        {"id": "c2", "question": "NovaSearch 刷新间隔？", "must_include": ["5 分钟"]},
    ]
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    config = tmp_path / "run.yaml"
    config.write_text(
        "experiment_name: cli_base\nrag_version: v1\nconfig:\n  client: mock\n  mock_mode: baseline\n  top_k: 3\n",
        encoding="utf-8",
    )
    main(["init-db", "--db", str(db)])
    main(["dataset", "import", str(jsonl), "--db", str(db)])
    main(["run", "--config", str(config), "--db", str(db)])
    return db


def test_report_generates_markdown_file(capsys, cli_env, tmp_path: Path):
    db = cli_env
    out = tmp_path / "reports" / "base.md"
    assert main(["report", "--experiment", "cli_base", "--output", str(out), "--db", str(db)]) == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "# Experiment Report" in content
    assert "## 3. Failure Summary" in content
    assert "cli_base" in content
    assert "实验报告已生成" in capsys.readouterr().out


def test_report_accepts_experiment_id(capsys, cli_env, tmp_path: Path):
    db = cli_env
    experiment_id = Storage(db).list_experiments()[0].experiment_id
    out = tmp_path / "by_id.md"
    assert main(["report", "--experiment", experiment_id, "--output", str(out), "--db", str(db)]) == 0
    assert out.exists()
    assert experiment_id in out.read_text(encoding="utf-8")


def test_report_default_output_path(capsys, cli_env, tmp_path: Path, monkeypatch):
    db = cli_env
    monkeypatch.chdir(tmp_path)  # 默认输出到 ./reports/
    assert main(["report", "--experiment", "cli_base", "--db", str(db)]) == 0
    files = list((tmp_path / "reports").glob("*.md"))
    assert files, "应生成 reports/ 下的默认文件"


def test_report_experiment_not_found(capsys, cli_env):
    db = cli_env
    assert main(["report", "--experiment", "不存在", "--db", str(db)]) == 1
    assert "未找到实验" in capsys.readouterr().err


def test_report_experiment_without_case_results(capsys, cli_env):
    db = cli_env
    Storage(db).create_experiment(name="empty_exp", rag_version="v1", config_yaml="name: empty_exp")
    assert main(["report", "--experiment", "empty_exp", "--db", str(db)]) == 1
    assert "没有 case_results" in capsys.readouterr().err