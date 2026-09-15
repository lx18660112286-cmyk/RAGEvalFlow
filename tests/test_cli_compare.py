"""CLI compare 命令测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragevalflow.core.storage import Storage
from ragevalflow.main import main
from ragevalflow.schemas.experiment_result import CaseResult
from ragevalflow.schemas.rag_output import RAGOutput


@pytest.fixture()
def compare_env(tmp_path: Path):
    """init-db + 导入 3 条样例 + 运行 baseline 与 agentic_v1 两个实验。"""
    db = tmp_path / "compare.db"
    cases = [
        {"id": "c1", "question": "如何申请 VPN？", "expected_docs": ["IT-VPN-001"], "must_include": ["工单系统"]},
        {"id": "c2", "question": "外包员工年假折算？", "expected_docs": ["HR-LEAVE-003", "HR-OUT-002"], "must_include": ["折算"], "expected_behavior": {"should_rewrite": True, "should_multi_hop": True}},
        {"id": "c3", "question": "NovaSearch 刷新间隔？", "must_include": ["5 分钟"]},
    ]
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    base_config = tmp_path / "baseline.yaml"
    base_config.write_text("experiment_name: baseline\nrag_version: v0\nconfig:\n  client: mock\n  mock_mode: baseline\n  top_k: 3\n", encoding="utf-8")
    agent_config = tmp_path / "agentic.yaml"
    agent_config.write_text("experiment_name: agentic_v1\nrag_version: v1\nconfig:\n  client: mock\n  mock_mode: agentic_v1\n  top_k: 5\n", encoding="utf-8")

    main(["init-db", "--db", str(db)])
    main(["dataset", "import", str(jsonl), "--db", str(db)])
    assert main(["run", "--config", str(base_config), "--db", str(db)]) == 0
    assert main(["run", "--config", str(agent_config), "--db", str(db)]) == 0
    return db


def test_compare_detects_agentic_improvement(capsys, compare_env, tmp_path: Path):
    db = compare_env
    out = tmp_path / "compare.md"
    assert main(
        ["compare", "--baseline", "baseline", "--candidate", "agentic_v1", "--output", str(out), "--db", str(db)]
    ) == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "# Experiment Comparison Report" in content
    assert "## 2. Metric Deltas" in content

    captured = capsys.readouterr().out
    assert "对比报告已生成" in captured
    assert "overall_status: " in captured
    assert "regressed_cases: 0" in captured
    # agentic 在延迟/检索文档覆盖率上应优于 baseline
    storage = Storage(db)
    base = storage.get_experiment_by_id_or_name("baseline")
    cand = storage.get_experiment_by_id_or_name("agentic_v1")
    base_results = storage.list_case_results(base.experiment_id)
    cand_results = storage.list_case_results(cand.experiment_id)
    base_latency = sum(r.metrics["latency_ms"] for r in base_results) / len(base_results)
    cand_latency = sum(r.metrics["latency_ms"] for r in cand_results) / len(cand_results)
    assert cand_latency < base_latency
    base_cov = sum(r.metrics["expected_doc_coverage"] for r in base_results) / len(base_results)
    cand_cov = sum(r.metrics["expected_doc_coverage"] for r in cand_results) / len(cand_results)
    assert cand_cov > base_cov


def test_compare_with_experiment_id(capsys, compare_env, tmp_path: Path):
    db = compare_env
    storage = Storage(db)
    base = storage.get_experiment_by_id_or_name("baseline")
    cand = storage.get_experiment_by_id_or_name("agentic_v1")
    out = tmp_path / "by_id.md"
    assert main(
        ["compare", "--baseline", base.experiment_id, "--candidate", cand.experiment_id, "--output", str(out), "--db", str(db)]
    ) == 0
    assert out.exists()


def test_compare_experiment_not_found(capsys, compare_env):
    db = compare_env
    assert main(["compare", "--baseline", "baseline", "--candidate", "不存在", "--db", str(db)]) == 1
    assert "未找到 candidate 实验" in capsys.readouterr().err


def test_compare_without_common_case_fails(capsys, tmp_path: Path):
    """baseline/candidate 无共同 case_id → 中文错误、退出码非 0。"""
    db = tmp_path / "disjoint.db"
    cases = [{"id": "c1", "question": "q1"}, {"id": "c2", "question": "q2"}]
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    config = tmp_path / "run.yaml"
    config.write_text("experiment_name: only_base\nrag_version: v0\nconfig:\n  client: mock\n  mock_mode: baseline\n", encoding="utf-8")
    main(["init-db", "--db", str(db)])
    main(["dataset", "import", str(jsonl), "--db", str(db)])
    main(["run", "--config", str(config), "--db", str(db)])

    # 手工创建 candidate 实验并写入不重叠的 case_results
    storage = Storage(db)
    base = storage.get_experiment_by_id_or_name("only_base")
    cand_row = storage.create_experiment(name="cand", rag_version="v1", config_yaml="name: cand")
    storage.save_case_results([
        CaseResult(
            experiment_id=cand_row.experiment_id,
            case_id="zz_other",
            rag_output=RAGOutput(question="q", answer="a"),
            metrics={"latency_ms": 1.0},
        )
    ])
    assert base is not None
    assert main(["compare", "--baseline", "only_base", "--candidate", "cand", "--db", str(db)]) == 1
    assert "没有共同" in capsys.readouterr().err