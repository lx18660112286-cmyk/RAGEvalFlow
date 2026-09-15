"""阶段 4：Dashboard 数据层（ui.data_loader）测试。

不依赖 streamlit，仅验证纯数据读取 / 聚合 / 对比 / 报告生成逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragevalflow.core.storage import Storage
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.experiment_result import CaseResult
from ragevalflow.schemas.rag_output import RAGOutput
from ragevalflow.ui import data_loader as dl


def _make_case(cid: str = "case_1") -> EvalCase:
    return EvalCase(
        id=cid,
        category="IT",
        question="如何申请 VPN？",
        reference_answer="向 IT 提交申请。",
        expected_docs=["IT-VPN-001"],
        must_include=["工单系统"],
        must_not_include=["邮件审批"],
    )


def _make_result(experiment_id: str, case_id: str, metrics: dict | None = None) -> CaseResult:
    return CaseResult(
        experiment_id=experiment_id,
        case_id=case_id,
        rag_output=RAGOutput(question="q", answer="a", contexts=[]),
        metrics=metrics or {
            "hit_at_3": 1.0,
            "expected_doc_coverage": 1.0,
            "must_include_coverage": 1.0,
            "answer_context_overlap_score": 0.5,
            "latency_ms": 120.0,
        },
        failure_types=[],
    )


@pytest.fixture()
def populated(db_path: Path) -> Path:
    """初始化库 + 导入 1 条 case + 运行两个实验（含 case_results）。"""
    storage = Storage(db_path)
    storage.init_db()
    storage.import_cases([_make_case("case_1"), _make_case("case_2")])

    base = storage.create_experiment(name="baseline", rag_version="v0", config_yaml="name: baseline")
    cand = storage.create_experiment(name="agentic_v1", rag_version="v1", config_yaml="name: agentic_v1")
    base_metrics = {"hit_at_3": 1.0, "expected_doc_coverage": 0.5, "must_include_coverage": 0.5, "answer_context_overlap_score": 0.1, "latency_ms": 300.0}
    cand_metrics = {"hit_at_3": 1.0, "expected_doc_coverage": 1.0, "must_include_coverage": 1.0, "answer_context_overlap_score": 0.6, "latency_ms": 150.0}
    storage.save_case_results(
        [
            _make_result(base.experiment_id, "case_1", base_metrics),
            _make_result(base.experiment_id, "case_2", base_metrics),
            _make_result(cand.experiment_id, "case_1", cand_metrics),
            _make_result(cand.experiment_id, "case_2", cand_metrics),
        ]
    )
    return db_path


# ---------------------------------------------------------------------------
# 1. 数据库不存在时返回友好错误
# ---------------------------------------------------------------------------

def test_load_overview_db_missing_returns_friendly_error(tmp_path: Path):
    missing = tmp_path / "nope.db"
    overview = dl.load_overview(missing)
    assert "error" in overview
    assert "数据库不存在" in overview["error"]


def test_load_experiments_db_missing_returns_empty(tmp_path: Path):
    missing = tmp_path / "nope.db"
    assert dl.load_experiments(missing) == []
    assert dl.list_case_ids(missing, "exp_x") == []


# ---------------------------------------------------------------------------
# 2. 空数据库返回空列表，不崩溃
# ---------------------------------------------------------------------------

def test_empty_db_returns_empty_lists(db_path: Path):
    storage = Storage(db_path)
    storage.init_db()
    assert dl.load_experiments(db_path) == []
    overview = dl.load_overview(db_path)
    assert "error" not in overview
    assert overview["experiments"] == 0
    assert overview["eval_cases"] == 0


def test_db_not_initialized_friendly_error(tmp_path: Path):
    # 只创建一个空的 sqlite 文件，不建表
    empty_file = tmp_path / "empty.db"
    empty_file.write_bytes(b"")
    assert "数据库未初始化" in dl.load_overview(empty_file)["error"]


# ---------------------------------------------------------------------------
# 3. 能读取 experiments
# ---------------------------------------------------------------------------

def test_load_experiments(populated: Path):
    items = dl.load_experiments(populated)
    assert len(items) == 2
    by_name = {e["name"]: e for e in items}
    assert set(by_name) == {"baseline", "agentic_v1"}
    base = by_name["baseline"]
    assert base["case_results"] == 2
    assert base["avg_hit_at_3"] == 1.0
    assert base["avg_latency_ms"] == 300.0
    assert base["failure_total"] == 0


# ---------------------------------------------------------------------------
# 4. 能读取 experiment detail
# ---------------------------------------------------------------------------

def test_load_experiment_detail(populated: Path):
    storage = Storage(populated)
    base = storage.get_experiment_by_id_or_name("baseline")
    detail = dl.load_experiment_detail(populated, base.experiment_id)
    assert "error" not in detail
    assert detail["name"] == "baseline"
    assert len(detail["case_results"]) == 2
    row = detail["case_results"][0]
    for col in ("case_id", "category", "hit_at_3", "expected_doc_coverage", "must_include_coverage", "answer_context_overlap_score", "latency_ms", "failure_types", "status"):
        assert col in row
    assert "aggregate_metrics" in detail


def test_load_experiment_detail_not_found(populated: Path):
    detail = dl.load_experiment_detail(populated, "不存在")
    assert "error" in detail


# ---------------------------------------------------------------------------
# 5. 能读取 case results / case detail
# ---------------------------------------------------------------------------

def test_list_and_load_case_detail(populated: Path):
    storage = Storage(populated)
    base = storage.get_experiment_by_id_or_name("baseline")
    case_ids = dl.list_case_ids(populated, base.experiment_id)
    assert case_ids == ["case_1", "case_2"]

    cd = dl.load_case_detail(populated, base.experiment_id, "case_1")
    assert "error" not in cd
    assert cd["question"] == "如何申请 VPN？"
    assert cd["expected_docs"] == ["IT-VPN-001"]
    assert cd["must_not_include"] == ["邮件审批"]
    assert cd["answer"] == "a"
    assert "metrics" in cd
    assert "findings" in cd


# ---------------------------------------------------------------------------
# 6. 能调用 compare 数据逻辑（复用 regression.compare_experiments）
# ---------------------------------------------------------------------------

def test_run_comparison_reports_improved(populated: Path):
    res = dl.run_comparison(populated, "baseline", "agentic_v1")
    assert "error" not in res
    report = res["report"]
    assert report.summary.overall_status == "improved"
    assert report.summary.regressed_cases == 0

    data = dl.comparison_to_dicts(report)
    assert data["improved_cases"] == 2
    assert data["regressed_cases"] == 0
    assert len(data["improved_case_rows"]) == 2
    assert any(m["metric_name"] == "latency_ms" for m in data["metric_deltas"])


def test_run_comparison_missing_experiment(populated: Path):
    res = dl.run_comparison(populated, "baseline", "不存在")
    assert "error" in res


# ---------------------------------------------------------------------------
# 报告生成（复用现有 markdown_report）
# ---------------------------------------------------------------------------

def test_generate_reports(populated: Path, tmp_path: Path):
    storage = Storage(populated)
    base = storage.get_experiment_by_id_or_name("baseline")
    cand = storage.get_experiment_by_id_or_name("agentic_v1")

    single = dl.generate_experiment_report(populated, base.experiment_id, output=str(tmp_path / "single.md"))
    assert "error" not in single
    assert Path(single["path"]).exists()

    cmp_res = dl.generate_comparison_report(populated, base.experiment_id, cand.experiment_id, output=str(tmp_path / "cmp.md"))
    assert "error" not in cmp_res
    assert Path(cmp_res["path"]).exists()


# ---------------------------------------------------------------------------
# resolve_db_path 优先级
# ---------------------------------------------------------------------------

def test_resolve_db_path_priority(monkeypatch: pytest.MonkeyPatch):
    assert dl.resolve_db_path("custom.db") == Path("custom.db")
    monkeypatch.setenv("RAGEFLOW_DB", "env.db")
    assert dl.resolve_db_path(None) == Path("env.db")
    monkeypatch.delenv("RAGEFLOW_DB")
    assert dl.resolve_db_path(None) == dl.DEFAULT_DB
