"""真实被测系统接入后的 Runner 测试。

不真正连接 DEEPSEEK / LightRAG（无网络），而是验证 Runner 对 agentic 客户端的：
- client=agentic_rag 分发正确；
- 客户端生命周期 close 在评测结束后被调用；
- 单 case 失败不留下半成品 experiment（事务一致性）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragevalflow.core import dataset as ds_lib
from ragevalflow.core import runner as runner_mod
from ragevalflow.core.runner import run_experiment
from ragevalflow.core.storage import Storage
from ragevalflow.integrations.rag_client import MockRAGClient
from ragevalflow.schemas.rag_output import RAGOutput, RuntimeInfo, Trace

AGENTIC_CONFIG = (
    "experiment_name: real_rag_test\n"
    "rag_version: polaris-live\n"
    "config:\n"
    "  client: agentic_rag\n"
    "  model: deepseek-chat\n"
)

CASES = [
    {"id": "r1", "question": "Mercury access token 有效期？", "expected_docs": ["api_auth.md"], "must_include": ["30 分钟"]},
    {"id": "r2", "question": "Order Service 部署方式？", "expected_docs": ["deployment.md"], "must_include": ["blue-green"]},
]


class FakeAgenticClient:
    """模拟 AgenticRAGClient 的形状（__init__ 签名 + answer + close）。"""

    instances: list["FakeAgenticClient"] = []
    fail_after: int | None = None  # 类级默认；可在测试中覆写，__init__ 会读取

    def __init__(self, *, working_dir=None, model="", pricing=None):
        self.working_dir = working_dir
        self.model = model
        self.pricing = pricing
        self.closed = False
        self.called = 0
        self.fail_after: int | None = FakeAgenticClient.fail_after
        self.instances.append(self)

    def answer(self, case) -> RAGOutput:
        self.called += 1
        if self.fail_after is not None and self.called > self.fail_after:
            raise RuntimeError(f"[case:{case.id}] simulate real-system failure")
        return RAGOutput(
            question=case.question,
            answer="测试答案",
            trace=Trace(retrieval_rounds=1),
            runtime=RuntimeInfo(latency_ms=1.0),
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def env(tmp_path: Path):
    db_path = tmp_path / "run.db"
    storage = Storage(db_path)
    storage.init_db()
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CASES) + "\n", encoding="utf-8")
    ds_lib.import_cases(storage, jsonl)
    config = tmp_path / "run.yaml"
    config.write_text(AGENTIC_CONFIG, encoding="utf-8")
    return db_path, config


def test_build_client_dispatches_agentic(monkeypatch):
    monkeypatch.setattr(runner_mod, "AgenticRAGClient", FakeAgenticClient)
    assert isinstance(runner_mod._build_client({"client": "agentic_rag"}), FakeAgenticClient)
    assert isinstance(runner_mod._build_client({"client": "python"}), FakeAgenticClient)
    assert isinstance(runner_mod._build_client({"client": "mock", "mock_mode": "baseline"}), MockRAGClient)
    assert isinstance(runner_mod._build_client({}), MockRAGClient)  # 默认 mock，向后兼容


def test_run_experiment_with_agentic_client_calls_close(monkeypatch, env):
    monkeypatch.setattr(runner_mod, "AgenticRAGClient", FakeAgenticClient)
    db_path, config = env
    summary = run_experiment(db_path, config)

    client = FakeAgenticClient.instances[-1]
    assert client.called == len(CASES)
    assert client.closed is True  # 评测结束释放被测系统生命周期
    assert summary.cases == len(CASES)
    # 结果已落库
    storage = Storage(db_path)
    rows = storage.list_experiments()
    assert len(rows) == 1
    assert storage.count_case_results(rows[0].experiment_id) == len(CASES)


def test_single_case_failure_leaves_no_half_experiment(monkeypatch, tmp_path, env):
    """某 case 调用失败 → 整个 run 失败，不写任何半成品 experiment / case_results。"""
    monkeypatch.setattr(runner_mod, "AgenticRAGClient", FakeAgenticClient)
    db_path, config = env
    storage = Storage(db_path)
    # 让首个 case 就失败
    FakeAgenticClient.instances.clear()
    FakeAgenticClient.fail_after = 0

    with pytest.raises(RuntimeError):
        run_experiment(db_path, config)

    assert len(storage.list_experiments()) == 0
    for row in storage.list_experiments():
        assert storage.count_case_results(row.experiment_id) == 0
    # 生命周期 close 仍被调用
    client = FakeAgenticClient.instances[-1]
    assert client.closed is True