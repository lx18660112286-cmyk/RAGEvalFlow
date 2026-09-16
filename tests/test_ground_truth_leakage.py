"""防 ground-truth 泄漏专项测试。

核心保证：调用真实 Agentic-RAG 时只传 ``case.question``（及其 case.id 用于 tracing），
reference_answer / expected_docs / must_include / must_not_include / expected_behavior 等
评测答案绝不进入被测系统输入，也不落进 case_results 的 config。
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

from ragevalflow.integrations.agentic_rag_client import AgenticRAGClient
from ragevalflow.schemas.eval_case import EvalCase

#: 这些 ground-truth 字段名绝不允许出现在 Adapter 的"构造被测系统输入"代码路径里。
GROUD_TRUTH_IDENTS = (
    "reference_answer",
    "expected_docs",
    "must_include",
    "must_not_include",
    "expected_behavior",
    "expected_tools",
    "forbidden_tools",
)


def test_adapter_source_never_reads_ground_truth_for_sut_input():
    """静态守卫：agentic_rag_client.py 中不得把任何 ground-truth 字段名用作代码标识符。

    用 ``tokenize`` 剔除注释与字符串（docstring/说明性文字允许提及这些词），
    只对真正的代码 token（NAME 标识符）做断言 —— 说明性文字不构成"读取 case 字段"。
    """
    src = Path(__file__).resolve().parents[1] / "ragevalflow" / "integrations" / "agentic_rag_client.py"
    text = src.read_text(encoding="utf-8")
    idents = _code_identifiers(text)
    for ident in GROUD_TRUTH_IDENTS:
        assert ident not in idents, f"Adapter 不应把 ground-truth 字段名 {ident} 当作代码标识符使用"


def _code_identifiers(text: str) -> frozenset[str]:
    """返回源码中真正的代码标识符（NAME token），剔除注释 / 字符串 / f-string 片段等。"""
    names: set[str] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type not in (tokenize.NAME, tokenize.OP):
            continue  # 跳过 COMMENT / STRING / NL / NEWLINE / INDENT / DEDENT / NUMBER 等
        names.add(tok.string)
    return frozenset(names)


def test_agent_result_mapping_does_not_access_ground_truth(capsys):
    """当传入含全部 ground-truth 的 EvalCase 时，Adapter 只暴露 case.question。"""
    calls: list[str] = []

    class _FakeAdapter:
        async def initialize(self):
            pass

        async def close(self):
            pass

    class _FakeOrch:
        async def run(self, message: str):
            calls.append(message)
            return _fake_agent_result()

    class _FakeBuilt:
        adapter = _FakeAdapter()
        tracer = None
        orchestrator = _FakeOrch()

    def builder(settings, working_dir, recorder):
        return _FakeBuilt()

    client = AgenticRAGClient(builder=builder)
    case = EvalCase(
        id="leak_001",
        question="Mercury access token 有效期是多少？",
        reference_answer="参考答案：30 分钟。",
        expected_docs=["api_auth.md"],
        must_include=["30 分钟"],
        must_not_include=["15 分钟"],
    )
    out = client.answer(case)
    client.close()

    assert calls == [case.question]
    payload = calls[0]
    assert payload == case.question  # 只传 question 本身
    for forbidden in ["参考答案", "api_auth.md", "30 分钟", "15 分钟"]:
        assert forbidden not in payload
        assert forbidden not in str(out.config)
    # contexts 为空（无检索）→ 对话无关，无泄密风险
    assert out.contexts == []


def _fake_agent_result():
    from types import SimpleNamespace

    return SimpleNamespace(
        status=SimpleNamespace(value="SUCCESS"),
        answer="答案",
        citations=[],
        tool_calls=[],
        steps=1,
        error=None,
        trace_id=None,
        original_query="",
        routing_steps=[],
    )


def test_runner_config_does_not_echo_ground_truth_into_experiment_config(tmp_path):
    """experiments.config 只保存 YAML 原文（不含数据集 ground-truth）。"""
    from ragevalflow.core import dataset as ds_lib
    from ragevalflow.core.storage import Storage

    db = tmp_path / "db.sqlite"
    st = Storage(db)
    st.init_db()
    yaml_text = (
        "experiment_name: cfg_leak\n"
        "rag_version: v\n"
        "config:\n"
        "  client: mock\n"
        "  mock_mode: baseline\n"
    )
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    exp = ds_lib.create_experiment_from_config(st, cfg)
    assert exp.config["experiment_name"] == "cfg_leak"
    row = st.get_experiment_by_id_or_name(exp.experiment_id)
    assert row is not None
    persisted = row.config or ""
    # 配置原文不应包含任何 ground-truth 内容（这里只是验证客户端配置不含评测答案）
    for ident in GROUD_TRUTH_IDENTS:
        assert ident not in persisted