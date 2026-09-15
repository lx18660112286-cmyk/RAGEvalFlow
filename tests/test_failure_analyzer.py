"""失败归因（failure_analyzer）测试：覆盖阶段 3 要求的全部归因规则。"""

from __future__ import annotations

from ragevalflow.analysis.failure_analyzer import analyze_failures
from ragevalflow.metrics import compute_all_metrics
from ragevalflow.schemas.eval_case import EvalCase, ExpectedBehavior
from ragevalflow.schemas.rag_output import Context, RAGOutput, RuntimeInfo, Trace


def _case(**kwargs) -> EvalCase:
    base = {"id": "c1", "question": "测试问题"}
    base.update(kwargs)
    return EvalCase(**base)


def _output(answer: str = "回答", contexts: list | None = None, trace: Trace | None = None, runtime: RuntimeInfo | None = None) -> RAGOutput:
    return RAGOutput(
        question="测试问题",
        answer=answer,
        contexts=contexts or [],
        trace=trace or Trace(),
        runtime=runtime or RuntimeInfo(),
    )


def test_retrieval_miss_attribution():
    """期望文档非空且 hit_at_3=0 → retrieval_miss（high）。"""
    case = _case(expected_docs=["DOC-A", "DOC-B"])
    output = _output(contexts=[Context(doc_id="DOC-X", chunk_id="x0", text="无关", rank=1)])
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    miss = [f for f in findings if f.failure_type == "retrieval_miss"]
    assert miss
    assert miss[0].severity == "high"
    assert miss[0].message and miss[0].evidence and miss[0].metric_values


def test_partial_retrieval_bad_ranking_medium():
    """expected_doc_coverage < 1.0 且非全 miss → bad_ranking（medium）。"""
    case = _case(expected_docs=["DOC-A", "DOC-B"])
    output = _output(contexts=[Context(doc_id="DOC-A", chunk_id="a0", text="A 内容", rank=1)])
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    partial = [f for f in findings if f.failure_type == "bad_ranking"]
    assert partial
    assert partial[0].severity == "medium"
    assert "partial" in partial[0].message.lower() or "部分" in partial[0].message


def test_must_include_missing_incomplete_answer():
    """must_include 未完全覆盖 → incomplete_answer（high）。"""
    case = _case(must_include=["关键字A", "关键字B"])
    output = _output(answer="只包含关键字A")
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "incomplete_answer" and "must_include" in f.message]
    assert hit
    assert hit[0].severity == "high"


def test_forbidden_claim_critical():
    """答案包含禁止内容 → forbidden_claim（critical）。"""
    case = _case(must_not_include=["邮件审批"])
    output = _output(answer="通过邮件审批完成")
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "forbidden_claim"]
    assert hit
    assert hit[0].severity == "critical"


def test_missing_multi_hop():
    """should_multi_hop=True 但 retrieval_rounds=1 → missing_multi_hop（high）。"""
    case = _case(expected_behavior=ExpectedBehavior(should_multi_hop=True))
    output = _output(contexts=[Context(doc_id="D", chunk_id="d0", text="文档", rank=1)], trace=Trace(retrieval_rounds=1))
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "missing_multi_hop"]
    assert hit
    assert hit[0].severity == "high"


def test_query_rewrite_missing():
    """should_rewrite=True 但 rewrite_used=0 → query_rewrite_missing（medium）。"""
    case = _case(expected_behavior=ExpectedBehavior(should_rewrite=True))
    output = _output(trace=Trace(query_rewrite=None))
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "query_rewrite_missing"]
    assert hit
    assert hit[0].severity == "medium"


def test_expected_tool_missing():
    """期望工具未被调用 → expected_tool_missing（high）。"""
    case = _case(expected_behavior=ExpectedBehavior(expected_tools=["kb_search"]))
    output = _output(trace=Trace(tools_used=["retriever"]))
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "expected_tool_missing"]
    assert hit
    assert hit[0].severity == "high"


def test_forbidden_tool_called_critical():
    """禁止工具被调用 → forbidden_tool_called（critical）。"""
    case = _case(expected_behavior=ExpectedBehavior(forbidden_tools=["public_web_search"]))
    output = _output(trace=Trace(tools_used=["public_web_search", "retriever"]))
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "forbidden_tool_called"]
    assert hit
    assert hit[0].severity == "critical"
    # 合并规则：禁止工具 + 工具缺失 同 case 各得一条
    assert len([f for f in findings if f.failure_type == "forbidden_tool_called"]) == 1


def test_low_faithfulness_unsupported_answer():
    """答案与上下文重叠度过低 → 低忠实度失败（medium）。"""
    case = _case(expected_docs=["DOC-A"])
    output = _output(
        answer="这是一个与上下文完全无关的答案内容。",
        contexts=[Context(doc_id="DOC-A", chunk_id="a0", text="回收站清理规则说明", rank=1)],
    )
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "incomplete_answer" and "忠实度" in f.message]
    assert hit
    assert hit[0].severity == "medium"


def test_over_retrieval_medium():
    """检索轮数超过上限 → over_retrieval（medium）。"""
    case = _case(expected_behavior=ExpectedBehavior(max_retrieval_rounds=1))
    output = _output(trace=Trace(retrieval_rounds=3))
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    hit = [f for f in findings if f.failure_type == "over_retrieval"]
    assert hit
    assert hit[0].severity == "medium"


def test_high_latency_and_cost_default_thresholds():
    """延迟/成本超过默认阈值 → latency_regression / cost_regression（medium）。"""
    case = _case()
    output = _output(runtime=RuntimeInfo(latency_ms=5000.0, input_tokens=100, output_tokens=100, estimated_cost=0.05))
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    types = {f.failure_type: f for f in findings}
    assert types["latency_regression"].severity == "medium"
    assert types["cost_regression"].severity == "medium"

    # 自定义阈值可配置
    findings2 = analyze_failures(case, output, metrics, latency_threshold_ms=8000.0, cost_threshold=0.1)
    assert "latency_regression" not in {f.failure_type for f in findings2}
    assert "cost_regression" not in {f.failure_type for f in findings2}


def test_no_failure_returns_empty_list():
    """无条件失败（约束均满足、检索命中、答案有支撑）→ 返回空列表。"""
    case = _case(expected_docs=["DOC-A"], must_include=["关键字A"], must_not_include=["禁用词"])
    output = _output(
        answer="关键字A DOC-A，内容完全来自检索上下文 DOC-A。",
        contexts=[Context(doc_id="DOC-A", chunk_id="a0", text="关键字A 内容 来自检索上下文 DOC-A", rank=1)],
        trace=Trace(query_rewrite="改写", retrieval_rounds=2, used_reranker=True, used_reflection=True),
    )
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    assert findings == []


def test_empty_constraints_do_not_trigger_failure():
    """空 expected_docs / must_include / expected_tools → 相关失败不触发。"""
    case = _case()  # 没有任何约束
    output = _output()
    metrics = compute_all_metrics(case, output)
    findings = analyze_failures(case, output, metrics)
    assert all(f.failure_type not in ("retrieval_miss", "bad_ranking", "incomplete_answer", "expected_tool_missing") for f in findings)