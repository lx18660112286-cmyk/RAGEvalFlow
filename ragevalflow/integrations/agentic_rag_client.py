"""AgenticRAGClient —— 真实 Agentic-RAG 被测系统的 Python Adapter（模式 A/C）。

被测系统 = ``Dev Knowledge Agent``（包名 ``polaris_agentic_rag``），其真实入口为：:

    built = build_agent(settings, working_dir=..., tracer=...)
    await built.adapter.initialize()
    result = await built.orchestrator.run(query)   # 单轮 query，无需 history/thread_id
    await built.adapter.close()

本客户端不改变 Agent 的任何决策，只做纯观测（薄 instrumentation）：
- 用 ``InMemoryTraceSink`` 采集真实 TraceEvent（拿 Agent-LLM 的 token usage、检索/路由事件）；
- 用一个只读 ``SearchRecorder`` 包装 ``LightRAGAdapter``（``KnowledgeSearchPort``），
  记录每次真实检索返回的 ``KnowledgeSearchResult``（拿 chunk 正文 / source_name）。

字段映射（真实来源，绝不伪造）见模块内 ``build_rag_output``。

刻意不在模块顶层 import ``polaris_agentic_rag``（RAGEvalFlow 系统 Python 可能未安装该被测包），
而采用惰性导入，保证 MockRAGClient / 其它集成不受影响。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.rag_output import Context, RAGOutput, RuntimeInfo, Trace

__all__ = ["AgenticRAGClient", "SearchRecorder", "build_rag_output"]


def _basename(source_name: str | None) -> str:
    """把 chunk/citation 的 source_name（可能是路径）规约为稳定文档名（basename）。

    Agentic-RAG 的 mapper 已把 file_path 归一化为 basename（如 ``api_auth.md``），
    此处再做一层防御：若出现带目录的路径，取 basename，保证 doc_id 是文档级稳定标识。
    """
    if not source_name:
        return ""
    return Path(str(source_name)).name


# ---------------------------------------------------------------------------
# 只读观测：包装 KnowledgeSearchPort（纯观测，不改检索决策）
# ---------------------------------------------------------------------------

class SearchRecorder:
    """转发到真实 KnowledgeSearchPort，并记录每次检索返回的 KnowledgeSearchResult。

    仅用于评测观测；不修改 query、plan，也不改变返回内容。
    ``trace_sink`` 由真实构建器注入，供客户端收集每个 TraceEvent（Agent-LLM usage 等）。
    """

    def __init__(self, delegate: object | None = None) -> None:
        self._delegate = delegate
        self.results: list = []
        self.trace_sink: object | None = None

    def bind(self, delegate: object) -> None:
        """绑定真实 KnowledgeSearchPort（LightRAGAdapter）。"""
        self._delegate = delegate

    def is_bound(self) -> bool:
        return self._delegate is not None and getattr(self._delegate, "initialized", True) is not False

    async def initialize(self) -> None:
        await self._delegate.initialize()  # type: ignore[attr-defined]

    async def close(self) -> None:
        await self._delegate.close()  # type: ignore[attr-defined]

    async def search(self, query: str, *, plan: object | None = None):
        result = await self._delegate.search(query, plan=plan)  # type: ignore[attr-defined]
        self.results.append(result)
        return result

    async def clear(self) -> None:
        self.results = []

    def recorded_events(self) -> list:
        """返回本次运行收集到的 TraceEvent（无则空列表）。"""
        sink = self.trace_sink
        if sink is None:
            return []
        events = getattr(sink, "events", None)
        if events is None:
            return []
        return list(events)


# ---------------------------------------------------------------------------
# 默认构建器（真实 wiring）；builder 参数可注入以支持测试
# ---------------------------------------------------------------------------

def _default_builder(settings, working_dir, recorder) -> object:
    """用被测系统自己的 composition root 组装真 Agent（惰性 import polaris）。"""
    from polaris_agentic_rag.adapters.lightrag.adapter import LightRAGAdapter
    from polaris_agentic_rag.bootstrap import build_agent, create_lightrag_adapter
    from polaris_agentic_rag.config.settings import get_settings
    from polaris_agentic_rag.observability.sinks import InMemoryTraceSink
    from polaris_agentic_rag.observability.tracer import Tracer

    if working_dir is not None:
        working_dir = Path(working_dir)
    real = create_lightrag_adapter(working_dir=working_dir)
    recorder.bind(real)  # recorder 代理真实 LightRAGAdapter（生命周期 + search）
    sink = InMemoryTraceSink()
    recorder.trace_sink = sink
    tracer = Tracer(sinks=[sink])
    agent = build_agent(
        settings or get_settings(),
        lightrag_adapter=recorder,  # KnowledgeSearchPort = recorder（只读观测）
        working_dir=working_dir,
        tracer=tracer,
    )
    return agent


class AgenticRAGClient:
    """一次评测实验复用的真实 Agentic-RAG 客户端。

    实现 ``RAGClient.answer(case) -> RAGOutput`` 协议；只把 ``case.question``（及其作为
    trace 的 case.id）传给被测系统，绝不传任何 ground-truth / 指标字段。

    生命周期：``answer`` 首次调用时在当前线程创建并暂存一个新事件循环，
    在同一个循环上 ``initialize`` 并复用于所有 case；``close()`` 在同一循环上关闭。
    """

    #: 仅当需要按 agent-LLM token 且提供真实定价时，才计算 estimated_cost。
    #: 缺省 None → estimated_cost 标记 unavailable（不伪造）。
    def __init__(
        self,
        *,
        builder=None,
        settings=None,
        working_dir=None,
        model: str = "",
        pricing: dict | None = None,
    ) -> None:
        self._builder = builder or _default_builder
        self._settings = settings
        self._working_dir = working_dir
        self._model = model
        self._pricing = pricing
        self._built = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        #: recorder 由客户端持有，构建时注入真实端口；每次 answer 前复位观测。
        self._recorder = SearchRecorder()

    # ---------------------------------------------------------- 生命周期

    def _ensure_built(self) -> tuple:
        """惰性构建 + 初始化；返回 (built_agent, loop)。"""
        if self._built is not None and self._loop is not None:
            return self._built, self._loop
        if self._closed:
            raise RuntimeError("AgenticRAGClient has been closed.")
        loop = asyncio.new_event_loop()
        try:
            built = self._builder(self._settings, self._working_dir, self._recorder)
            loop.run_until_complete(built.adapter.initialize())
        except BaseException:
            loop.close()
            raise
        self._built = built
        self._loop = loop
        return built, loop

    def close(self) -> None:
        """在同一循环上关闭 LightRAG 内核并释放循环（幂等）。"""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        built = self._built
        if loop is None or built is None:
            return
        try:
            loop.run_until_complete(built.adapter.close())
        finally:
            loop.close()
            self._loop = None

    def __enter__(self) -> "AgenticRAGClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------- answer

    def answer(self, case: EvalCase) -> RAGOutput:
        """对单条 case 运行真实 Agentic-RAG 一次，返回标准化 RAGOutput。

        只传 ``case.question``（真实生产环境能拿到的信息）。任何 ground-truth
        （reference_answer / expected_docs / must_include / must_not_include /
        expected_behavior / 指标标签）都不会出现在被测系统输入里。
        """
        built, loop = self._ensure_built()
        #: 纯观测状态复位（不影响被测系统）
        if self._recorder.is_bound():
            loop.run_until_complete(self._recorder.clear())
        else:
            self._recorder.results = []

        start = time.perf_counter()
        try:
            result = loop.run_until_complete(built.orchestrator.run(case.question))
        except BaseException as exc:  # noqa: BLE001 - 被测系统调用失败 → 上抛，交由 runner 事务处理
            raise RuntimeError(
                f"[case:{case.id}] Agentic-RAG orchestrator.run 失败: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        return build_rag_output(
            question=case.question,
            original_query=case.question,
            agent_result=result,
            recorded_results=list(self._recorder.results),
            trace_events=self._recorder.recorded_events(),
            model=self._model,
            latency_ms=latency_ms,
            pricing=self._pricing,
            inner_config=self._agent_config(),
        )

    def _agent_config(self) -> dict:
        """记录被测系统身份与运行位置（不含任何 secret / ground-truth）。"""
        cfg: dict = {"integration": "python", "client": "agentic_rag"}
        built = self._built
        if built is not None and built.tracer is not None:
            cfg["tracer"] = "enabled"
        if self._working_dir:
            cfg["working_dir"] = str(self._working_dir)
        if self._model:
            cfg["agent_model"] = self._model
        return cfg


# ---------------------------------------------------------------------------
# 纯映射函数：AgentResult + 观测结果 -> RAGOutput（可独立测试）
# ---------------------------------------------------------------------------

def build_rag_output(
    *,
    question: str,
    original_query: str,
    agent_result,
    recorded_results: list,
    trace_events: list,
    model: str = "",
    latency_ms: float,
    pricing: dict | None = None,
    inner_config: dict | None = None,
) -> RAGOutput:
    """把一次真实 Agent 运行结果映射为 RAGOutput（不伪造任何字段）。

    - contexts: 来自 recorder 记录的真实 chunk；doc_id = basename(source_name)（文档级）；
      score 未提供 → None（unavailable），不打分。
    - trace: retrieval_rounds = 真实检索执行次数（RoutingStep 数）；used_reranker /
      used_reflection 如实为 false（系统确实未实现，不是伪造）；tools_used 来自真实工具名。
    - runtime: latency_ms 由外层真实计时；tokens 来自真实 MODEL_CALL_COMPLETED usage；
      estimated_cost 仅在 token+model+pricing 齐备时计算，否则 None。
    """
    output = _map_agent_result(agent_result, original_query)
    contexts = _map_contexts(recorded_results)
    trace = output.trace
    trace.retrieval_rounds = max(len(agent_result.routing_steps), 1)
    tools = _tools_from(result=agent_result)
    if tools:
        trace.tools_used = tools
        trace.steps = _map_steps(agent_result)
    rewritten = _first_rewrite(agent_result, original_query)
    if rewritten is not None:
        trace.query_rewrite = rewritten
    #: query_rewrite 若等于原文则视为未改写（无独立 rewrite 节点，保持偏保守）
    if trace.query_rewrite == original_query:
        trace.query_rewrite = None

    input_tokens, output_tokens = _collect_tokens(trace_events)
    estimated_cost = _estimate_cost(input_tokens, output_tokens, model, pricing)

    config: dict = {"model": model or "deepseek-chat"}
    status = output.config.get("status") if output.config else None
    if status:
        config["status"] = status
    if inner_config:
        config = {**inner_config, **config}

    return RAGOutput(
        question=question,
        answer=output.answer,
        contexts=contexts,
        trace=trace,
        runtime=RuntimeInfo(
            latency_ms=round(latency_ms, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
        ),
        config=config,
    )


def _map_agent_result(agent_result, original_query: str):
    """把 AgentResult 映射为最小 RAGOutput（answers / trace 骨架）。"""
    return RAGOutput(
        question=original_query,
        answer=agent_result.answer if agent_result.answer else "",
        trace=Trace(
            retrieval_rounds=1,
            used_reranker=False,      # 系统确未实现 reranker
            used_reflection=False,    # 系统确未实现 reflection
            steps=[],
            tools_used=[],
        ),
        config={"status": agent_result.status.value if hasattr(agent_result.status, "value") else str(agent_result.status)},
    )


def _map_contexts(recorded_results) -> list[Context]:
    """从记录的真实检索结果聚合 chunk，规约 doc_id 到文档级，按首次出现去重。"""
    contexts: list[Context] = []
    seen: set[str] = set()
    rank = 0
    for result in recorded_results:
        for chunk in getattr(result, "evidence", None).chunks if getattr(result, "evidence", None) else []:
            doc_id = _basename(getattr(chunk, "source_name", None))
            chunk_id = getattr(chunk, "chunk_id", None)
            #: 稳定标识去重（同一 chunk 在多轮被重复检索只保留一次）
            key = chunk_id or f"{doc_id}#{getattr(chunk, 'reference_id', '')}"
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            contexts.append(
                Context(
                    doc_id=doc_id or chunk_id or str(rank),
                    chunk_id=chunk_id or None,
                    text=getattr(chunk, "content", None) or "",
                    score=None,  # Agentic-RAG 未提供检索得分 → unavailable
                    rank=rank,
                )
            )
    return contexts


def _tools_from(*, result) -> list[str]:
    """去重后的真实工具名（来自 ToolCallRecord.name）。"""
    names: list[str] = []
    for record in getattr(result, "tool_calls", []):
        name = getattr(record, "name", None)
        if name and name not in names:
            names.append(name)
    return names


def _map_steps(result) -> list[dict]:
    """把真实工具调用转成 trace.steps（tool / round / status / duration_ms）。"""
    steps: list[dict] = []
    for i, record in enumerate(getattr(result, "tool_calls", []), start=1):
        step: dict = {"tool": getattr(record, "name", ""), "round": i}
        status = getattr(record, "result_status", None)
        if status:
            step["status"] = status
        duration = getattr(record, "duration_ms", None)
        if duration is not None:
            step["duration_ms"] = round(duration, 3)
        routing = getattr(record, "routing", None)
        if routing is not None:
            step["intent"] = getattr(routing, "intent", None)
            step["strategy"] = getattr(routing, "strategy", None)
        steps.append(step)
    return steps


def _first_rewrite(result, original_query: str) -> str | None:
    """工具传给检索的实际 query；若与原文不同视为发生了真实改写（无独立 rewrite 节点）。"""
    for step in getattr(result, "routing_steps", []):
        tool_query = getattr(step, "tool_query", None)
        if tool_query and tool_query != original_query:
            return tool_query
    return None


def _collect_tokens(trace_events) -> tuple[int | None, int | None]:
    """从 MODEL_CALL_COMPLETED 的真实 usage 汇总 Agent-LLM 的 input/output token。

    只累加存在且非 None 的数值；若完全没有观测到 usage → 返回 (None, None)（unavailable）。
    """
    input_total = 0
    output_total = 0
    seen = False
    for event in trace_events:
        if _event_name(event) != "MODEL_CALL_COMPLETED":
            continue
        attrs = getattr(event, "attributes", None) or {}
        it = attrs.get("input_tokens")
        ot = attrs.get("output_tokens")
        if isinstance(it, int):
            input_total += it
            seen = True
        if isinstance(ot, int):
            output_total += ot
            seen = True
    if not seen:
        return None, None
    return input_total, output_total


def _event_name(event) -> str:
    et = getattr(event, "event_type", None)
    if et is None:
        return ""
    return getattr(et, "value", str(et))


def _estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    model: str,
    pricing: dict | None,
) -> float | None:
    """仅在 token + model + pricing 齐备时计算 estimated_cost；否则 None（不伪造成本）。"""
    if input_tokens is None or output_tokens is None:
        return None
    if not model or not pricing:
        return None
    try:
        in_price = float(pricing["input_per_1m_tokens"])
        out_price = float(pricing["output_per_1m_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    return round((input_tokens * in_price + output_tokens * out_price) / 1_000_000, 6)