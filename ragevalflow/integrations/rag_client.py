"""RAGClient 协议与 MockRAGClient。

MockRAGClient 支持两种确定性的 mock 模式（不调用 LLM、不联网、结果完全可复现）：

- baseline：刻意表现较弱 —— 期望文档只命中一半、答案只含部分 must_include、
  不使用 query rewrite / reranker / reflection、retrieval_rounds=1；
- agentic_v1：表现更好 —— 期望文档全命中、答案覆盖全部 must_include、
  对 should_rewrite / should_multi_hop 模拟合理 trace、使用 reranker / reflection。

mock_mode 与 top_k 来自实验 YAML 的 config 段，例如：
    config:
      client: mock
      mock_mode: agentic_v1
      top_k: 5
"""

from __future__ import annotations

from typing import Protocol

from ragevalflow.schemas.eval_case import EvalCase, ExpectedBehavior
from ragevalflow.schemas.rag_output import Context, RAGOutput, RuntimeInfo, Trace

MOCK_MODES = ("baseline", "agentic_v1")

# 每 1 token 的估算单价（元），用于 deterministic 的成本模拟
_COST_PER_TOKEN = 0.00001


class RAGClient(Protocol):
    """外部 RAG / Agentic-RAG 系统的统一接口。"""

    def answer(self, case: EvalCase) -> RAGOutput: ...


def _stable_seed(text: str) -> int:
    """由文本派生稳定整数（确定性，无随机）。"""
    return sum(ord(ch) for ch in text)


def _context_text(doc_id: str) -> str:
    return f"文档 {doc_id} 的上下文片段，包含与该问题相关的关键信息。"


class MockRAGClient:
    """确定性 mock RAG 客户端。"""

    def __init__(self, mode: str = "baseline", top_k: int = 3):
        if mode not in MOCK_MODES:
            raise ValueError(f"未知 mock_mode: {mode!r}，可选值: {', '.join(MOCK_MODES)}")
        self.mode = mode
        self.top_k = max(1, int(top_k))

    # ------------------------------------------------------------------ answer

    def answer(self, case: EvalCase) -> RAGOutput:
        runner = _BaselineRunner(case, self.top_k) if self.mode == "baseline" else _AgenticRunner(case, self.top_k)
        return runner.build()

    def __repr__(self) -> str:  # pragma: no cover - 仅调试辅助
        return f"MockRAGClient(mode={self.mode!r}, top_k={self.top_k})"


# ---------------------------------------------------------------------------
# 两种模式的确定性"剧本"
# ---------------------------------------------------------------------------

class _MockRunnerBase:
    mode: str = ""

    def __init__(self, case: EvalCase, top_k: int):
        self.case = case
        self.top_k = top_k
        self.seed = _stable_seed(case.id)

    def build(self) -> RAGOutput:
        return RAGOutput(
            question=self.case.question,
            answer=self._answer(),
            contexts=self._contexts(),
            trace=self._trace(),
            runtime=self._runtime(),
            config={"client": "mock", "mock_mode": self.mode, "top_k": self.top_k},
        )

    def _contexts(self) -> list[Context]: ...
    def _answer(self) -> str: ...
    def _trace(self) -> Trace: ...

    def _strip_forbidden(self, answer: str) -> str:
        """移除答案中所有 must_not_include 短语，确保 mock 不会输出禁止内容。"""
        for kw in self.case.must_not_include:
            answer = answer.replace(kw, "")
        return answer

    def _runtime(self) -> RuntimeInfo:
        input_tokens = 200 + self.seed % 80
        output_tokens = 120 + self.seed % 60
        return RuntimeInfo(
            latency_ms=0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=(input_tokens + output_tokens) * _COST_PER_TOKEN,
        )


class _BaselineRunner(_MockRunnerBase):
    """baseline：命中一半期望文档，答案只覆盖一半 must_include，无 agentic 能力。"""

    mode = "baseline"

    def _contexts(self) -> list[Context]:
        expected = self.case.expected_docs
        hit_count = max(1, len(expected) // 2)  # 期望文档较多时只命中一半
        hit_docs = expected[:hit_count]
        return [
            Context(doc_id=doc, chunk_id=f"{doc}-c0", text=_context_text(doc), score=0.5 + i * 0.05, rank=i + 1)
            for i, doc in enumerate(hit_docs[: self.top_k])
        ]

    def _answer(self) -> str:
        answer = f"根据检索到的相关内容，以下是关于“{self.case.question}”的回答摘要。"
        # 只附加偶数索引的 must_include（模拟覆盖不全）
        missing = [kw for kw in self.case.must_include[::2] if kw not in answer]
        if missing:
            answer += "；".join(missing)
        return answer

    def _trace(self) -> Trace:
        return Trace(
            query_rewrite=None,
            retrieval_rounds=1,
            used_reranker=False,
            used_reflection=False,
            steps=[{"tool": "retriever", "round": 1}],
            tools_used=["retriever"],
        )

    def _runtime(self) -> RuntimeInfo:
        runtime = super()._runtime()
        return runtime.model_copy(update={"latency_ms": 260.0 + self.seed % 90})


class _AgenticRunner(_MockRunnerBase):
    """agentic_v1：命中全部期望文档，覆盖全部 must_include，启用 rewrite / multi-hop / reranker / reflection。"""

    mode = "agentic_v1"

    def _contexts(self) -> list[Context]:
        expected = self.case.expected_docs[: self.top_k]
        return [
            Context(doc_id=doc, chunk_id=f"{doc}-c0", text=_context_text(doc), score=0.85 + i * 0.03, rank=i + 1)
            for i, doc in enumerate(expected)
        ]

    def _answer(self) -> str:
        base = self.case.reference_answer or f"关于“{self.case.question}”的检索答案。"
        missing = [kw for kw in self.case.must_include if kw not in base]
        if missing:
            base += "；".join(missing)
        # 模拟引用标注：在答案中附上被引用文档的 doc_id
        if self.case.expected_docs:
            base += "\n参考来源：" + "、".join(self.case.expected_docs)
        # 关键修复（阶段 3.1）：参考答案可能含 must_not_include 短语（如 case_it_002），
        # 先剥离禁止内容，避免 agentic_v1 输出 forbidden_claim。
        return self._strip_forbidden(base)

    def _trace(self) -> Trace:
        behavior = self.case.expected_behavior or ExpectedBehavior()
        rounds = 1
        if behavior.should_multi_hop:
            # 需要多跳且上限允许时执行 2 轮检索
            if behavior.max_retrieval_rounds is None or behavior.max_retrieval_rounds >= 2:
                rounds = 2
        steps = [{"tool": "retriever", "round": r} for r in range(1, rounds + 1)]
        steps.append({"tool": "reranker", "round": rounds})
        tools_used = list(behavior.expected_tools)
        if "retriever" not in tools_used:
            tools_used = ["retriever"] + tools_used
        tools_used = tools_used + ["reranker"]
        return Trace(
            query_rewrite=f"查询改写：{self.case.question}" if behavior.should_rewrite else None,
            retrieval_rounds=rounds,
            used_reranker=True,
            used_reflection=True,
            steps=steps,
            tools_used=tools_used,
        )

    def _runtime(self) -> RuntimeInfo:
        runtime = super()._runtime()
        return runtime.model_copy(update={"latency_ms": 120.0 + self.seed % 60})