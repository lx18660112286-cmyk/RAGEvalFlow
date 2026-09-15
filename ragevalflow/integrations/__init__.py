"""外部 RAG / Agentic-RAG 系统集成。

阶段 2 仅提供 RAGClient 协议与 MockRAGClient（确定性、不联网、不调用 LLM）。
真实外部 RAG API 调用（httpx）与 Ragas / DeepEval adapter 属于阶段 5 预留。
"""

from ragevalflow.integrations.rag_client import RAGClient, MockRAGClient

__all__ = ["RAGClient", "MockRAGClient"]