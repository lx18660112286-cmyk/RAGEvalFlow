"""真实 Agentic-RAG smoke test：对少量 case 跑一次，验证 Adapter→被测系统契约。

用法（需在被测系统 .venv 中，且该 venv 已安装 ragevalflow、能 import polaris_agentic_rag）：

    pip install -e "D:/myself-prove/RAGEvalFlow"          # 一次即可
    $env:DEEPSEEK_API_KEY = "..."                          # 被测系统读环境变量
    & "D:/myself-prove/Dev Knowledge Agent/.venv/Scripts/python.exe" scripts/smoke_agentic_rag.py --cases 3

只读取被测系统能拿到的信息（case.question）；评测 ground-truth 绝不传入被测系统。
仅打印，不写库 —— 契约无误后再用 `run` 跑完整 benchmark。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragevalflow.integrations.agentic_rag_client import (
    AgenticRAGClient,
    _default_builder,
)
from ragevalflow.schemas.eval_case import EvalCase

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "agentic_rag_eval_cases.jsonl"
DEFAULT_WORKDIR = r"D:/myself-prove/Dev Knowledge Agent/.local/agent_live"


def load_cases(jsonl_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(EvalCase.model_validate(json.loads(line)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="真实 Agentic-RAG smoke test（只读，不写库）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--working-dir", default=DEFAULT_WORKDIR)
    parser.add_argument("--cases", type=int, default=3, help="运行前 N 条 case")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    cases = load_cases(args.dataset)[: args.cases]
    if not cases:
        print("数据集为空，无 case 可测。")
        return

    client = AgenticRAGClient(
        builder=_default_builder,
        working_dir=args.working_dir,
        model=args.model,
        pricing=None,  # 未提供定价 → estimated_cost unavailable（不伪造成本）
    )
    try:
        for case in cases:
            print("=" * 72)
            print(f"[case] {case.id} | {case.category}")
            print(f"  Q: {case.question}")
            try:
                out = client.answer(case)
            except Exception as exc:  # noqa: BLE001 - smoke 阶段如实记录失败
                print(f"  ! FAILED: {type(exc).__name__}: {exc}")
                continue
            print(f"  status : {out.config.get('status')}")
            print(f"  answer : {out.answer[:200]}{'…' if len(out.answer) > 200 else ''}")
            print(f"  docs   : {[c.doc_id for c in out.contexts]}")
            rounds = out.trace.retrieval_rounds
            print(
                f"  trace  : rounds={rounds} tools={out.trace.tools_used}"
                f" reranker={out.trace.used_reranker} reflection={out.trace.used_reflection}"
                f" rewrite={out.trace.query_rewrite!r}"
            )
            print(
                f"  runtime: latency_ms={out.runtime.latency_ms}"
                f" in_tok={out.runtime.input_tokens} out_tok={out.runtime.output_tokens}"
                f" cost={out.runtime.estimated_cost}"
            )
    finally:
        client.close()
        print("=" * 72)
        print("smoke done.")


if __name__ == "__main__":
    main()