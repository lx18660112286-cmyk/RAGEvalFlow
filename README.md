# RAGEvalFlow

面向 **Agentic-RAG 应用**的自动评测、失败诊断与优化建议平台。

> 它不生产 RAG，也不只是调用 Ragas 输出几个分数——而是用于**评测已有 RAG / Agentic-RAG 系统**的评测工作流工具：
> 数据集管理 → 实验运行 → 确定性指标 → 失败归因 → 回归检测 → Markdown 实验报告。

当前仓库为 **阶段 4**：在阶段 1（脚手架/存储/数据集/CLI）+ 阶段 2（确定性指标/MockRAGClient/Runner/`run`）+ 阶段 3（失败归因/回归检测/Markdown 报告/`report`/`compare`）之上，新增 **Streamlit Dashboard 展示层**，可完成「数据集 → mock RAG → 指标 → 失败归因 → 回归对比 → Markdown 报告 → Dashboard 可视化」完整闭环。

---

## 技术栈

Python 3.11+ · Pydantic v2 · SQLAlchemy 2.0 + SQLite · PyYAML · pandas · Streamlit · pytest

> 真实 RAG API 客户端（httpx）留待后续阶段。
> 不实现 FastAPI、Ragas、DeepEval、LLM judge，不引入 Redis / Kafka / Celery / K8s / 多租户，不自动读取 `.env`。

## 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

## 快速开始（阶段 4 Quickstart）

```bash
python -m ragevalflow.main init-db
python -m ragevalflow.main dataset import datasets/eval_cases.jsonl
python -m ragevalflow.main run --config configs/baseline.yaml
python -m ragevalflow.main run --config configs/agentic_v1.yaml

streamlit run ragevalflow/ui/streamlit_app.py
```

`run` 命令流程：读取 YAML 配置 → 创建实验（同名失败）→ 用 MockRAGClient 对每条样例作答 → 计算全部确定性指标 → **失败归因（写入 failure_types / failure_findings）** → case_results 事务化落库 → 输出平均指标与失败汇总。
`report` 生成单实验 Markdown 报告；`compare` 生成 baseline vs candidate 回归对比报告（无共同 case_id、找不到实验、无 case_results 均中文报错且退出码非 0）。

### 阶段 4 Dashboard（Streamlit）

```bash
streamlit run ragevalflow/ui/streamlit_app.py
```

只做展示层，复用现有评测 / 对比 / 报告逻辑，不新增评测能力。含 5 个区域：

| 页签 | 内容 |
|---|---|
| Overview | 数据库路径、eval_cases / experiments / case_results 计数、最新实验、报告文件提示 |
| Experiments | 实验列表（含 case_results 数、avg_hit_at_3 / expected_doc_coverage / must_include_coverage / answer_context_overlap_score / latency_ms、failure_total） |
| Experiment Detail | 单实验聚合指标、失败类型分布 / 严重度分布柱状图、case 结果表 |
| Case Detail | 单个 case 的 question / reference_answer / expected_docs / must_include / must_not_include / RAG 答案 / contexts / trace / metrics / findings |
| Compare | 复用 `regression.compare_experiments` 的 overall_status、metric deltas、regressed / improved cases |

Sidebar 可输入数据库路径（优先级：输入 > 环境变量 `RAGEFLOW_DB` > `./data/ragevalflow.db`），并提供「Generate Experiment Report」「Generate Comparison Report」按钮调用现有 `markdown_report` 生成器。
数据库不存在 / 未初始化时页面显示中文提示，不抛 stack trace。

### 阶段 3 Quickstart（历史）

```bash
python -m ragevalflow.main run --config configs/baseline.yaml
python -m ragevalflow.main run --config configs/agentic_v1.yaml

python -m ragevalflow.main report --experiment baseline --output reports/baseline.md
python -m ragevalflow.main report --experiment agentic_v1 --output reports/agentic_v1.md
python -m ragevalflow.main compare --baseline baseline --candidate agentic_v1 --output reports/compare.md
```

### 历史 Quickstart（阶段 1）

```bash
python -m ragevalflow.main experiments create --config configs/baseline.yaml
python -m ragevalflow.main experiments list
python -m ragevalflow.main dataset list
python -m ragevalflow.main dataset export out.jsonl
```

### 数据库路径

优先级：`CLI --db` > 环境变量 `RAGEFLOW_DB` > 默认 `./data/ragevalflow.db`。

```bash
python -m ragevalflow.main --db my_data/eval.db init-db
# 或
$env:RAGEFLOW_DB = "my_data/eval.db"   # PowerShell
export RAGEFLOW_DB="my_data/eval.db"   # bash
python -m ragevalflow.main dataset list
```

> 说明：**不自动读取 `.env`**（未引入 python-dotenv），`.env.example` 仅作为手动设置环境变量的示例参考。

## CLI 命令一览

| 命令 | 说明 |
|---|---|
| `init-db [--db]` | 初始化 SQLite 数据库与全部表（幂等） |
| `run --config FILE [--name] [--db]` | 运行一次评测实验（Mock RAG + 确定性指标 + 失败归因，阶段 2/3） |
| `report --experiment ID_OR_NAME [--output] [--db]` | 生成单实验 Markdown 报告（阶段 3） |
| `compare --baseline B --candidate C [--output] [--db]` | 生成 baseline vs candidate 回归对比报告（阶段 3） |
| `dataset list [--db]` | 列出已导入的评测样例 |
| `dataset import FILE [--db] [--dry-run]` | 从 JSONL 导入（事务语义，`--dry-run` 只校验不写库） |
| `dataset export OUTPUT [--db]` | 将库中样例导出为 JSONL |
| `experiments create --config FILE [--name] [--db]` | 从 YAML 创建实验（`--name` > YAML `experiment_name` > YAML `name`；同名报错） |
| `experiments list [--db]` | 列出全部实验 |

## 运行测试

```bash
pytest
```

## 确定性指标（阶段 2）

全部为纯规则计算，不调用 LLM judge、不联网、结果可复现。统一策略：约束列表为空时相关比例指标返回 `0.0`（报告中以 N/A 注明，不代表失败）。

| 类别 | 指标 |
|---|---|
| 检索（retrieval） | `hit_at_1/3/5`、`mrr`、`expected_doc_coverage`、`context_precision_simple` |
| 答案（answer） | `must_include_coverage`、`forbidden_claim_rate`、`answer_length`、`exact_keyword_coverage` |
| 忠实度（faithfulness） | `answer_context_overlap_score`、`citation_doc_coverage` |
| trace（agentic） | `retrieval_rounds`、`rewrite_used`、`reranker_used`、`reflection_used`、`over_retrieval`、`missing_multi_hop`、`expected_tool_coverage`、`forbidden_tool_rate` |
| 成本（cost） | `latency_ms`、`input_tokens`、`output_tokens`、`estimated_cost`、`total_tokens` |

`compute_all_metrics(case, output)` 聚合为单一扁平 dict，写入 `case_results.metrics`。

## 失败归因（阶段 3）

`ragevalflow/analysis/failure_analyzer.py::analyze_failures(case, output, metrics)` 基于规则输出 `FailureFinding[]`（failure_type / severity / message / evidence / metric_values），复用既有 13 类 `FailureType`，把建议规则适配到枚举名：

| 建议规则 | 归因类型 | 默认 severity |
|---|---|---|
| 期望文档全未命中（hit_at_3=0） | `retrieval_miss` | high |
| 部分命中（coverage<1.0）/ 低上下文精度 | `bad_ranking` | medium（must_include 已满足时 low） |
| must_include 未全覆盖 | `incomplete_answer` | high |
| 答案与上下文重叠过低（<0.1） | `incomplete_answer`（低忠实度） | medium |
| 含禁止内容 | `forbidden_claim` | critical |
| 应重写未重写 | `query_rewrite_missing` | medium |
| 应多跳未多跳 | `missing_multi_hop` | high |
| 检索轮数超限 | `over_retrieval` | medium |
| 期望工具未调用 | `expected_tool_missing` | high |
| 调用禁止工具 | `forbidden_tool_called` | critical |
| 延迟 > 3000ms / 成本 > 0.02 | `latency_regression` / `cost_regression` | medium |

> 阈值可通过 `analyze_failures(latency_threshold_ms=..., cost_threshold=...)` 配置；空约束不触发任何失败。

## 回归检测（阶段 3）

`ragevalflow/analysis/regression.py::compare_experiments(baseline_results, candidate_results, thresholds=None)`：

- 只比较共同 case_id；baseline 独有 → `missing_candidate_case`，candidate 独有 → `new_candidate_case`；
- 指标默认方向：`higher_is_better`（hit_at_*/mrr/coverage/overlap/citation/expected_tool_coverage 等）、`lower_is_better`（forbidden_claim_rate/latency/estimated_cost/total_tokens/over_retrieval/missing_multi_hop/forbidden_tool_rate）；
- 默认阈值：绝对下降 ≥ 0.05、相对下降 ≥ 10%、延迟增幅 ≥ 20%、成本增幅 ≥ 20%、新增失败类型 > 0；
- 输出 `RegressionReport`（summary / metric_deltas / case_deltas）。

## MockRAGClient（阶段 2）

`ragevalflow/integrations/rag_client.py` 提供 `MockRAGClient`，两种确定性模式（不调用 LLM、不联网）：

- `baseline`：命中约一半期望文档、答案只覆盖一半必需关键词、无 rewrite / multi-hop / reranker / reflection；
- `agentic_v1`：命中全部期望文档、覆盖全部必需关键词、模拟引用标注、按 `expected_behavior` 生成更合理的 trace（rewrite / multi-hop / reranker / reflection / tools）。

> 注意：baseline 与 agentic_v1 的 `avg_hit_at_3` 均为 1.0，对比二者差异请优先看 `expected_doc_coverage` / `must_include_coverage` / `answer_context_overlap_score` / `latency_ms`。

从实验 YAML 的 `config` 段读取，例如：

```yaml
config:
  client: mock
  mock_mode: agentic_v1
  top_k: 5
```

## 数据模型（阶段 1）

- **EvalCase**：评价样例（id / category / question / reference_answer / expected_docs / must_include / must_not_include / expected_behavior）
- **ExpectedBehavior**：Agentic 行为期望（should_rewrite / should_multi_hop / max_retrieval_rounds / expected_tools / forbidden_tools）
- **RAGOutput**：RAG 系统输出（answer / contexts / trace / runtime / config），`contexts` 条数不限（展示层自行截断）
- **Trace / RuntimeInfo**：执行轨迹与运行时信息（延迟 / token / 成本）
- **Experiment / CaseResult**：实验与单案例结果（阶段 3 起 `failure_types` / `failure_findings` 保存失败归因）
- **FailureType / FailureFinding**：13 类失败归因类型与归因条目（failure_type / severity / message / evidence / metric_values）

### SQLite 表结构（3 张用户表）

| 表 | 说明 |
|---|---|
| `eval_cases` | 评测样例，`id` 主键 |
| `experiments` | 实验，`name` 唯一约束，`config` 列保存完整 YAML 原文 |
| `case_results` | 单案例结果（metrics / rag_output / failure_types / failure_findings JSON；旧库自动迁移增加 failure_findings 列） |

## 项目结构

```
RAGEvalFlow/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/                 # 实验 YAML 配置（run / experiments create 使用）
│   ├── baseline.yaml
│   └── agentic_v1.yaml
├── datasets/
│   └── eval_cases.jsonl     # 12 条内置评测样例
├── ragevalflow/
│   ├── main.py              # CLI 入口（init-db/run/report/compare/dataset/experiments）
│   ├── ui/                  # Streamlit Dashboard（阶段 4）
│   │   ├── streamlit_app.py # 页面（Overview/Experiments/Detail/Case/Compare）
│   │   └── data_loader.py   # 纯数据读取/聚合/报告生成（不依赖 streamlit，可单测）
│   ├── integrations/
│   │   └── rag_client.py    # RAGClient 协议 + MockRAGClient（阶段 2）
│   ├── metrics/             # 确定性指标（阶段 2）
│   ├── analysis/            # 失败归因 + 回归检测（阶段 3）
│   │   ├── failure_analyzer.py
│   │   └── regression.py
│   ├── reporting/           # Markdown 报告（阶段 3）
│   │   └── markdown_report.py
│   ├── schemas/             # Pydantic v2 数据模型
│   └── core/
│       ├── storage.py       # SQLAlchemy 2.0 + SQLite 存储层
│       ├── dataset.py       # JSONL 导入导出（事务）+ YAML 实验配置
│       └── runner.py        # 实验运行器（指标 + 失败归因）
└── tests/                   # pytest 单元测试
```

## 阶段规划

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 1 | 脚手架、Pydantic 模型、SQLite 存储、数据集导入导出、CLI | ✅ 已完成 |
| 阶段 2 | 确定性指标 + MockRAGClient + Runner + `run` 命令 | ✅ 已完成 |
| 阶段 3 | 失败归因 + 回归检测 + Markdown 报告 + `compare`/`report` 命令 | ✅ 已完成 |
| 阶段 4 | Streamlit Dashboard（展示层） | ✅ 已完成 |

### 仍未实现（真实外部集成）

- ❌ FastAPI
- ❌ Ragas / DeepEval adapter
- ❌ LLM judge
- ❌ 真实外部 RAG API 调用（httpx）
- ❌ 自动读取 `.env`