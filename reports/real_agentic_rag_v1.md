# Experiment Report

## 1. Summary

- experiment_id: `exp_20260916_104752_322f92`
- name: `real_agentic_rag_v1`
- rag_version: `polaris-agentic-rag-live`
- case count: 10
- created_at: 2026-09-16T10:47:52.326606

## 2. Aggregate Metrics

> 约束为空时对应比例指标为 0.0 表示“无约束”(N/A)，不代表失败。

### 检索指标 (retrieval)

| 指标 | 平均值 |
|---|---|
| `hit_at_1` | 0.8000 |
| `hit_at_3` | 1.000 |
| `hit_at_5` | 1.000 |
| `mrr` | 0.9000 |
| `expected_doc_coverage` | 1.000 |
| `context_precision_simple` | 0.2583 |

### 答案指标 (answer)

| 指标 | 平均值 |
|---|---|
| `must_include_coverage` | 1.000 |
| `forbidden_claim_rate` | 0.4000 |
| `answer_length` | 599.000 |
| `exact_keyword_coverage` | 1.000 |

### 忠实度指标 (faithfulness)

| 指标 | 平均值 |
|---|---|
| `answer_context_overlap_score` | 0.7539 |
| `citation_doc_coverage` | 1.000 |

### Trace 指标 (agentic)

| 指标 | 平均值 |
|---|---|
| `retrieval_rounds` | 1.000 |
| `rewrite_used` | 0.8000 |
| `reranker_used` | 0.0000 |
| `reflection_used` | 0.0000 |
| `over_retrieval` | 0.0000 |
| `missing_multi_hop` | 0.0000 |
| `expected_tool_coverage` | 1.000 |
| `forbidden_tool_rate` | 0.0000 |

### 成本/运行时指标 (cost)

| 指标 | 平均值 |
|---|---|
| `latency_ms` | 22410 |
| `input_tokens` | 25001 |
| `output_tokens` | 2038 |
| `total_tokens` | 27039 |

## 3. Failure Summary

### 失败类型分布

| 失败类型 | 数量 | 最高严重度 |
|---|---|---|
| `bad_ranking` | 10 | low |
| `latency_regression` | 10 | medium |
| `forbidden_claim` | 4 | critical |

### 严重度分布

| 严重度 | 数量 |
|---|---|
| critical | 4 |
| medium | 10 |
| low | 10 |

### Top 失败样例

| case_id | 失败类型 | 原因 |
|---|---|---|
| `real_arch_components` | bad_ranking、forbidden_claim、latency_regression | 检索上下文精度较低，但已满足 must_include |
| `real_auth_scope_403` | bad_ranking、forbidden_claim、latency_regression | 检索上下文精度较低，但已满足 must_include |
| `real_deploy_bluegreen` | bad_ranking、forbidden_claim、latency_regression | 检索上下文精度较低，但已满足 must_include |
| `real_deploy_rollback` | bad_ranking、forbidden_claim、latency_regression | 检索上下文精度较低，但已满足 must_include |
| `real_arch_orderflow` | bad_ranking、latency_regression | 检索上下文精度较低，但已满足 must_include |

## 4. Case Results

| case_id | category | `hit_at_3` | `expected_doc_coverage` | `must_include_coverage` | `forbidden_claim_rate` | `answer_context_overlap_score` | `latency_ms` | `estimated_cost` | failure_types | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `real_arch_components` | arch | 1.000 | 1.000 | 1.000 | 1.000 | 0.8787 | 30623 | 0.0000 | bad_ranking、forbidden_claim、latency_regression | FAIL |
| `real_arch_orderflow` | arch | 1.000 | 1.000 | 1.000 | 0.0000 | 0.7250 | 18169 | 0.0000 | bad_ranking、latency_regression | WARN |
| `real_auth_refresh` | auth | 1.000 | 1.000 | 1.000 | 0.0000 | 0.6941 | 22264 | 0.0000 | bad_ranking、latency_regression | WARN |
| `real_auth_scope_403` | auth | 1.000 | 1.000 | 1.000 | 1.000 | 0.6749 | 13058 | 0.0000 | bad_ranking、forbidden_claim、latency_regression | FAIL |
| `real_auth_token_ttl` | auth | 1.000 | 1.000 | 1.000 | 0.0000 | 0.8365 | 24311 | 0.0000 | bad_ranking、latency_regression | WARN |
| `real_auth_verify_dep` | auth | 1.000 | 1.000 | 1.000 | 0.0000 | 0.7180 | 20002 | 0.0000 | bad_ranking、latency_regression | WARN |
| `real_deploy_bluegreen` | deploy | 1.000 | 1.000 | 1.000 | 1.000 | 0.8060 | 19250 | 0.0000 | bad_ranking、forbidden_claim、latency_regression | FAIL |
| `real_deploy_rollback` | deploy | 1.000 | 1.000 | 1.000 | 1.000 | 0.7425 | 26368 | 0.0000 | bad_ranking、forbidden_claim、latency_regression | FAIL |
| `real_incident_symptom` | incident | 1.000 | 1.000 | 1.000 | 0.0000 | 0.7795 | 26455 | 0.0000 | bad_ranking、latency_regression | WARN |
| `real_incident_traceid` | incident | 1.000 | 1.000 | 1.000 | 0.0000 | 0.6836 | 23598 | 0.0000 | bad_ranking、latency_regression | WARN |

## 5. Recommendations

- 存在 10 个 partial retrieval / 低上下文精度 → 建议优化检索召回与排序。
- 检测到 forbidden claim / forbidden tool → 建议加强内容与工具调用护栏。
- 存在 10 个高延迟 → 建议优化检索轮数 / 上下文长度 / 链路延迟。

## 6. Limitations

- 当前为 deterministic metrics，全部结果可复现，不依赖任何模型判定。
- 不含 LLM judge（未对答案做模型打分/偏好评估）。
- 不含 Ragas / DeepEval 集成。
- MockRAGClient 仅用于本地评测流程验证，不等价于真实 RAG 系统表现。
- 失败归因基于规则阈值（如 overlap < 0.1、latency > 3000ms 等），可能出现边界误判。
