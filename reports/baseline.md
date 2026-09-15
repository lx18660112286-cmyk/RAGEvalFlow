# Experiment Report

## 1. Summary

- experiment_id: `exp_20260915_202509_9efa15`
- name: `baseline`
- rag_version: `mock-baseline-v0`
- case count: 12
- created_at: 2026-09-15T20:25:09.240033

## 2. Aggregate Metrics

> 约束为空时对应比例指标为 0.0 表示“无约束”(N/A)，不代表失败。

### 检索指标 (retrieval)

| 指标 | 平均值 |
|---|---|
| `hit_at_1` | 1.000 |
| `hit_at_3` | 1.000 |
| `hit_at_5` | 1.000 |
| `mrr` | 1.000 |
| `expected_doc_coverage` | 0.9167 |
| `context_precision_simple` | 1.000 |

### 答案指标 (answer)

| 指标 | 平均值 |
|---|---|
| `must_include_coverage` | 0.8750 |
| `forbidden_claim_rate` | 0.0000 |
| `answer_length` | 45.333 |
| `exact_keyword_coverage` | 0.8750 |

### 忠实度指标 (faithfulness)

| 指标 | 平均值 |
|---|---|
| `answer_context_overlap_score` | 0.0740 |
| `citation_doc_coverage` | 0.0000 |

### Trace 指标 (agentic)

| 指标 | 平均值 |
|---|---|
| `retrieval_rounds` | 1.000 |
| `rewrite_used` | 0.0000 |
| `reranker_used` | 0.0000 |
| `reflection_used` | 0.0000 |
| `over_retrieval` | 0.0000 |
| `missing_multi_hop` | 0.0833 |
| `expected_tool_coverage` | 0.0000 |
| `forbidden_tool_rate` | 0.0000 |

### 成本/运行时指标 (cost)

| 指标 | 平均值 |
|---|---|
| `latency_ms` | 315.583 |
| `input_tokens` | 225.583 |
| `output_tokens` | 145.583 |
| `estimated_cost` | 0.0037 |
| `total_tokens` | 371.167 |

## 3. Failure Summary

### 失败类型分布

| 失败类型 | 数量 | 最高严重度 |
|---|---|---|
| `incomplete_answer` | 10 | high |
| `bad_ranking` | 2 | medium |
| `query_rewrite_missing` | 2 | medium |
| `missing_multi_hop` | 1 | high |

### 严重度分布

| 严重度 | 数量 |
|---|---|
| high | 4 |
| medium | 11 |

### Top 失败样例

| case_id | 失败类型 | 原因 |
|---|---|---|
| `case_hr_001` | incomplete_answer | 答案缺少期望包含的必需内容（must_include 未完全覆盖）；答案与检索上下文重叠度过低，存在低忠实度/不支持（幻觉）风险 |
| `case_hr_003` | bad_ranking、incomplete_answer、query_rewrite_missing、missing_multi_hop | 部分期望文档未被检索命中（partial retrieval） |
| `case_it_001` | bad_ranking、incomplete_answer | 部分期望文档未被检索命中（partial retrieval） |
| `case_prod_001` | incomplete_answer | 答案缺少期望包含的必需内容（must_include 未完全覆盖） |
| `case_fin_001` | incomplete_answer | 答案与检索上下文重叠度过低，存在低忠实度/不支持（幻觉）风险 |

## 4. Case Results

| case_id | category | `hit_at_3` | `expected_doc_coverage` | `must_include_coverage` | `forbidden_claim_rate` | `answer_context_overlap_score` | `latency_ms` | `estimated_cost` | failure_types | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `case_fin_001` | 财务 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0244 | 334.000 | 0.0039 | incomplete_answer | WARN |
| `case_hr_001` | HR | 1.000 | 1.000 | 0.5000 | 0.0000 | 0.0263 | 325.000 | 0.0033 | incomplete_answer | FAIL |
| `case_hr_002` | HR | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0256 | 326.000 | 0.0033 | incomplete_answer | WARN |
| `case_hr_003` | HR | 1.000 | 0.5000 | 1.000 | 0.0000 | 0.0270 | 327.000 | 0.0033 | bad_ranking、incomplete_answer、query_rewrite_missing、missing_multi_hop | FAIL |
| `case_hr_004` | HR | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0227 | 328.000 | 0.0034 | incomplete_answer | WARN |
| `case_it_001` | IT | 1.000 | 0.5000 | 0.5000 | 0.0000 | 0.0698 | 328.000 | 0.0034 | bad_ranking、incomplete_answer | FAIL |
| `case_it_002` | IT | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0286 | 329.000 | 0.0034 | incomplete_answer | WARN |
| `case_prod_001` | 产品 | 1.000 | 1.000 | 0.5000 | 0.0000 | 0.1800 | 274.000 | 0.0043 | incomplete_answer | FAIL |
| `case_prod_002` | 产品 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2041 | 275.000 | 0.0043 | 无 | OK |
| `case_prod_003` | 产品 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2115 | 276.000 | 0.0043 | 无 | OK |
| `case_sec_001` | 安全 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0238 | 332.000 | 0.0038 | incomplete_answer、query_rewrite_missing | WARN |
| `case_sec_002` | 安全 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0444 | 333.000 | 0.0039 | incomplete_answer | WARN |

## 5. Recommendations

- 存在 2 个 partial retrieval / 低上下文精度 → 建议优化检索召回与排序。
- 存在 10 个 incomplete_answer（must_include 缺失或低忠实度）→ 建议改善答案合成 prompt，并加强基于检索上下文的引用 grounding。
- 存在 1 个 missing multi-hop → 建议改善 planner / 查询分解。
- 存在 2 个 query rewrite 缺失 → 建议提升查询理解与改写能力。

## 6. Limitations

- 当前为 deterministic metrics，全部结果可复现，不依赖任何模型判定。
- 不含 LLM judge（未对答案做模型打分/偏好评估）。
- 不含 Ragas / DeepEval 集成。
- MockRAGClient 仅用于本地评测流程验证，不等价于真实 RAG 系统表现。
- 失败归因基于规则阈值（如 overlap < 0.1、latency > 3000ms 等），可能出现边界误判。
