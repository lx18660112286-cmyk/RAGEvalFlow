# Experiment Report

## 1. Summary

- experiment_id: `exp_20260915_202513_9b3d05`
- name: `agentic_v1`
- rag_version: `mock-agentic-v1`
- case count: 12
- created_at: 2026-09-15T20:25:13.727200

## 2. Aggregate Metrics

> 约束为空时对应比例指标为 0.0 表示“无约束”(N/A)，不代表失败。

### 检索指标 (retrieval)

| 指标 | 平均值 |
|---|---|
| `hit_at_1` | 1.000 |
| `hit_at_3` | 1.000 |
| `hit_at_5` | 1.000 |
| `mrr` | 1.000 |
| `expected_doc_coverage` | 1.000 |
| `context_precision_simple` | 1.000 |

### 答案指标 (answer)

| 指标 | 平均值 |
|---|---|
| `must_include_coverage` | 1.000 |
| `forbidden_claim_rate` | 0.0000 |
| `answer_length` | 51.667 |
| `exact_keyword_coverage` | 1.000 |

### 忠实度指标 (faithfulness)

| 指标 | 平均值 |
|---|---|
| `answer_context_overlap_score` | 0.2948 |
| `citation_doc_coverage` | 1.000 |

### Trace 指标 (agentic)

| 指标 | 平均值 |
|---|---|
| `retrieval_rounds` | 1.083 |
| `rewrite_used` | 0.1667 |
| `reranker_used` | 1.000 |
| `reflection_used` | 1.000 |
| `over_retrieval` | 0.0000 |
| `missing_multi_hop` | 0.0000 |
| `expected_tool_coverage` | 0.0000 |
| `forbidden_tool_rate` | 0.0000 |

### 成本/运行时指标 (cost)

| 指标 | 平均值 |
|---|---|
| `latency_ms` | 145.583 |
| `input_tokens` | 225.583 |
| `output_tokens` | 145.583 |
| `estimated_cost` | 0.0037 |
| `total_tokens` | 371.167 |

## 3. Failure Summary

> 未检测到失败归因（或所有相关约束为空，视为 N/A）。

## 4. Case Results

| case_id | category | `hit_at_3` | `expected_doc_coverage` | `must_include_coverage` | `forbidden_claim_rate` | `answer_context_overlap_score` | `latency_ms` | `estimated_cost` | failure_types | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `case_fin_001` | 财务 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.3421 | 164.000 | 0.0039 | 无 | OK |
| `case_hr_001` | HR | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2195 | 125.000 | 0.0033 | 无 | OK |
| `case_hr_002` | HR | 1.000 | 1.000 | 1.000 | 0.0000 | 0.3235 | 126.000 | 0.0033 | 无 | OK |
| `case_hr_003` | HR | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2807 | 127.000 | 0.0033 | 无 | OK |
| `case_hr_004` | HR | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2727 | 128.000 | 0.0034 | 无 | OK |
| `case_it_001` | IT | 1.000 | 1.000 | 1.000 | 0.0000 | 0.1818 | 128.000 | 0.0034 | 无 | OK |
| `case_it_002` | IT | 1.000 | 1.000 | 1.000 | 0.0000 | 0.3030 | 129.000 | 0.0034 | 无 | OK |
| `case_prod_001` | 产品 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2727 | 164.000 | 0.0043 | 无 | OK |
| `case_prod_002` | 产品 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.3654 | 165.000 | 0.0043 | 无 | OK |
| `case_prod_003` | 产品 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.4146 | 166.000 | 0.0043 | 无 | OK |
| `case_sec_001` | 安全 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2826 | 162.000 | 0.0038 | 无 | OK |
| `case_sec_002` | 安全 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.2791 | 163.000 | 0.0039 | 无 | OK |

## 5. Recommendations

- 未发现显著失败，当前配置表现稳定（可维持）。

## 6. Limitations

- 当前为 deterministic metrics，全部结果可复现，不依赖任何模型判定。
- 不含 LLM judge（未对答案做模型打分/偏好评估）。
- 不含 Ragas / DeepEval 集成。
- MockRAGClient 仅用于本地评测流程验证，不等价于真实 RAG 系统表现。
- 失败归因基于规则阈值（如 overlap < 0.1、latency > 3000ms 等），可能出现边界误判。
