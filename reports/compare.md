# Experiment Comparison Report

## 1. Summary

- baseline experiment: `exp_20260915_202509_9efa15`
- candidate experiment: `exp_20260915_202513_9b3d05`
- common cases: 12
- overall status: **improved**（整体改善）
- improved cases: 12 / regressed cases: 0 / unchanged: 0

## 2. Metric Deltas

| 指标 | 方向 | baseline_avg | candidate_avg | delta | status |
|---|---|---|---|---|---|
| `answer_context_overlap_score` | ↑ | 0.0740 | 0.2948 | +0.2208 | improved |
| `hit_at_1` | ↑ | 1.000 | 1.000 | +0.0000 | unchanged |
| `hit_at_5` | ↑ | 1.000 | 1.000 | +0.0000 | unchanged |
| `hit_at_3` | ↑ | 1.000 | 1.000 | +0.0000 | unchanged |
| `must_include_coverage` | ↑ | 0.8750 | 1.000 | +0.1250 | improved |
| `expected_doc_coverage` | ↑ | 0.9167 | 1.000 | +0.0833 | improved |
| `context_precision_simple` | ↑ | 1.000 | 1.000 | +0.0000 | unchanged |
| `mrr` | ↑ | 1.000 | 1.000 | +0.0000 | unchanged |
| `expected_tool_coverage` | ↑ | 0.0000 | 0.0000 | +0.0000 | unchanged |
| `citation_doc_coverage` | ↑ | 0.0000 | 1.000 | +1.0000 | improved |
| `exact_keyword_coverage` | ↑ | 0.8750 | 1.000 | +0.1250 | improved |
| `latency_ms` | ↓ | 315.583 | 145.583 | -170.0000 | improved |
| `over_retrieval` | ↓ | 0.0000 | 0.0000 | +0.0000 | unchanged |
| `total_tokens` | ↓ | 371.167 | 371.167 | +0.0000 | unchanged |
| `forbidden_tool_rate` | ↓ | 0.0000 | 0.0000 | +0.0000 | unchanged |
| `missing_multi_hop` | ↓ | 0.0833 | 0.0000 | -0.0833 | improved |
| `forbidden_claim_rate` | ↓ | 0.0000 | 0.0000 | +0.0000 | unchanged |
| `estimated_cost` | ↓ | 0.0037 | 0.0037 | +0.0000 | unchanged |

## 3. Regressed Cases

> 无退化 case。

## 4. Improved Cases

| case_id | 改善指标 | 解决的失败类型 |
|---|---|---|
| `case_fin_001` | answer_context_overlap_score:+0.318、citation_doc_coverage:+1.000、latency_ms:-170.000 | incomplete_answer |
| `case_hr_001` | answer_context_overlap_score:+0.193、must_include_coverage:+0.500、citation_doc_coverage:+1.000、exact_keyword_coverage:+0.500、latency_ms:-200.000 | incomplete_answer |
| `case_hr_002` | answer_context_overlap_score:+0.298、citation_doc_coverage:+1.000、latency_ms:-200.000 | incomplete_answer |
| `case_hr_003` | answer_context_overlap_score:+0.254、expected_doc_coverage:+0.500、citation_doc_coverage:+1.000、latency_ms:-200.000、missing_multi_hop:-1.000 | bad_ranking、incomplete_answer、missing_multi_hop、query_rewrite_missing |
| `case_hr_004` | answer_context_overlap_score:+0.250、citation_doc_coverage:+1.000、latency_ms:-200.000 | incomplete_answer |
| `case_it_001` | answer_context_overlap_score:+0.112、must_include_coverage:+0.500、expected_doc_coverage:+0.500、citation_doc_coverage:+1.000、exact_keyword_coverage:+0.500、latency_ms:-200.000 | bad_ranking、incomplete_answer |
| `case_it_002` | answer_context_overlap_score:+0.274、citation_doc_coverage:+1.000、latency_ms:-200.000 | incomplete_answer |
| `case_prod_001` | answer_context_overlap_score:+0.093、must_include_coverage:+0.500、citation_doc_coverage:+1.000、exact_keyword_coverage:+0.500、latency_ms:-110.000 | incomplete_answer |
| `case_prod_002` | answer_context_overlap_score:+0.161、citation_doc_coverage:+1.000、latency_ms:-110.000 | - |
| `case_prod_003` | answer_context_overlap_score:+0.203、citation_doc_coverage:+1.000、latency_ms:-110.000 | - |
| `case_sec_001` | answer_context_overlap_score:+0.259、citation_doc_coverage:+1.000、latency_ms:-170.000 | incomplete_answer、query_rewrite_missing |
| `case_sec_002` | answer_context_overlap_score:+0.235、citation_doc_coverage:+1.000、latency_ms:-170.000 | incomplete_answer |

## 5. Recommendation

**建议升级 candidate**：整体指标改善且无退化 case。
