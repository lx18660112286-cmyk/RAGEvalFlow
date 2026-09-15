"""RAGEvalFlow Dashboard（阶段 4）。

启动：
    streamlit run ragevalflow/ui/streamlit_app.py

只做展示层，复用现有评测/对比/报告逻辑，不新增评测能力。
5 个区域：Overview / Experiments / Experiment Detail / Case Detail / Compare。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ragevalflow.ui import data_loader as dl

st.set_page_config(page_title="RAGEvalFlow Dashboard", page_icon="🧪", layout="wide")
st.title("RAGEvalFlow · 实验评测 Dashboard")

# ---------------------------------------------------------------------------
# Sidebar：数据库路径 + 报告生成
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("数据库")
    db_input = st.text_input("数据库路径", value=str(dl.DEFAULT_DB))
    db_path = dl.resolve_db_path(db_input)
    st.caption("优先级：本输入 > 环境变量 RAGEFLOW_DB > ./data/ragevalflow.db")

    st.divider()
    st.header("报告生成")
    exp_rows = dl.load_experiments(db_path)
    exp_options = [e["experiment_id"] for e in exp_rows]
    if exp_options:
        single = st.selectbox(
            "单实验报告",
            exp_options,
            key="side_exp",
            format_func=lambda i: f"{i} ({next(e['name'] for e in exp_rows if e['experiment_id'] == i)})",
        )
        if st.button("Generate Experiment Report", key="side_gen_single"):
            res = dl.generate_experiment_report(db_path, single)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success(f"已生成：`{res['path']}`")
        if len(exp_options) >= 2:
            b = st.selectbox("对比 baseline", exp_options, index=0, key="side_base", format_func=lambda i: i)
            c = st.selectbox("对比 candidate", exp_options, index=len(exp_options) - 1, key="side_cand", format_func=lambda i: i)
            if st.button("Generate Comparison Report", key="side_gen_cmp"):
                res = dl.generate_comparison_report(db_path, b, c)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success(f"已生成：`{res['path']}`")
    else:
        st.info("暂无实验")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_exp, tab_detail, tab_case, tab_cmp = st.tabs(
    ["1. Overview", "2. Experiments", "3. Experiment Detail", "4. Case Detail", "5. Compare"]
)

# ---- 1. Overview ----
with tab_overview:
    overview = dl.load_overview(db_path)
    if "error" in overview:
        st.error(f"⚠️ {overview['error']}")
        st.markdown(
            "请先初始化数据库并运行实验：\n\n"
            "```bash\n"
            "python -m ragevalflow.main init-db\n"
            "python -m ragevalflow.main dataset import datasets/eval_cases.jsonl\n"
            "python -m ragevalflow.main run --config configs/baseline.yaml\n"
            "python -m ragevalflow.main run --config configs/agentic_v1.yaml\n"
            "```"
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("eval_cases", overview["eval_cases"])
        c2.metric("experiments", overview["experiments"])
        c3.metric("case_results", overview["case_results"])
        latest = overview.get("latest_experiment")
        c4.metric("最新实验", latest["name"] if latest else "—")

        st.markdown(f"**数据库路径**：`{overview['db_path']}`")
        if latest:
            st.markdown(
                f"- 最新实验：`{latest['name']}` (`{latest['experiment_id']}` · rag_version={latest['rag_version']})"
            )
        st.markdown("**Markdown 报告状态**")
        for hint in overview["report_hints"]:
            st.markdown(f"- {hint}")

# ---- 2. Experiments ----
with tab_exp:
    exp_rows = dl.load_experiments(db_path)
    if not exp_rows:
        st.info("暂无实验数据，请先运行实验。")
    else:
        st.subheader("实验列表")
        st.dataframe(pd.DataFrame(exp_rows), width="stretch", hide_index=True)

# ---- 3. Experiment Detail ----
with tab_detail:
    exp_rows = dl.load_experiments(db_path)
    if not exp_rows:
        st.info("暂无实验数据。")
    else:
        exp_options = [e["experiment_id"] for e in exp_rows]
        exp_id = st.selectbox(
            "选择实验",
            exp_options,
            key="detail_exp",
            format_func=lambda i: f"{i} ({next(e['name'] for e in exp_rows if e['experiment_id'] == i)})",
        )
        detail = dl.load_experiment_detail(db_path, exp_id)
        if "error" in detail:
            st.error(detail["error"])
        else:
            st.subheader(f"实验 `{detail['name']}`（{detail['experiment_id']}）")
            c1, c2, c3 = st.columns(3)
            c1.metric("case_results", len(detail["case_results"]))
            c2.metric("failure_total", sum(1 for row in detail["case_results"] if row["status"] != "OK"))
            c3.metric("created_at", detail["created_at"])

            st.markdown("### 聚合指标")
            agg = detail["aggregate_metrics"]
            keys = [
                ("avg_hit_at_3", "hit_at_3"),
                ("avg_expected_doc_coverage", "expected_doc_coverage"),
                ("avg_must_include_coverage", "must_include_coverage"),
                ("avg_answer_context_overlap_score", "answer_context_overlap_score"),
                ("avg_latency_ms", "latency_ms"),
                ("forbidden_claim_rate", "forbidden_claim_rate"),
            ]
            cols = st.columns(len(keys))
            for col, (label, key) in zip(cols, keys):
                col.metric(label, f"{agg.get(key, 0.0):.4f}")

            st.markdown("### 失败类型分布")
            ftd = detail["failure_type_dist"]
            if ftd:
                st.bar_chart(pd.DataFrame(ftd).set_index("failure_type"))
            else:
                st.success("未检测到失败归因（或相关约束为空，视为 N/A）。")

            st.markdown("### 严重度分布")
            sd = detail["severity_dist"]
            if sd:
                st.bar_chart(pd.DataFrame(sd).set_index("severity"))
            else:
                st.caption("无失败归因明细。")

            st.markdown("### Case 结果表")
            st.dataframe(pd.DataFrame(detail["case_results"]), width="stretch", hide_index=True)

# ---- 4. Case Detail ----
with tab_case:
    exp_rows = dl.load_experiments(db_path)
    if not exp_rows:
        st.info("暂无实验数据。")
    else:
        exp_options = [e["experiment_id"] for e in exp_rows]
        exp_id = st.selectbox(
            "选择实验",
            exp_options,
            key="case_exp",
            format_func=lambda i: f"{i} ({next(e['name'] for e in exp_rows if e['experiment_id'] == i)})",
        )
        case_ids = dl.list_case_ids(db_path, exp_id)
        if not case_ids:
            st.info("该实验没有 case_results。")
        else:
            case_id = st.selectbox("选择 case", case_ids, key="case_sel")
            st.subheader(f"Case 明细：`{case_id}`")
            cd = dl.load_case_detail(db_path, exp_id, case_id)
            if "error" in cd:
                st.error(cd["error"])
            else:
                col_q, col_cat = st.columns([4, 1])
                col_q.markdown(f"**问题**：{cd['question']}")
                col_cat.metric("category", cd["category"])

                st.markdown("**参考答案**")
                st.write(cd["reference_answer"] or "（无）")
                col_e, col_m, col_mn = st.columns(3)
                col_e.markdown("**expected_docs**")
                col_e.write(cd["expected_docs"] or "（无）")
                col_m.markdown("**must_include**")
                col_m.write(cd["must_include"] or "（无）")
                col_mn.markdown("**must_not_include**")
                col_mn.write(cd["must_not_include"] or "（无）")

                st.markdown("**RAG 答案**")
                st.write(cd["answer"])

                with st.expander("检索上下文 (contexts)"):
                    if cd["contexts"]:
                        st.dataframe(pd.DataFrame(cd["contexts"]), width="stretch", hide_index=True)
                    else:
                        st.caption("（无上下文）")

                with st.expander("执行轨迹 (trace)"):
                    st.json(cd["trace"])
                with st.expander("运行时 (runtime)"):
                    st.json(cd["runtime"])

                st.markdown("**指标 (metrics)**")
                st.json(cd["metrics"])

                st.markdown("**失败归因 (findings)**")
                if cd["findings"]:
                    for f in cd["findings"]:
                        sev = f["severity"]
                        color = {"critical": ":red[", "high": ":orange[", "medium": ":yellow[", "low": ":blue["}.get(sev, "")
                        st.markdown(f"- **{color}{f['failure_type']}**{']' if color else ''} · severity=`{sev}`")
                        st.caption(f"message: {f['message']}")
                        st.caption(f"evidence: {f['evidence']}")
                else:
                    st.success("✅ 无失败归因 —— 该 case 通过。")

# ---- 5. Compare ----
with tab_cmp:
    exp_rows = dl.load_experiments(db_path)
    if len(exp_rows) < 2:
        st.info("需要至少 2 个实验才能进行对比。")
    else:
        exp_options = [e["experiment_id"] for e in exp_rows]
        base_id = st.selectbox("baseline 实验", exp_options, index=0, key="cmp_base")
        cand_id = st.selectbox(
            "candidate 实验",
            exp_options,
            index=len(exp_options) - 1,
            key="cmp_cand",
        )
        if st.button("运行对比 (Compare)", key="cmp_run", type="primary"):
            st.session_state["cmp_result"] = dl.run_comparison(db_path, base_id, cand_id)

        res = st.session_state.get("cmp_result")
        if res is not None:
            if "error" in res:
                st.error(res["error"])
            else:
                data = dl.comparison_to_dicts(res["report"])
                overall = data["overall_status"]
                color_map = {"improved": ":green[", "regressed": ":red[", "mixed": ":orange[", "unchanged": ":gray["}
                st.markdown(f"### 总体状态：{color_map.get(overall, '')}**{overall}**{']' if overall in color_map else ''}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("improved_cases", data["improved_cases"])
                c2.metric("regressed_cases", data["regressed_cases"])
                c3.metric("unchanged_cases", data["unchanged_cases"])
                c4.metric("common_cases", data["common_cases"])

                st.markdown("### Metric Deltas")
                if data["metric_deltas"]:
                    st.dataframe(pd.DataFrame(data["metric_deltas"]), width="stretch", hide_index=True)
                else:
                    st.caption("（无共同可判定指标变化）")

                st.markdown("### Regressed Cases")
                if data["regressed_case_rows"]:
                    st.dataframe(pd.DataFrame(data["regressed_case_rows"]), width="stretch", hide_index=True)
                else:
                    st.success("✅ 无退化 case。")

                st.markdown("### Improved Cases")
                if data["improved_case_rows"]:
                    st.dataframe(pd.DataFrame(data["improved_case_rows"]), width="stretch", hide_index=True)
                else:
                    st.caption("（无改善 case）")
