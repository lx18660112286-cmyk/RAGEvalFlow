"""RAGEvalFlow CLI 入口。

阶段 1 支持命令（统一子命令风格）：
    python -m ragevalflow.main init-db [--db PATH]
    python -m ragevalflow.main dataset list [--db PATH]
    python -m ragevalflow.main dataset import FILE [--db PATH] [--dry-run]
    python -m ragevalflow.main dataset export OUTPUT [--db PATH]
    python -m ragevalflow.main experiments create --config FILE [--name NAME] [--db PATH]
    python -m ragevalflow.main experiments list [--db PATH]

阶段 2 新增：
    python -m ragevalflow.main run --config FILE [--name NAME] [--db PATH]

阶段 3 新增：
    python -m ragevalflow.main report --experiment ID_OR_NAME [--output PATH] [--db PATH]
    python -m ragevalflow.main compare --baseline ID_OR_NAME --candidate ID_OR_NAME [--output PATH] [--db PATH]

数据库路径优先级：CLI --db > 环境变量 RAGEFLOW_DB > 默认 ./data/ragevalflow.db
（本项目不自动读取 .env。）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ragevalflow.analysis.regression import ComparisonError, compare_experiments
from ragevalflow.core import dataset as ds_lib
from ragevalflow.core.runner import RunnerError, run_experiment
from ragevalflow.core.storage import DuplicateExperimentError, Storage
from ragevalflow.reporting.markdown_report import generate_comparison_report, generate_experiment_report

DEFAULT_DB = Path("data") / "ragevalflow.db"


def resolve_db_path(cli_db: str | None) -> Path:
    """数据库路径优先级：CLI --db > 环境变量 RAGEFLOW_DB > 默认 ./data/ragevalflow.db。"""
    if cli_db:
        return Path(cli_db)
    env_db = os.environ.get("RAGEFLOW_DB")
    if env_db:
        return Path(env_db)
    return DEFAULT_DB


def add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite 数据库路径（默认取环境变量 RAGEFLOW_DB，均未设置时用 ./data/ragevalflow.db）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragevalflow.main",
        description="RAGEvalFlow：面向 Agentic-RAG 应用的自动评测、失败诊断与优化建议平台（阶段 1）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- init-db ----
    p_init = sub.add_parser("init-db", help="初始化 SQLite 数据库与全部表（幂等）")
    add_db_arg(p_init)

    # ---- run ----
    p_run = sub.add_parser("run", help="运行一次评测实验（Mock RAG + 确定性指标 + 失败归因）")
    p_run.add_argument("--config", required=True, help="实验 YAML 配置文件路径")
    p_run.add_argument("--name", default=None, help="实验名（优先级：--name > YAML experiment_name > YAML name）")
    add_db_arg(p_run)

    # ---- report ----
    p_report = sub.add_parser("report", help="生成单实验 Markdown 报告（阶段 3）")
    p_report.add_argument("--experiment", required=True, help="experiment_id 或实验名")
    p_report.add_argument("--output", default=None, help="输出 Markdown 路径（默认 reports/{name}_{id}.md）")
    add_db_arg(p_report)

    # ---- compare ----
    p_compare = sub.add_parser("compare", help="对比 baseline 与 candidate 实验并生成 Markdown 报告（阶段 3）")
    p_compare.add_argument("--baseline", required=True, help="baseline 的 experiment_id 或实验名")
    p_compare.add_argument("--candidate", required=True, help="candidate 的 experiment_id 或实验名")
    p_compare.add_argument("--output", default=None, help="输出 Markdown 路径（默认 reports/compare_{b}_{c}.md）")
    add_db_arg(p_compare)

    # ---- dataset ----
    p_ds = sub.add_parser("dataset", help="评测数据集管理")
    ds_sub = p_ds.add_subparsers(dest="dataset_command", required=True)

    p_ds_list = ds_sub.add_parser("list", help="列出数据集中的评测样例")
    add_db_arg(p_ds_list)

    p_imp = ds_sub.add_parser("import", help="从 JSONL 导入评测样例（事务语义：全部合法才写入）")
    p_imp.add_argument("file", help="JSONL 文件路径")
    p_imp.add_argument("--dry-run", action="store_true", help="只校验不写入数据库")
    add_db_arg(p_imp)

    p_exp = ds_sub.add_parser("export", help="将数据库中的评测样例导出为 JSONL")
    p_exp.add_argument("output", help="输出 JSONL 文件路径")
    add_db_arg(p_exp)

    # ---- experiments ----
    p_ex = sub.add_parser("experiments", help="实验管理")
    ex_sub = p_ex.add_subparsers(dest="experiments_command", required=True)

    p_ec = ex_sub.add_parser("create", help="从 YAML 配置创建实验")
    p_ec.add_argument("--config", required=True, help="实验 YAML 配置文件路径")
    p_ec.add_argument(
        "--name",
        default=None,
        help="实验名（优先级：--name > YAML experiment_name > YAML name）",
    )
    add_db_arg(p_ec)

    p_el = ex_sub.add_parser("list", help="列出全部实验")
    add_db_arg(p_el)

    return parser


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def cmd_init_db(args: argparse.Namespace) -> int:
    storage = Storage(resolve_db_path(args.db))
    storage.init_db()
    tables = storage.user_table_names()
    print(f"数据库已初始化: {storage.db_path}")
    print(f"用户表 ({len(tables)} 张): {', '.join(tables)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    summary = run_experiment(
        db_path=resolve_db_path(args.db),
        config_path=args.config,
        experiment_name=args.name,
    )
    print("实验运行完成")
    print(f"experiment_id: {summary.experiment_id}")
    print(f"name: {summary.name}")
    print(f"cases: {summary.cases}")
    print(f"failure_total: {summary.failure_total}")
    metrics = summary.avg_metrics
    print(f"avg_hit_at_3: {metrics.get('hit_at_3', 0.0):.4f}")
    print(f"avg_expected_doc_coverage: {metrics.get('expected_doc_coverage', 0.0):.4f}")
    print(f"avg_must_include_coverage: {metrics.get('must_include_coverage', 0.0):.4f}")
    print(f"avg_answer_context_overlap_score: {metrics.get('answer_context_overlap_score', 0.0):.4f}")
    print(f"avg_latency_ms: {metrics.get('latency_ms', 0.0):.2f}")
    print(f"avg_estimated_cost: {metrics.get('estimated_cost', 0.0):.4f}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """生成单实验 Markdown 报告。"""
    storage = Storage(resolve_db_path(args.db))
    row = storage.get_experiment_by_id_or_name(args.experiment)
    if row is None:
        print(
            f"错误：未找到实验 '{args.experiment}'（请确认 experiment_id 或实验名）",
            file=sys.stderr,
        )
        return 1
    results = storage.list_case_results(row.experiment_id)
    if not results:
        print(f"错误：实验 '{row.name}' 没有 case_results，请先执行 run 生成结果", file=sys.stderr)
        return 1

    cases = {c.id: c for c in storage.list_cases()}
    md = generate_experiment_report(
        experiment_id=row.experiment_id,
        name=row.name,
        rag_version=row.rag_version,
        created_at=row.created_at,
        case_results=results,
        cases=cases,
    )
    out = Path(args.output) if args.output else Path("reports") / f"{row.name}_{row.experiment_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"实验报告已生成: {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """对比 baseline 与 candidate 并生成 Markdown 报告。"""
    storage = Storage(resolve_db_path(args.db))
    baseline = storage.get_experiment_by_id_or_name(args.baseline)
    if baseline is None:
        print(f"错误：未找到 baseline 实验 '{args.baseline}'（请确认 experiment_id 或实验名）", file=sys.stderr)
        return 1
    candidate = storage.get_experiment_by_id_or_name(args.candidate)
    if candidate is None:
        print(f"错误：未找到 candidate 实验 '{args.candidate}'（请确认 experiment_id 或实验名）", file=sys.stderr)
        return 1

    baseline_results = storage.list_case_results(baseline.experiment_id)
    candidate_results = storage.list_case_results(candidate.experiment_id)
    if not baseline_results:
        print(f"错误：baseline 实验 '{baseline.name}' 没有 case_results，请先执行 run", file=sys.stderr)
        return 1
    if not candidate_results:
        print(f"错误：candidate 实验 '{candidate.name}' 没有 case_results，请先执行 run", file=sys.stderr)
        return 1

    report = compare_experiments(baseline_results, candidate_results)  # 无共同 case 时抛出 ComparisonError
    md = generate_comparison_report(report)
    out = (
        Path(args.output)
        if args.output
        else Path("reports") / f"compare_{baseline.name}_{candidate.name}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"对比报告已生成: {out}")
    print(f"overall_status: {report.summary.overall_status}")
    print(f"regressed_cases: {report.summary.regressed_cases}")
    print(f"improved_cases: {report.summary.improved_cases}")
    return 0


def cmd_dataset_list(args: argparse.Namespace) -> int:
    storage = Storage(resolve_db_path(args.db))
    cases = storage.list_cases()
    if not cases:
        print("数据集为空，请先执行 dataset import。")
        return 0
    for i, case in enumerate(cases, start=1):
        question = case.question if len(case.question) <= 40 else case.question[:39] + "…"
        print(f"{i:>3}. [{case.id}] ({case.category}) {question}")
    print(f"共 {len(cases)} 条样例。")
    return 0


def cmd_dataset_import(args: argparse.Namespace) -> int:
    storage = Storage(resolve_db_path(args.db))
    stats = ds_lib.import_cases(storage, args.file, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run 校验通过：{stats.total} 条样例，未写入数据库。")
    else:
        print(
            f"导入完成：total={stats.total}，新增={stats.inserted}，更新={stats.updated}，"
            f"数据库={storage.db_path}"
        )
    return 0


def cmd_dataset_export(args: argparse.Namespace) -> int:
    storage = Storage(resolve_db_path(args.db))
    n = ds_lib.export_cases(storage, args.output)
    print(f"已导出 {n} 条样例到 {args.output}")
    return 0


def cmd_experiments_create(args: argparse.Namespace) -> int:
    storage = Storage(resolve_db_path(args.db))
    exp = ds_lib.create_experiment_from_config(storage, args.config, name=args.name)
    print(
        f"实验已创建：id={exp.experiment_id} name={exp.name} rag_version={exp.rag_version} "
        f"(配置已保存完整 YAML 原文)"
    )
    return 0


def cmd_experiments_list(args: argparse.Namespace) -> int:
    storage = Storage(resolve_db_path(args.db))
    rows = storage.list_experiments()
    if not rows:
        print("暂无实验，请先执行 experiments create。")
        return 0
    for row in rows:
        case_count = storage.count_case_results(row.experiment_id)
        print(
            f"[{row.experiment_id}] name={row.name} rag_version={row.rag_version} "
            f"created_at={row.created_at} case_results={case_count}"
        )
    print(f"共 {len(rows)} 个实验。")
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "init-db":
        return cmd_init_db(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "dataset":
        commands = {
            "list": cmd_dataset_list,
            "import": cmd_dataset_import,
            "export": cmd_dataset_export,
        }
        return commands[args.dataset_command](args)
    if args.command == "experiments":
        commands = {
            "create": cmd_experiments_create,
            "list": cmd_experiments_list,
        }
        return commands[args.experiments_command](args)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (
        ds_lib.DatasetValidationError,
        ds_lib.DatasetFileError,
        ds_lib.ConfigError,
        DuplicateExperimentError,
        RunnerError,
        ComparisonError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI 兜底
        print(f"未预期错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())