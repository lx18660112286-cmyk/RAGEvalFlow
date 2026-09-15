"""SQLite 存储层（SQLAlchemy 2.0 + SQLite）。

阶段 1 仅包含 3 张用户表：eval_cases / experiments / case_results。
- JSON 字段一律以 TEXT 存储，由 Pydantic 负责编解码；
- experiments.config 保存实验 YAML 配置的完整原文；
- dataset 导入使用事务语义：全部合法才写入，任一非法则整体失败（由调用方保证先校验）。
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import Text, create_engine, func, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ragevalflow.schemas.eval_case import EvalCase, ExpectedBehavior
from ragevalflow.schemas.experiment_result import CaseResult
from ragevalflow.schemas.failure import FailureFinding
from ragevalflow.schemas.rag_output import RAGOutput


class DuplicateExperimentError(Exception):
    """同名实验已存在。"""


class Base(DeclarativeBase):
    pass


class EvalCaseRow(Base):
    __tablename__ = "eval_cases"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str] = mapped_column(Text, default="general")
    question: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_docs: Mapped[str] = mapped_column(Text, default="[]")
    must_include: Mapped[str] = mapped_column(Text, default="[]")
    must_not_include: Mapped[str] = mapped_column(Text, default="[]")
    expected_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text)


class ExperimentRow(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)  # 实验名唯一约束
    rag_version: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[str] = mapped_column(Text)  # 实验 YAML 配置完整原文
    created_at: Mapped[str] = mapped_column(Text)


class CaseResultRow(Base):
    __tablename__ = "case_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(Text)
    case_id: Mapped[str] = mapped_column(Text)
    rag_output: Mapped[str] = mapped_column(Text)
    metrics: Mapped[str] = mapped_column(Text, default="{}")
    failure_types: Mapped[str] = mapped_column(Text, default="[]")
    failure_findings: Mapped[str] = mapped_column(Text, default="[]")


# ---------------------------------------------------------------------------
# JSON / 时间辅助函数
# ---------------------------------------------------------------------------

def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ORM 行 <-> Pydantic 模型互转
# ---------------------------------------------------------------------------

def case_to_row(case: EvalCase, created_at: str | None = None) -> EvalCaseRow:
    return EvalCaseRow(
        id=case.id,
        category=case.category,
        question=case.question,
        reference_answer=case.reference_answer,
        expected_docs=_dumps(case.expected_docs),
        must_include=_dumps(case.must_include),
        must_not_include=_dumps(case.must_not_include),
        expected_behavior=_dumps(case.expected_behavior.model_dump()) if case.expected_behavior else None,
        created_at=created_at or _now_iso(),
    )


def row_to_case(row: EvalCaseRow) -> EvalCase:
    behavior = _loads(row.expected_behavior)
    return EvalCase(
        id=row.id,
        category=row.category,
        question=row.question,
        reference_answer=row.reference_answer,
        expected_docs=_loads(row.expected_docs, []),
        must_include=_loads(row.must_include, []),
        must_not_include=_loads(row.must_not_include, []),
        expected_behavior=ExpectedBehavior.model_validate(behavior) if behavior else None,
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class Storage:
    """SQLite 存储门面：建表、评测样例读写（事务化 upsert）、实验管理。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 使用 as_posix() 保证 Windows 路径下 sqlite:/// URL 正确
        self._engine = create_engine(f"sqlite:///{self.db_path.as_posix()}")
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._maybe_migrate_case_results()

    def _maybe_migrate_case_results(self) -> None:
        """阶段 2 及更早建库的 case_results 无 failure_findings 列，这里做幂等迁移（不破坏旧行为）。"""
        try:
            columns = [c["name"] for c in inspect(self._engine).get_columns("case_results")]
        except Exception:  # noqa: BLE001 - 表尚未创建时跳过
            return
        if "failure_findings" not in columns:
            with self._engine.begin() as conn:
                conn.execute(text("ALTER TABLE case_results ADD COLUMN failure_findings TEXT NOT NULL DEFAULT '[]'"))

    def init_db(self) -> None:
        """创建全部表（已存在则跳过，幂等）。"""
        Base.metadata.create_all(self._engine)

    def user_table_names(self) -> list[str]:
        """返回数据库中用户创建的表名（排除 sqlite_% 内部表）。"""
        return [t for t in inspect(self._engine).get_table_names() if not t.startswith("sqlite_")]

    # ---- 评测样例 ----

    def import_cases(self, cases: Sequence[EvalCase]) -> dict[str, int]:
        """在单个事务内 upsert 全部样例。

        注意：调用方必须先完成全量 Pydantic 校验，本方法假设入参全部合法。
        """
        inserted = updated = 0
        with self._session_factory.begin() as session:
            for case in cases:
                row = session.get(EvalCaseRow, case.id)
                if row is None:
                    session.add(case_to_row(case))
                    inserted += 1
                else:
                    row.category = case.category
                    row.question = case.question
                    row.reference_answer = case.reference_answer
                    row.expected_docs = _dumps(case.expected_docs)
                    row.must_include = _dumps(case.must_include)
                    row.must_not_include = _dumps(case.must_not_include)
                    row.expected_behavior = (
                        _dumps(case.expected_behavior.model_dump()) if case.expected_behavior else None
                    )
                    updated += 1
        return {"total": inserted + updated, "inserted": inserted, "updated": updated}

    def list_cases(self) -> list[EvalCase]:
        with self._session_factory() as session:
            rows = session.execute(select(EvalCaseRow).order_by(EvalCaseRow.id)).scalars().all()
            return [row_to_case(r) for r in rows]

    def count_cases(self) -> int:
        with self._session_factory() as session:
            return session.scalar(select(func.count()).select_from(EvalCaseRow))

    # ---- 实验 ----

    def create_experiment(self, name: str, rag_version: str, config_yaml: str) -> ExperimentRow:
        """创建实验。

        实验 ID 格式 exp_YYYYMMDD_HHMMSS_<6 位随机后缀>，避免同一秒内创建冲突；
        同名实验重复创建抛出 DuplicateExperimentError（事务回滚，库不被污染）。
        """
        ts = datetime.now()
        experiment_id = f"exp_{ts:%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
        with self._session_factory.begin() as session:
            exists = session.execute(
                select(ExperimentRow).where(ExperimentRow.name == name)
            ).scalar_one_or_none()
            if exists is not None:
                raise DuplicateExperimentError(
                    f"实验名 '{name}' 已存在（experiment_id={exists.experiment_id}），请使用其他名称"
                )
            row = ExperimentRow(
                experiment_id=experiment_id,
                name=name,
                rag_version=rag_version,
                config=config_yaml,
                # 微秒精度，保证同秒内多次创建也能按时间先后排序
                created_at=ts.isoformat(timespec="microseconds"),
            )
            session.add(row)
            session.flush()
            return row

    def list_experiments(self) -> list[ExperimentRow]:
        with self._session_factory() as session:
            rows = session.execute(
                select(ExperimentRow).order_by(
                    ExperimentRow.created_at.desc(), ExperimentRow.experiment_id.desc()
                )
            ).scalars().all()
            return list(rows)

    def count_case_results(self, experiment_id: str) -> int:
        with self._session_factory() as session:
            return session.scalar(
                select(func.count()).select_from(CaseResultRow).where(CaseResultRow.experiment_id == experiment_id)
            )

    def save_case_results(self, results: Sequence[CaseResult]) -> int:
        """事务化写入一批 case_results（任一失败则整体回滚）。"""
        with self._session_factory.begin() as session:
            for result in results:
                session.add(CaseResultRow(
                    experiment_id=result.experiment_id,
                    case_id=result.case_id,
                    rag_output=_dumps(result.rag_output.model_dump()),
                    metrics=_dumps(result.metrics),
                    failure_types=_dumps(result.failure_types),
                    failure_findings=_dumps([f.model_dump() for f in result.failure_findings]),
                ))
        return len(results)

    def list_case_results(self, experiment_id: str) -> list[CaseResult]:
        with self._session_factory() as session:
            rows = session.execute(
                select(CaseResultRow)
                .where(CaseResultRow.experiment_id == experiment_id)
                .order_by(CaseResultRow.case_id)
            ).scalars().all()
            return [_row_to_case_result(r) for r in rows]

    # ---- 实验查询辅助（阶段 3）----

    def get_experiment_by_id_or_name(self, identifier: str) -> ExperimentRow | None:
        """按 experiment_id 或 name 精确匹配实验（id 优先）。"""
        with self._session_factory() as session:
            row = session.get(ExperimentRow, identifier)
            if row is not None:
                return row
            return session.execute(
                select(ExperimentRow).where(ExperimentRow.name == identifier)
            ).scalar_one_or_none()

    def get_case_result_with_case(self, experiment_id: str, case_id: str) -> tuple[CaseResult, EvalCase] | None:
        """返回指定 (experiment, case) 的结果与其关联的评测样例；不存在返回 None。"""
        with self._session_factory() as session:
            row = session.execute(
                select(CaseResultRow).where(
                    CaseResultRow.experiment_id == experiment_id,
                    CaseResultRow.case_id == case_id,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            case_row = session.get(EvalCaseRow, case_id)
            if case_row is None:
                return None
            return _row_to_case_result(row), row_to_case(case_row)


def _row_to_case_result(row: CaseResultRow) -> CaseResult:
    findings_raw = _loads(row.failure_findings, []) if hasattr(row, "failure_findings") else []
    findings = [FailureFinding.model_validate(f) for f in findings_raw] if findings_raw else []
    return CaseResult(
        experiment_id=row.experiment_id,
        case_id=row.case_id,
        rag_output=RAGOutput.model_validate(_loads(row.rag_output)),
        metrics=_loads(row.metrics, {}),
        failure_types=_loads(row.failure_types, []),
        failure_findings=findings,
    )