"""数据集与实验配置的读写。

- load_cases_from_jsonl / import_cases / export_cases：JSONL 导入导出（事务语义）；
- load_config_yaml / resolve_experiment_name / create_experiment_from_config：实验配置读取与创建。

阶段 1 不包含任何指标计算、runner、MockRAGClient 逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ragevalflow.core.storage import Storage
from ragevalflow.schemas.eval_case import EvalCase
from ragevalflow.schemas.experiment_result import Experiment


class DatasetFileError(Exception):
    """数据文件读取失败。"""


class DatasetValidationError(Exception):
    """数据校验失败（含全部行级错误明细），导入整体失败、数据库零修改。"""


class ConfigError(Exception):
    """实验配置文件错误。"""


@dataclass
class ImportStats:
    total: int
    inserted: int
    updated: int


# ---------------------------------------------------------------------------
# JSONL 导入导出
# ---------------------------------------------------------------------------

def load_cases_from_jsonl(path: str | Path) -> list[EvalCase]:
    """读取 JSONL 并做逐条 Pydantic 校验。

    任一非法 → 抛出 DatasetValidationError（携带全部错误明细），不返回部分数据。
    """
    path = Path(path)
    if not path.exists():
        raise DatasetFileError(f"数据集文件不存在: {path}")

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cases: list[EvalCase] = []
    errors: list[str] = []

    for idx, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"第 {idx} 行: 不是合法 JSON（{exc.msg}）")
            continue
        try:
            cases.append(EvalCase.model_validate(data))
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            errors.append(f"第 {idx} 行: 字段校验失败 [{detail}]")

    if errors:
        raise DatasetValidationError(
            "数据集校验未通过，导入已中止（数据库未被修改）：\n" + "\n".join(errors)
        )
    return cases


def serialize_cases_to_jsonl(cases: list[EvalCase]) -> str:
    """将样例序列化为 JSONL 文本（语义与源文件保持一致）。"""
    if not cases:
        return ""
    return "\n".join(json.dumps(c.model_dump(), ensure_ascii=False) for c in cases) + "\n"


def import_cases(storage: Storage, path: str | Path, dry_run: bool = False) -> ImportStats:
    """事务化导入：先全量读取并校验，全部合法才写入数据库。

    dry_run=True 时只校验不写入。
    """
    cases = load_cases_from_jsonl(path)
    if dry_run:
        return ImportStats(total=len(cases), inserted=0, updated=0)
    stats = storage.import_cases(cases)
    return ImportStats(**stats)


def export_cases(storage: Storage, path: str | Path) -> int:
    """将数据库中全部样例导出为 JSONL，返回导出条数。"""
    cases = storage.list_cases()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialize_cases_to_jsonl(cases), encoding="utf-8")
    return len(cases)


# ---------------------------------------------------------------------------
# 实验配置（YAML）
# ---------------------------------------------------------------------------

def load_config_yaml(path: str | Path) -> tuple[dict[str, Any], str]:
    """读取 YAML 配置，返回 (解析后的字典, 原始文本)。"""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except Exception as exc:  # noqa: BLE001 - 统一转成用户可读错误
        raise ConfigError(f"解析 YAML 失败: {path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射（dict），当前为 {type(data).__name__}: {path}")
    return data, raw


def resolve_experiment_name(data: dict[str, Any], cli_name: str | None) -> str:
    """实验名优先级：--name > YAML 顶层 experiment_name > YAML 顶层 name。"""
    if cli_name:
        return cli_name
    for key in ("experiment_name", "name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ConfigError("无法确定实验名：请通过 --name 参数，或在 YAML 顶层提供 experiment_name / name 字段")


def create_experiment_from_config(storage: Storage, config_path: str | Path, name: str | None = None) -> Experiment:
    """从 YAML 配置创建实验：config 列保存完整 YAML 原文，rag_version 取 YAML 顶层字段。"""
    data, raw = load_config_yaml(config_path)
    exp_name = resolve_experiment_name(data, name)
    rag_version = str(data.get("rag_version") or "")
    row = storage.create_experiment(name=exp_name, rag_version=rag_version, config_yaml=raw)
    return Experiment(
        experiment_id=row.experiment_id,
        name=row.name,
        rag_version=row.rag_version,
        config=data,
        created_at=datetime.fromisoformat(row.created_at),
    )