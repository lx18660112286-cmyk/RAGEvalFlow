"""pytest 共享 fixture：每个测试使用独立的临时 SQLite 数据库。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragevalflow.core.storage import Storage


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
def storage(db_path: Path) -> Storage:
    store = Storage(db_path)
    store.init_db()
    return store