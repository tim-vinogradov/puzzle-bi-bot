"""Подключение к базе с данными (только чтение) и безопасное выполнение запросов."""
from __future__ import annotations

import decimal
import logging
import sqlite3
import threading
import time
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from .config import settings
from .sql_guard import validate_select

log = logging.getLogger(__name__)


class QueryError(RuntimeError):
    """Ошибка выполнения запроса к базе (таймаут, слишком много строк и т. п.)."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = settings.effective_database_url
    if url.startswith("sqlite"):
        from .loader import ensure_sqlite  # локальный импорт, чтобы избежать цикла

        from pathlib import Path

        db_path = Path(url.split("sqlite:///", 1)[1]).resolve()
        ensure_sqlite(db_path)
        # Открываем файл строго в режиме только чтения (URI корректен и для путей Windows).
        ro_uri = db_path.as_uri() + "?mode=ro"
        engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(ro_uri, uri=True, check_same_thread=False))
    else:
        timeout_ms = settings.query_timeout_sec * 1000
        engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={timeout_ms} -c default_transaction_read_only=on"},
        )
    return engine


def dialect_name() -> str:
    return get_engine().dialect.name  # 'postgresql' | 'sqlite'


def _to_python(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value


def _install_sqlite_timeout(dbapi_conn: sqlite3.Connection, deadline: float) -> None:
    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    dbapi_conn.set_progress_handler(handler, 20000)


def run_select(sql: str, params: dict | None = None, *, max_rows: int | None = None,
               trusted: bool = False) -> pd.DataFrame:
    """Выполняет SELECT и возвращает DataFrame.

    Каждый запрос проходит проверку `validate_select` (кроме собранных нашим же
    конструктором — `trusted=True`, но и они проверяются в тестах).
    Работает таймаут и ограничение на число строк.
    """
    engine = get_engine()
    limit = max_rows or settings.max_rows
    dialect = engine.dialect.name
    checked_sql = sql if trusted else validate_select(sql, dialect, max_rows=limit)
    deadline = time.monotonic() + settings.query_timeout_sec
    started = time.monotonic()
    try:
        with engine.connect() as conn:
            raw = conn.connection.dbapi_connection
            if dialect == "sqlite":
                _install_sqlite_timeout(raw, deadline)
            result = conn.execute(text(checked_sql), params or {})
            columns = list(result.keys())
            rows = result.fetchmany(limit + 1)
            if dialect == "sqlite":
                raw.set_progress_handler(None, 0)
    except Exception as err:  # noqa: BLE001 — переводим в понятную пользователю ошибку
        message = str(err).lower()
        if "interrupted" in message or "statement timeout" in message or "canceling statement" in message:
            raise QueryError(f"Запрос выполнялся дольше {settings.query_timeout_sec} с и был остановлен") from err
        if "read-only" in message or "readonly" in message:
            raise QueryError("База доступна только для чтения — изменение данных запрещено") from err
        raise QueryError(f"Ошибка базы данных: {err}") from err
    if len(rows) > limit:
        raise QueryError(f"Запрос вернул больше {limit:,} строк — уточните период или разрез".replace(",", " "))
    frame = pd.DataFrame([tuple(_to_python(v) for v in row) for row in rows], columns=columns)
    log.info("SQL выполнен за %.2f с, строк: %d", time.monotonic() - started, len(frame))
    return frame


_anchor_lock = threading.Lock()
_anchor_cache: dict[str, pd.Timestamp] = {}


def data_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Первая и последняя дата продаж в базе (кэшируется)."""
    with _anchor_lock:
        if "min" not in _anchor_cache:
            frame = run_select("SELECT MIN(sale_date) AS dmin, MAX(sale_date) AS dmax FROM sales", trusted=True)
            _anchor_cache["min"] = pd.Timestamp(frame.iloc[0]["dmin"])
            _anchor_cache["max"] = pd.Timestamp(frame.iloc[0]["dmax"])
        return _anchor_cache["min"], _anchor_cache["max"]


def reset_caches() -> None:
    get_engine.cache_clear()
    _anchor_cache.clear()


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record):  # pragma: no cover - служебное
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA query_only = ON")
        cursor.close()
