"""Хранилище дашбордов: сохранённые дашборды, история версий, черновики сессий.

Отдельная SQLite-база (storage/app.db). Данные таблиц хранятся сжатыми и
дедуплицируются по хешу: десять версий одного дашборда с одинаковыми данными
занимают место один раз.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import settings

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS dashboards (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    query TEXT,
    description TEXT,
    dataset TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    share_token TEXT UNIQUE,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS versions (
    dashboard_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    meta_json TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY (dashboard_id, version)
);
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY,
    data BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    messages_json TEXT,
    draft_meta TEXT,
    draft_hash TEXT,
    updated_at TEXT
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(path or settings.app_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Сериализация таблиц
# ---------------------------------------------------------------------------


def _frame_to_obj(frame: pd.DataFrame | None) -> dict | None:
    if frame is None:
        return None
    data = {}
    dtypes = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            dtypes[col] = "datetime"
            data[col] = series.dt.strftime("%Y-%m-%d").tolist()
        elif pd.api.types.is_bool_dtype(series):
            dtypes[col] = "bool"
            data[col] = series.astype(bool).tolist()
        elif pd.api.types.is_integer_dtype(series):
            dtypes[col] = "int"
            data[col] = series.astype(int).tolist()
        elif pd.api.types.is_float_dtype(series):
            dtypes[col] = "float"
            data[col] = [None if (v is None or np.isnan(v)) else float(v) for v in series.tolist()]
        else:
            dtypes[col] = "str"
            data[col] = series.astype(str).tolist()
    return {"columns": list(frame.columns), "dtypes": dtypes, "data": data}


def _obj_to_frame(obj: dict | None) -> pd.DataFrame | None:
    if obj is None:
        return None
    frame = pd.DataFrame({col: obj["data"][col] for col in obj["columns"]}, columns=obj["columns"])
    for col, dtype in obj["dtypes"].items():
        if dtype == "datetime":
            frame[col] = pd.to_datetime(frame[col])
        elif dtype == "float":
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
        elif dtype == "int":
            frame[col] = frame[col].astype("int64")
        elif dtype == "bool":
            frame[col] = frame[col].astype(bool)
        else:
            frame[col] = frame[col].astype(object)
    return frame


def pack_frames(table: pd.DataFrame, prev: pd.DataFrame | None, links: pd.DataFrame | None) -> tuple[str, bytes]:
    payload = json.dumps({"table": _frame_to_obj(table), "prev": _frame_to_obj(prev), "links": _frame_to_obj(links)},
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob = gzip.compress(payload, compresslevel=6, mtime=0)
    return hashlib.sha1(blob).hexdigest(), blob


def unpack_frames(blob: bytes) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    obj = json.loads(gzip.decompress(blob).decode("utf-8"))
    return _obj_to_frame(obj["table"]), _obj_to_frame(obj.get("prev")), _obj_to_frame(obj.get("links"))


_blob_cache: dict[str, tuple] = {}
_digest_by_identity: dict[tuple, str] = {}


def _put_blob(conn: sqlite3.Connection, frames: tuple) -> str:
    # Таблицы в состоянии не меняются на месте, поэтому одинаковые объекты = одинаковые данные:
    # не пересжимаем их при каждом автосохранении.
    key = tuple(id(f) for f in frames)
    cached = _digest_by_identity.get(key)
    if cached and cached in _blob_cache and all(a is b for a, b in zip(_blob_cache[cached], frames)):
        return cached
    digest, blob = pack_frames(*frames)
    conn.execute("INSERT OR IGNORE INTO blobs(hash, data) VALUES (?, ?)", (digest, blob))
    _blob_cache[digest] = frames
    _digest_by_identity[key] = digest
    if len(_digest_by_identity) > 200:
        _digest_by_identity.pop(next(iter(_digest_by_identity)))
    return digest


def _get_blob(conn: sqlite3.Connection, digest: str) -> tuple:
    if digest in _blob_cache:
        return _blob_cache[digest]
    row = conn.execute("SELECT data FROM blobs WHERE hash = ?", (digest,)).fetchone()
    if row is None:
        raise KeyError("Данные дашборда не найдены")
    frames = unpack_frames(row["data"])
    if len(_blob_cache) > 30:
        _blob_cache.pop(next(iter(_blob_cache)))
    _blob_cache[digest] = frames
    _digest_by_identity[tuple(id(f) for f in frames)] = digest
    return frames


# ---------------------------------------------------------------------------
# Дашборды
# ---------------------------------------------------------------------------


def _row_to_info(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "name": row["name"], "query": row["query"] or "", "description": row["description"] or "",
        "dataset": row["dataset"], "created_at": row["created_at"], "updated_at": row["updated_at"],
        "version": row["current_version"], "share_token": row["share_token"], "deleted": bool(row["deleted_at"]),
    }


def find_by_name(name: str, *, include_deleted: bool = False) -> dict | None:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM dashboards").fetchall()
    target = _norm(name)
    for row in rows:
        if (include_deleted or not row["deleted_at"]) and _norm(row["name"]) == target:
            return _row_to_info(row)
    return None


def get_info(dashboard_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM dashboards WHERE id = ?", (dashboard_id,)).fetchone()
    return _row_to_info(row) if row else None


def save_new(name: str, meta: dict, frames: tuple, *, description: str = "", note: str = "Создан") -> dict:
    dashboard_id = uuid.uuid4().hex[:10]
    now = _now()
    with connect() as conn:
        digest = _put_blob(conn, frames)
        conn.execute(
            "INSERT INTO dashboards(id, name, query, description, dataset, created_at, updated_at, current_version, share_token)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (dashboard_id, name, meta.get("query", ""), description, meta.get("dataset"), now, now, secrets.token_urlsafe(9)),
        )
        conn.execute("INSERT INTO versions VALUES (?, 1, ?, ?, ?, ?)",
                     (dashboard_id, json.dumps(meta, ensure_ascii=False), digest, now, note))
    return get_info(dashboard_id)  # type: ignore[return-value]


def save_version(dashboard_id: str, meta: dict, frames: tuple, *, name: str | None = None,
                 description: str | None = None, note: str = "Обновлён") -> dict:
    now = _now()
    with connect() as conn:
        row = conn.execute("SELECT current_version FROM dashboards WHERE id = ?", (dashboard_id,)).fetchone()
        if row is None:
            raise KeyError("Дашборд не найден")
        version = row["current_version"] + 1
        digest = _put_blob(conn, frames)
        conn.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?)",
                     (dashboard_id, version, json.dumps(meta, ensure_ascii=False), digest, now, note))
        conn.execute("UPDATE dashboards SET current_version = ?, updated_at = ?, name = COALESCE(?, name),"
                     " description = COALESCE(?, description), query = ? WHERE id = ?",
                     (version, now, name, description, meta.get("query", ""), dashboard_id))
    return get_info(dashboard_id)  # type: ignore[return-value]


def load(dashboard_id: str, version: int | None = None) -> tuple[dict, dict, tuple]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM dashboards WHERE id = ?", (dashboard_id,)).fetchone()
        if row is None:
            raise KeyError("Дашборд не найден")
        ver = version or row["current_version"]
        vrow = conn.execute("SELECT * FROM versions WHERE dashboard_id = ? AND version = ?", (dashboard_id, ver)).fetchone()
        if vrow is None:
            raise KeyError(f"Версия {ver} не найдена")
        frames = _get_blob(conn, vrow["data_hash"])
    info = _row_to_info(row)
    info["loaded_version"] = ver
    info["version_created_at"] = vrow["created_at"]
    return info, json.loads(vrow["meta_json"]), frames


def versions(dashboard_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT version, created_at, note FROM versions WHERE dashboard_id = ? ORDER BY version",
                            (dashboard_id,)).fetchall()
    return [dict(r) for r in rows]


def rename(dashboard_id: str, new_name: str) -> dict:
    with connect() as conn:
        conn.execute("UPDATE dashboards SET name = ?, updated_at = ? WHERE id = ?", (new_name, _now(), dashboard_id))
    return get_info(dashboard_id)  # type: ignore[return-value]


def delete(dashboard_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE dashboards SET deleted_at = ? WHERE id = ?", (_now(), dashboard_id))


def deleted_dashboards() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM dashboards WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC").fetchall()
    return [_row_to_info(r) for r in rows]


def restore(dashboard_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE dashboards SET deleted_at = NULL WHERE id = ?", (dashboard_id,))


def copy_dashboard(dashboard_id: str, new_name: str) -> dict:
    info, meta, frames = load(dashboard_id)
    meta = dict(meta)
    meta["name"] = new_name
    return save_new(new_name, meta, frames, description=info.get("description", ""), note=f"Копия «{info['name']}»")


def by_share_token(token: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM dashboards WHERE share_token = ? AND deleted_at IS NULL", (token,)).fetchone()
    return _row_to_info(row) if row else None


_STEM_ENDINGS = re.compile(r"(ами|ями|ого|его|ому|ему|ыми|ими|ах|ях|ов|ев|ей|ой|ый|ий|ая|яя|ое|ее|ые|ие|ам|ям|ом|ем|ы|и|а|я|о|е|у|ю|ь)$")


def _norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[«»\"'“”„]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(word: str) -> str:
    word = _norm(word)
    if len(word) <= 4:
        return word
    return _STEM_ENDINGS.sub("", word) or word


DATASET_WORDS = {"sales": "продажи план факт", "discounts": "скидки скидка причины", "clients": "клиенты продажи клиентов"}


def list_dashboards(query: str = "", sort: str = "updated") -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM dashboards WHERE deleted_at IS NULL").fetchall()
    items = [_row_to_info(r) for r in rows]
    if query:
        stems = [_stem(w) for w in re.findall(r"[\wё-]+", query.lower()) if len(w) > 2]
        stems = [s for s in stems if s not in {"дашборд", "про", "для", "по", "мои", "мой", "все", "найд", "найти"}]
        scored = []
        for item in items:
            hay = _norm(" ".join([item["name"], item["query"], item["description"], DATASET_WORDS.get(item["dataset"] or "", "")]))
            score = sum(1 for s in stems if s and s in hay)
            if score:
                scored.append((score, item))
        items = [it for _, it in sorted(scored, key=lambda p: -p[0])] if stems else items
    if sort == "name":
        items.sort(key=lambda it: it["name"].lower())
    elif sort == "created":
        items.sort(key=lambda it: it["created_at"], reverse=True)
    elif not query:
        items.sort(key=lambda it: it["updated_at"], reverse=True)
    return items


def unique_name(base: str) -> str:
    existing = {_norm(i["name"]) for i in list_dashboards()}
    if _norm(base) not in existing:
        return base
    n = 2
    while _norm(f"{base} ({n})") in existing:
        n += 1
    return f"{base} ({n})"


# ---------------------------------------------------------------------------
# Сессии (автосохранение черновика и переписки)
# ---------------------------------------------------------------------------


def save_session(session_id: str, messages: list[dict], draft_meta: dict | None, frames: tuple | None) -> None:
    with connect() as conn:
        digest = _put_blob(conn, frames) if frames is not None else None
        conn.execute(
            "INSERT INTO sessions(id, messages_json, draft_meta, draft_hash, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET messages_json = excluded.messages_json, draft_meta = excluded.draft_meta,"
            " draft_hash = excluded.draft_hash, updated_at = excluded.updated_at",
            (session_id, json.dumps(messages, ensure_ascii=False), json.dumps(draft_meta, ensure_ascii=False) if draft_meta else None,
             digest, _now()),
        )


def load_session(session_id: str) -> tuple[list[dict], dict | None, tuple | None] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        frames = _get_blob(conn, row["draft_hash"]) if row["draft_hash"] else None
    messages = json.loads(row["messages_json"] or "[]")
    meta = json.loads(row["draft_meta"]) if row["draft_meta"] else None
    return messages, meta, frames

