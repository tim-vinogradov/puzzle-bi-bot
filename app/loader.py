"""Загрузка тестовых данных кейса (CSV) в базу.

Использование:
    python -m app.loader                     # собрать встроенную SQLite (data/sales.db)
    python -m app.loader --postgres URL      # загрузить в PostgreSQL и создать роль только для чтения

URL для PostgreSQL — под администратором, например
    postgresql://postgres:postgres@localhost:5432/puzzle
После загрузки бот подключается под ролью `bi_reader` (только SELECT).
"""
from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import sys
import threading
from pathlib import Path

from .config import settings

TABLES: dict[str, list[tuple[str, str, str]]] = {
    # имя колонки, тип PostgreSQL, тип SQLite
    "regions": [
        ("region_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("region_name", "TEXT NOT NULL", "TEXT NOT NULL"),
        ("regional_center", "TEXT", "TEXT"),
    ],
    "products": [
        ("product_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("product_name", "TEXT NOT NULL", "TEXT NOT NULL"),
        ("category", "TEXT", "TEXT"),
        ("unit_price", "NUMERIC(12,2)", "REAL"),
        ("unit", "TEXT", "TEXT"),
        ("launch_date", "DATE", "TEXT"),
        ("discontinued_date", "DATE", "TEXT"),
    ],
    "clients": [
        ("client_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("client_name", "TEXT NOT NULL", "TEXT NOT NULL"),
        ("inn", "TEXT", "TEXT"),
        ("segment", "TEXT", "TEXT"),
        ("region_id", "INTEGER REFERENCES regions(region_id)", "INTEGER"),
        ("registered_date", "DATE", "TEXT"),
        ("is_key_client", "BOOLEAN", "INTEGER"),
    ],
    "contracts": [
        ("contract_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("contract_number", "TEXT", "TEXT"),
        ("client_id", "INTEGER REFERENCES clients(client_id)", "INTEGER"),
        ("start_date", "DATE", "TEXT"),
        ("end_date", "DATE", "TEXT"),
        ("status", "TEXT", "TEXT"),
        ("payment_terms_days", "INTEGER", "INTEGER"),
        ("base_discount_pct", "NUMERIC(6,2)", "REAL"),
    ],
    "outlets": [
        ("outlet_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("outlet_name", "TEXT NOT NULL", "TEXT NOT NULL"),
        ("region_id", "INTEGER REFERENCES regions(region_id)", "INTEGER"),
        ("city", "TEXT", "TEXT"),
        ("outlet_type", "TEXT", "TEXT"),
        ("address", "TEXT", "TEXT"),
        ("opened_date", "DATE", "TEXT"),
        ("closed_date", "DATE", "TEXT"),
    ],
    "sales": [
        ("sale_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("sale_date", "DATE NOT NULL", "TEXT NOT NULL"),
        ("contract_id", "INTEGER REFERENCES contracts(contract_id)", "INTEGER"),
        ("client_id", "INTEGER REFERENCES clients(client_id)", "INTEGER"),
        ("outlet_id", "INTEGER REFERENCES outlets(outlet_id)", "INTEGER"),
        ("product_id", "INTEGER REFERENCES products(product_id)", "INTEGER"),
        ("quantity", "INTEGER", "INTEGER"),
        ("amount", "NUMERIC(14,2)", "REAL"),
    ],
    "sales_plan": [
        ("plan_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("plan_month", "DATE NOT NULL", "TEXT NOT NULL"),
        ("outlet_id", "INTEGER REFERENCES outlets(outlet_id)", "INTEGER"),
        ("product_id", "INTEGER REFERENCES products(product_id)", "INTEGER"),
        ("plan_amount", "NUMERIC(14,2)", "REAL"),
        ("plan_qty", "INTEGER", "INTEGER"),
    ],
    "discounts": [
        ("discount_id", "INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY"),
        ("sale_id", "INTEGER REFERENCES sales(sale_id)", "INTEGER"),
        ("client_id", "INTEGER REFERENCES clients(client_id)", "INTEGER"),
        ("discount_date", "DATE", "TEXT"),
        ("discount_amount", "NUMERIC(14,2)", "REAL"),
        ("discount_pct", "NUMERIC(6,2)", "REAL"),
        ("reason", "TEXT", "TEXT"),
    ],
}

INDEXES = [
    ("sales", "sale_date"), ("sales", "outlet_id, product_id"), ("sales", "client_id"),
    ("sales_plan", "plan_month"), ("sales_plan", "outlet_id, product_id"),
    ("discounts", "sale_id"), ("discounts", "client_id"), ("discounts", "discount_date"),
    ("outlets", "region_id"), ("clients", "region_id"),
]

_lock = threading.Lock()


def _csv_path(table: str) -> Path:
    path = settings.data_dir / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Положите CSV из архива кейса в папку data/ (см. README)."
        )
    return path


def _convert(value: str, sqlite_type: str):
    if value in ("\\N", ""):
        return None
    if sqlite_type.startswith("INTEGER"):
        if value in ("true", "false"):
            return 1 if value == "true" else 0
        return int(value)
    if sqlite_type.startswith("REAL"):
        return float(value)
    return value


def build_sqlite(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    conn = sqlite3.connect(tmp_path)
    try:
        for table, columns in TABLES.items():
            ddl = ", ".join(f"{name} {sqlite_type}" for name, _, sqlite_type in columns)
            conn.execute(f"CREATE TABLE {table} ({ddl})")
            with open(_csv_path(table), encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                expected = [c[0] for c in columns]
                if header != expected:
                    raise ValueError(f"{table}.csv: ожидались колонки {expected}, получены {header}")
                types = [c[2] for c in columns]
                rows = ([_convert(v, t) for v, t in zip(row, types)] for row in reader if row)
                placeholders = ", ".join("?" for _ in columns)
                conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        for table, cols in INDEXES:
            name = f"ix_{table}_{cols.replace(', ', '_')}"
            conn.execute(f"CREATE INDEX {name} ON {table} ({cols})")
        conn.commit()
    finally:
        conn.close()
    tmp_path.replace(db_path)


def ensure_sqlite(db_path: str | Path) -> None:
    """Создаёт SQLite из CSV при первом запуске (или если CSV обновились)."""
    db_path = Path(db_path)
    with _lock:
        csv_files = [settings.data_dir / f"{t}.csv" for t in TABLES]
        newest_csv = max((p.stat().st_mtime for p in csv_files if p.exists()), default=0)
        if db_path.exists() and db_path.stat().st_mtime >= newest_csv:
            return
        print(f"Собираю базу {db_path} из CSV — это займёт несколько секунд…", file=sys.stderr)
        build_sqlite(db_path)


def load_postgres(admin_url: str, reader_password: str = "bi_reader_pwd") -> None:
    import psycopg2  # импорт здесь, чтобы SQLite-режим не требовал драйвера

    url = admin_url.replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()
    for table in reversed(list(TABLES)):
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    for table, columns in TABLES.items():
        ddl = ", ".join(f"{name} {pg_type}" for name, pg_type, _ in columns)
        cur.execute(f"CREATE TABLE {table} ({ddl})")
        with open(_csv_path(table), encoding="utf-8") as fh:
            cur.copy_expert(
                f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '\\N', ENCODING 'UTF8')",
                io.StringIO(fh.read()),
            )
        print(f"  {table}: загружено", file=sys.stderr)
    for table, cols in INDEXES:
        name = f"ix_{table}_{cols.replace(', ', '_')}"
        cur.execute(f"CREATE INDEX {name} ON {table} ({cols})")
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'bi_reader'")
    if cur.fetchone() is None:
        cur.execute("CREATE ROLE bi_reader LOGIN PASSWORD %s", (reader_password,))
    cur.execute("SELECT current_database()")
    dbname = cur.fetchone()[0]
    cur.execute(f'GRANT CONNECT ON DATABASE "{dbname}" TO bi_reader')
    cur.execute("GRANT USAGE ON SCHEMA public TO bi_reader")
    cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO bi_reader")
    cur.execute("ALTER ROLE bi_reader SET default_transaction_read_only = on")
    conn.commit()
    cur.close()
    conn.close()
    print("Готово. Строка подключения для бота (.env):", file=sys.stderr)
    host_part = url.split("@", 1)[1] if "@" in url else "localhost:5432/" + dbname
    print(f"DATABASE_URL=postgresql+psycopg2://bi_reader:{reader_password}@{host_part}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка данных кейса в базу")
    parser.add_argument("--postgres", help="URL PostgreSQL под администратором")
    parser.add_argument("--reader-password", default="bi_reader_pwd")
    args = parser.parse_args()
    if args.postgres:
        load_postgres(args.postgres, args.reader_password)
    else:
        target = settings.data_dir / "sales.db"
        build_sqlite(target)
        print(f"SQLite собрана: {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
