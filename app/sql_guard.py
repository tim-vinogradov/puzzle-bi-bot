"""Проверка SQL перед выполнением.

Бот ходит в базу только на чтение, но мы не полагаемся на одни лишь права:
любой SQL (и собранный конструктором, и написанный языковой моделью) проходит
разбор синтаксическим анализатором sqlglot и отклоняется, если:

* в тексте больше одного оператора;
* это не SELECT (INSERT/UPDATE/DELETE/DROP/CREATE/COPY/SET и т. п.);
* используются таблицы вне белого списка (в том числе системные каталоги);
* вызываются опасные функции (pg_sleep, dblink, чтение файлов и т. п.).

К запросу без LIMIT добавляется ограничение на число строк.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {
    "regions", "products", "clients", "contracts", "outlets",
    "sales", "sales_plan", "discounts",
}

FORBIDDEN_FUNCTION_PREFIXES = (
    "pg_", "dblink", "lo_", "load_extension", "readfile", "writefile",
    "set_config", "current_setting", "query_to_xml", "txid_", "copy",
    "file", "url", "remote", "s3", "system", "sleep",
)

# Имена узлов берём через getattr: в разных версиях sqlglot набор классов немного отличается.
FORBIDDEN_NODES = tuple(
    getattr(exp, name) for name in (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command", "Merge", "Into",
        "Set", "Transaction", "Commit", "Rollback", "Grant", "Pragma", "Use", "LoadData", "Copy", "Lock",
        "TruncateTable", "Attach", "Detach",
    ) if hasattr(exp, name)
)


class UnsafeSQLError(ValueError):
    """SQL не прошёл проверку безопасности."""


def _dialect(dialect: str) -> str:
    return {"postgresql": "postgres", "sqlite": "sqlite", "clickhouse": "clickhouse"}.get(dialect, dialect)


def validate_select(sql: str, dialect: str = "postgresql", max_rows: int | None = None) -> str:
    """Проверяет, что SQL — один безопасный SELECT. Возвращает (возможно, дополненный LIMIT) SQL."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQLError("Пустой запрос")
    if len(text) > 20000:
        raise UnsafeSQLError("Слишком длинный запрос")
    try:
        statements = [s for s in sqlglot.parse(text, read=_dialect(dialect)) if s is not None]
    except sqlglot.errors.ParseError as err:  # type: ignore[attr-defined]
        raise UnsafeSQLError(f"Не удалось разобрать SQL: {err}") from err
    if len(statements) != 1:
        raise UnsafeSQLError("Разрешён ровно один оператор SELECT")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        raise UnsafeSQLError("Разрешены только запросы на чтение (SELECT)")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQLError(f"Запрещённая операция: {type(node).__name__}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        schema = (table.db or "").lower()
        if schema and schema not in {"public", "main"}:
            raise UnsafeSQLError(f"Доступ к схеме «{schema}» запрещён")
        if name in cte_names:
            continue
        if name not in ALLOWED_TABLES:
            raise UnsafeSQLError(f"Таблица «{name}» не входит в список разрешённых")

    for func in tree.find_all(exp.Func):
        fname = ""
        if isinstance(func, exp.Anonymous):
            fname = str(func.this or "").lower()
        else:
            fname = (func.sql_name() or "").lower()
        if any(fname.startswith(p) for p in FORBIDDEN_FUNCTION_PREFIXES):
            raise UnsafeSQLError(f"Функция «{fname}» запрещена")

    if max_rows and not (isinstance(tree, exp.Select) and tree.args.get("limit") is not None):
        # исходный текст не перегенерируем (чтобы не испортить параметры :start/:end)
        return f"SELECT * FROM ({text}) AS guarded_q LIMIT {int(max_rows) + 1}"
    return text
