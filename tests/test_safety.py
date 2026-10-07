"""Безопасность: SQL только на чтение, формулы без исполнения кода, таймауты."""
import pytest

from app.calc import CalcError, eval_expr, parse_expr
from app.sql_guard import UnsafeSQLError, validate_select


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales",
    "DROP TABLE sales",
    "UPDATE sales SET amount = 0",
    "INSERT INTO sales VALUES (1)",
    "SELECT 1; DROP TABLE sales",
    "SELECT * FROM pg_catalog.pg_user",
    "SELECT * FROM users",
    "SELECT pg_sleep(100)",
    "COPY sales TO '/tmp/x'",
    "CREATE TABLE x AS SELECT * FROM sales",
    "SELECT * FROM sqlite_master",
    "SELECT load_extension('evil')",
])
def test_dangerous_sql_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        validate_select(sql, "postgresql")


def test_select_allowed_and_limited():
    sql = validate_select("SELECT region_name FROM regions", "postgresql", max_rows=100)
    assert "LIMIT 101" in sql.upper()
    with_cte = "WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t"
    assert validate_select(with_cte, "sqlite")


@pytest.mark.parametrize("expr", [
    "__import__('os').system('rm -rf /')",
    "fact.__class__",
    "open('x')",
    "fact ** 1000000",
    "[x for x in range(10**9)]",
    "lambda: 1",
    "fact if plan else 0",
    "'text'",
])
def test_formula_sandbox(expr):
    with pytest.raises(CalcError):
        parse_expr(expr)


def test_formula_division_by_zero_is_nan():
    import pandas as pd

    frame = pd.DataFrame({"fact": [10.0, 5.0], "plan": [0.0, 10.0]})
    result = eval_expr("(fact - plan) / plan * 100", frame)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(-50.0)


def test_database_is_read_only():
    from app import db

    with pytest.raises(Exception):
        db.run_select("DELETE FROM regions")
    frame = db.run_select("SELECT COUNT(*) AS n FROM regions")
    assert int(frame.iloc[0]["n"]) == 8
