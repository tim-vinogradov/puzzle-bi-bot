"""Вычисления над единой таблицей — на стороне бота, после выгрузки из базы.

Пользователь формулирует расчёт словами, бот превращает его в описание
вычисления (dict) — никакого исполняемого кода. Формулы разбираются модулем
`ast` по белому списку (числа, колонки, + − × ÷, скобки, abs/round), поэтому
«лишнего» формула сделать не может.

Каждое вычисление умеет считаться на двух уровнях:
* в таблице (строка = месяц × точка × продукт) — `materialize`;
* на уровне агрегации графика (например, по регионам) — `aggregate`.
Отношения («процент выполнения») пересчитываются после суммирования, а не
усредняются по строкам — так цифры на графиках и в KPI всегда корректны.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .semantic import DERIVED, DIM_LABELS, MEASURE_META, column_label, column_unit

MAX_EXPR_LEN = 300
MAX_EXPR_DEPTH = 25
ALLOWED_FUNCS = {"abs", "round"}

GRAIN_FREQ = {"month": "MS", "quarter": "QS", "year": "YS"}


class CalcError(ValueError):
    """Вычисление не удалось (ошибка в формуле, неизвестная колонка и т. п.)."""


# ---------------------------------------------------------------------------
# Безопасные формулы
# ---------------------------------------------------------------------------


def parse_expr(expr: str) -> ast.Expression:
    expr = (expr or "").strip()
    if not expr:
        raise CalcError("Пустая формула")
    if len(expr) > MAX_EXPR_LEN:
        raise CalcError("Слишком длинная формула")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as err:
        raise CalcError(f"Не удалось разобрать формулу «{expr}»") from err
    _check(tree.body, 0)
    return tree


def _check(node: ast.AST, depth: int) -> None:
    if depth > MAX_EXPR_DEPTH:
        raise CalcError("Слишком сложная формула")
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise CalcError("Разрешены только + − × ÷")
        _check(node.left, depth + 1)
        _check(node.right, depth + 1)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise CalcError("Недопустимая операция")
        _check(node.operand, depth + 1)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise CalcError("В формуле допустимы только числа и колонки")
    elif isinstance(node, ast.Name):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", node.id):
            raise CalcError(f"Недопустимое имя «{node.id}»")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS or node.keywords:
            raise CalcError("Из функций разрешены только abs() и round()")
        if not 1 <= len(node.args) <= 2:
            raise CalcError("Неверное число аргументов функции")
        for arg in node.args:
            _check(arg, depth + 1)
    else:
        raise CalcError(f"Недопустимый элемент формулы: {type(node).__name__}")


def expr_names(expr: str) -> list[str]:
    tree = parse_expr(expr)
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in ALLOWED_FUNCS})


def _safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.where(b != 0, np.nan)
    elif b == 0:
        b = np.nan
    return a / b


def eval_expr(expr: str, frame: pd.DataFrame) -> pd.Series:
    tree = parse_expr(expr)

    def ev(node: ast.AST):
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return _safe_div(left, right)
        if isinstance(node, ast.UnaryOp):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in frame.columns:
                raise CalcError(f"Нет колонки «{column_label(node.id)}» ({node.id})")
            return pd.to_numeric(frame[node.id], errors="coerce").astype(float)
        if isinstance(node, ast.Call):
            args = [ev(a) for a in node.args]
            if node.func.id == "abs":
                return abs(args[0])
            digits = int(args[1]) if len(args) > 1 else 0
            return args[0].round(digits) if isinstance(args[0], pd.Series) else round(args[0], digits)
        raise CalcError("Недопустимый элемент формулы")

    result = ev(tree.body)
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=frame.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def humanize_expr(expr: str, calcs: list[dict] | None = None) -> str:
    tree = parse_expr(expr)
    prec = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2}
    ops = {ast.Add: " + ", ast.Sub: " − ", ast.Mult: " × ", ast.Div: " ÷ "}

    def label(name: str) -> str:
        text = column_label(name, calcs)
        return re.sub(r",\s*(₽|шт|%)$", "", text)

    def render(node: ast.AST, parent: int = 0, right: bool = False) -> str:
        if isinstance(node, ast.BinOp):
            p = prec[type(node.op)]
            s = render(node.left, p) + ops[type(node.op)] + render(node.right, p, True)
            need = p < parent or (right and p == parent)
            return f"({s})" if need else s
        if isinstance(node, ast.UnaryOp):
            return "−" + render(node.operand, 3)
        if isinstance(node, ast.Constant):
            value = node.value
            return str(int(value)) if float(value).is_integer() else str(value).replace(".", ",")
        if isinstance(node, ast.Name):
            return label(node.id)
        if isinstance(node, ast.Call):
            inner = ", ".join(render(a) for a in node.args)
            return f"{'модуль' if node.func.id == 'abs' else 'округл'}({inner})"
        return "?"

    return render(tree.body)


# ---------------------------------------------------------------------------
# Описание вычислений
# ---------------------------------------------------------------------------

STAT_LABELS = {"mean": "Среднее", "median": "Медиана", "min": "Минимум", "max": "Максимум", "sum": "Сумма"}
DIM_GENITIVE = {
    "region": "региона", "outlet": "точки", "product": "продукта", "category": "категории",
    "client": "клиента", "city": "города", "segment": "сегмента", "reason": "причины",
    "outlet_type": "типа точки",
}
DIM_DATIVE = {
    "region": "региону", "outlet": "точке", "product": "продукту", "category": "категории",
    "client": "клиенту", "city": "городу", "segment": "сегменту", "reason": "причине",
    "outlet_type": "типу точки",
}


def _short(col: str, calcs: list[dict] | None = None) -> str:
    return re.sub(r",\s*(₽|шт|%)$", "", column_label(col, calcs)).lower()


def make_calc(kind: str, *, existing: list[dict], source_text: str = "", **params: Any) -> dict:
    """Создаёт описание вычисления с понятной подписью и формулой на русском."""
    measure = params.get("measure") or "fact"
    by = params.get("by")
    unit = column_unit(measure, existing) or ""
    calc: dict[str, Any] = {"kind": kind, "source_text": source_text,
                            "created_at": datetime.now().isoformat(timespec="seconds")}
    if kind == "expr":
        expr = params["expr"]
        parse_expr(expr)
        label = params.get("label") or "Расчёт"
        calc.update(expr=expr, label=label, unit=params.get("unit", ""),
                    formula=humanize_expr(expr, existing))
        base_id = params.get("id") or "calc"
    elif kind == "share":
        what = DIM_GENITIVE.get(by, "") if by else "строки"
        calc.update(measure=measure, by=by, unit="%",
                    label=params.get("label") or f"Доля {what} в общем объёме, %",
                    formula=f"{_short(measure, existing).capitalize()} {DIM_GENITIVE.get(by, '') if by else ''} ÷ итог по всем × 100".replace("  ", " "))
        base_id = f"share_{by or 'row'}"
    elif kind == "change":
        lag = params.get("lag", "prev")
        mode = params.get("mode", "pct")
        lag_text = {"prev": "к предыдущему периоду", "month": "к предыдущему месяцу", "year": "к прошлому году"}[lag]
        unit_out = "%" if mode == "pct" else unit
        calc.update(measure=measure, lag=lag, mode=mode, unit=unit_out,
                    label=params.get("label") or f"Динамика {lag_text}{', %' if mode == 'pct' else ', ' + unit}",
                    formula=(f"({_short(measure, existing)} ÷ {_short(measure, existing)} {lag_text} − 1) × 100"
                             if mode == "pct" else f"{_short(measure, existing)} − {_short(measure, existing)} {lag_text}"))
        base_id = f"{'yoy' if lag == 'year' else 'mom'}_{measure}"
    elif kind == "cumsum":
        calc.update(measure=measure, unit=unit,
                    label=params.get("label") or f"Накопительный итог: {_short(measure, existing)}{', ' + unit if unit else ''}",
                    formula=f"сумма «{_short(measure, existing)}» с начала периода нарастающим итогом")
        base_id = f"cum_{measure}"
    elif kind == "rank":
        desc = params.get("desc", True)
        calc.update(measure=measure, by=by, desc=desc, unit="",
                    label=params.get("label") or f"Ранг {DIM_GENITIVE.get(by, '') if by else ''} по «{_short(measure, existing)}»".replace("  ", " "),
                    formula=f"место по «{_short(measure, existing)}» ({'1 — наибольшее' if desc else '1 — наименьшее'})")
        base_id = f"rank_{by or 'row'}"
    elif kind == "group_stat":
        stat = params.get("stat", "mean")
        calc.update(measure=measure, by=by, stat=stat, unit=unit,
                    label=params.get("label") or f"{STAT_LABELS[stat]} «{_short(measure, existing)}» по {DIM_DATIVE.get(by, 'группе')}{', ' + unit if unit else ''}",
                    formula=f"{STAT_LABELS[stat].lower()} «{_short(measure, existing)}» внутри {DIM_GENITIVE.get(by, 'группы')}")
        base_id = f"{stat}_{measure}_{by or 'all'}"
    else:
        raise CalcError(f"Неизвестный тип вычисления: {kind}")
    ids = {c["id"] for c in existing}
    calc_id, n = base_id, 2
    while calc_id in ids or calc_id in MEASURE_META or calc_id in DERIVED or calc_id in DIM_LABELS:
        calc_id, n = f"{base_id}_{n}", n + 1
    calc["id"] = calc_id
    return calc


def calc_dependencies(calc: dict) -> list[str]:
    if calc["kind"] == "expr":
        return expr_names(calc["expr"])
    return [calc["measure"]]


# ---------------------------------------------------------------------------
# Агрегация и расчёт на уровне графика
# ---------------------------------------------------------------------------


def bucket(series: pd.Series, grain: str) -> pd.Series:
    series = pd.to_datetime(series)
    if grain == "quarter":
        return series.dt.to_period("Q").dt.start_time
    if grain == "year":
        return series.dt.to_period("Y").dt.start_time
    return series.dt.to_period("M").dt.start_time


def _offset(grain: str, lag: str) -> pd.DateOffset:
    if lag == "year":
        return pd.DateOffset(years=1)
    return {"month": pd.DateOffset(months=1), "quarter": pd.DateOffset(months=3), "year": pd.DateOffset(years=1)}[grain]


def _numeric_cols(frame: pd.DataFrame, base: list[str]) -> list[str]:
    return [c for c in base if c in frame.columns]


def _sum_by(frame: pd.DataFrame, by: list[str], measures: list[str]) -> pd.DataFrame:
    cols = _numeric_cols(frame, measures)
    if not by:
        return pd.DataFrame([frame[cols].sum()]) if len(frame) else pd.DataFrame([{c: 0.0 for c in cols}])
    return frame.groupby(by, dropna=False, observed=True, sort=True)[cols].sum().reset_index()


def resolve_order(needed: list[str], calcs: list[dict]) -> list[str]:
    """Порядок расчёта с учётом зависимостей (вычисление может ссылаться на другое)."""
    by_id = {c["id"]: c for c in calcs}
    order: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in order:
            return
        if name in visiting:
            raise CalcError("Циклическая ссылка между вычислениями")
        visiting.add(name)
        if name in by_id:
            for dep in calc_dependencies(by_id[name]):
                visit(dep)
        elif name in DERIVED:
            for dep in expr_names(DERIVED[name]["expr"]):
                visit(dep)
        visiting.discard(name)
        order.append(name)

    for name in needed:
        visit(name)
    return order


def aggregate(frame: pd.DataFrame, by: list[str], needed: list[str], calcs: list[dict], *,
              measures: list[str], grain: str = "month", prev: pd.DataFrame | None = None) -> pd.DataFrame:
    """Группирует таблицу по `by` (period — с учётом зерна) и считает нужные показатели."""
    by = list(dict.fromkeys(by))
    work = frame
    if "period" in by and "period" in frame.columns:
        work = frame.assign(period=bucket(frame["period"], grain))
    out = _sum_by(work, by, measures)
    by_id = {c["id"]: c for c in calcs}
    prev_work = None
    if prev is not None and len(prev):
        prev_work = prev.assign(period=bucket(prev["period"], grain)) if "period" in by and "period" in prev.columns else prev

    for name in resolve_order(needed, calcs):
        if name in out.columns:
            continue
        if name in DERIVED:
            out[name] = eval_expr(DERIVED[name]["expr"], out)
            continue
        calc = by_id.get(name)
        if calc is None:
            if name in measures:
                out[name] = 0.0
                continue
            raise CalcError(f"Неизвестный показатель «{name}»")
        out[name] = _calc_on_aggregate(calc, out, work, by, grain, prev_work, measures, calcs)
    return out


def _calc_on_aggregate(calc: dict, out: pd.DataFrame, raw: pd.DataFrame, by: list[str], grain: str,
                       prev: pd.DataFrame | None, measures: list[str], calcs: list[dict]) -> pd.Series:
    kind = calc["kind"]
    if kind == "expr":
        return eval_expr(calc["expr"], out)
    m = calc["measure"]
    if m not in out.columns:
        sub = aggregate(raw, by, [m], calcs, measures=measures, grain=grain)
        out[m] = sub[m].values if len(sub) == len(out) else np.nan
    values = pd.to_numeric(out[m], errors="coerce")
    others = [d for d in by if d != "period"]

    if kind == "share":
        if "period" in by and others:
            totals = out.groupby("period")[m].transform("sum")
        else:
            totals = values.sum()
        return _safe_div(values, totals) * 100

    if kind == "cumsum":
        if "period" not in by:
            return values
        ordered = out.sort_values("period")
        cum = ordered.groupby(others, dropna=False)[m].cumsum() if others else ordered[m].cumsum()
        return cum.reindex(out.index)

    if kind == "rank":
        asc = not calc.get("desc", True)
        if "period" in by and others:
            return out.groupby("period")[m].rank(ascending=asc, method="min")
        return values.rank(ascending=asc, method="min")

    if kind == "group_stat":
        stat = calc.get("stat", "mean")
        if m not in raw.columns:
            raise CalcError(f"Статистика «{STAT_LABELS[stat]}» доступна только для исходных колонок")
        if by:
            agg = raw.groupby(by, dropna=False, observed=True)[m].agg(stat).reset_index(name="__v")
            merged = out[by].merge(agg, on=by, how="left")
            return pd.Series(merged["__v"].values, index=out.index)
        return pd.Series(raw[m].agg(stat), index=out.index)

    if kind == "change":
        mode = calc.get("mode", "pct")
        lag = calc.get("lag", "prev")
        if "period" in by:
            combined_raw = pd.concat([prev, raw], ignore_index=True) if prev is not None else raw
            full = _sum_by(combined_raw, by, [c for c in measures if c in combined_raw.columns])
            if m not in full.columns:
                full = aggregate(combined_raw, by, [m], calcs, measures=measures, grain=grain)
            shifted = full[by + [m]].copy()
            shifted["period"] = shifted["period"] + _offset(grain, lag)
            shifted = shifted.rename(columns={m: "__prev"})
            merged = out[by].merge(shifted, on=by, how="left")
            previous = pd.Series(merged["__prev"].values, index=out.index)
        else:
            if prev is None or not len(prev):
                return pd.Series(np.nan, index=out.index)
            prev_agg = aggregate(prev, by, [m], calcs, measures=measures, grain=grain)
            if others:
                merged = out[others].merge(prev_agg[others + [m]].rename(columns={m: "__prev"}), on=others, how="left")
                previous = pd.Series(merged["__prev"].values, index=out.index)
            else:
                previous = pd.Series(prev_agg[m].iloc[0] if len(prev_agg) else np.nan, index=out.index)
        if mode == "pct":
            return (_safe_div(values, previous) - 1) * 100
        return values - previous

    raise CalcError(f"Неизвестный тип вычисления: {kind}")


# ---------------------------------------------------------------------------
# Расчёт в самой таблице (строка = месяц × точка × продукт)
# ---------------------------------------------------------------------------


def materialize(frame: pd.DataFrame, calcs: list[dict], *, dims: list[str], measures: list[str],
                prev: pd.DataFrame | None = None, only: list[str] | None = None) -> pd.DataFrame:
    """Возвращает таблицу с добавленными колонками вычислений."""
    out = frame.copy()
    key = [d for d in dims if d in out.columns and d != "period"]
    wanted = only if only is not None else [c["id"] for c in calcs]
    by_id = {c["id"]: c for c in calcs}
    for name in resolve_order(wanted, calcs):
        if name in out.columns:
            continue
        if name in DERIVED:
            out[name] = eval_expr(DERIVED[name]["expr"], out)
            continue
        calc = by_id.get(name)
        if calc is None:
            continue
        out[name] = _calc_on_rows(calc, out, key, prev)
    return out


def _calc_on_rows(calc: dict, out: pd.DataFrame, key: list[str], prev: pd.DataFrame | None) -> pd.Series:
    kind = calc["kind"]
    if kind == "expr":
        return eval_expr(calc["expr"], out)
    m = calc["measure"]
    if m not in out.columns:
        raise CalcError(f"Сначала нужна колонка «{column_label(m)}»")
    values = pd.to_numeric(out[m], errors="coerce")
    by = calc.get("by")
    if kind == "share":
        total = values.sum()
        if by and by in out.columns:
            return _safe_div(out.groupby(by, dropna=False)[m].transform("sum"), total) * 100
        return _safe_div(values, total) * 100
    if kind == "cumsum":
        if "period" not in out.columns:
            return values.cumsum()
        ordered = out.sort_values("period")
        return ordered.groupby(key, dropna=False)[m].cumsum().reindex(out.index) if key else ordered[m].cumsum().reindex(out.index)
    if kind == "rank":
        asc = not calc.get("desc", True)
        if by and by in out.columns:
            totals = out.groupby(by, dropna=False)[m].transform("sum")
            return totals.rank(ascending=asc, method="dense")
        return values.rank(ascending=asc, method="min")
    if kind == "group_stat":
        stat = calc.get("stat", "mean")
        if by and by in out.columns:
            return out.groupby(by, dropna=False)[m].transform(stat)
        return pd.Series(values.agg(stat), index=out.index)
    if kind == "change":
        if "period" not in out.columns:
            return pd.Series(np.nan, index=out.index)
        lag = calc.get("lag", "prev")
        offset = pd.DateOffset(years=1) if lag == "year" else pd.DateOffset(months=1)
        pool = pd.concat([prev, out[key + ["period", m]]], ignore_index=True) if prev is not None else out[key + ["period", m]]
        pool = pool.groupby(key + ["period"], dropna=False)[m].sum().reset_index()
        pool["period"] = pool["period"] + offset
        merged = out[key + ["period"]].merge(pool.rename(columns={m: "__prev"}), on=key + ["period"], how="left")
        previous = pd.Series(merged["__prev"].values, index=out.index)
        if calc.get("mode", "pct") == "pct":
            return (_safe_div(values, previous) - 1) * 100
        return values - previous
    raise CalcError(f"Неизвестный тип вычисления: {kind}")
