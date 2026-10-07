"""Действия, в которые переводится запрос пользователя.

И встроенный разборщик, и языковая модель выдают список таких действий
(обычные словари). Перед выполнением каждое действие проверяется здесь —
неизвестные поля и значения отбрасываются, поэтому модель не может
«протащить» ничего лишнего.
"""
from __future__ import annotations

from typing import Any

DIMS = {"region", "city", "outlet", "outlet_type", "category", "product", "client", "segment", "reason", "period"}
DATASETS = {"sales", "discounts", "clients"}
BUILD_INTENTS = {"dashboard", "chart", "compare", "structure", "planfact_dim", "trend", "top", "exec"}
METRICS = {"amount", "qty", "discount", "deals", "count", "avg_pct"}
GRAINS = {"month", "quarter", "year"}
VIEWS = {"abs", "pct", "dyn"}
SERIES = {"both", "plan", "fact"}
CHART_TYPES = {"planfact_period", "structure", "top", "exec_line", "heatmap", "planfact_dim", "scatter_planfact",
               "trend", "bar_dim"}
CALC_KINDS = {"expr", "share", "change", "cumsum", "rank", "group_stat"}
PERIOD_KINDS = {"last_quarter", "last_month", "last_year", "last_n_months", "ytd", "year", "quarter", "month",
                "all", "range", "prev_year"}
EXPORT_FORMATS = {"html", "csv", "xlsx", "png", "pdf"}

ACTION_TYPES = {
    "build", "custom_sql", "show_table", "add_calc", "remove_calc", "modify_calc", "rebuild_chart", "add_chart",
    "remove_chart", "set_filter", "set_switch", "save", "list", "open", "rename", "delete", "restore", "copy",
    "export", "undo", "history", "revert", "refresh", "help", "clarify", "ui_info", "reply", "share",
    "set_period", "answer", "about_data", "show_sql",
}
ANSWER_KINDS = {"top", "bottom", "total", "count", "share", "entity", "yoy"}


class ActionError(ValueError):
    pass


def _pick(value: Any, allowed: set, default: Any = None) -> Any:
    return value if value in allowed else default


def _str(value: Any, limit: int = 200) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _int(value: Any, lo: int, hi: int) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, number))


def _str_list(value: Any, limit: int = 50) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip()[:200] for v in list(value)[:limit] if str(v).strip()]


def validate_period(period: Any) -> dict | None:
    if not isinstance(period, dict):
        return None
    kind = _pick(period.get("kind"), PERIOD_KINDS)
    if not kind:
        return None
    out: dict[str, Any] = {"kind": kind}
    for key in ("year", "quarter", "month", "n", "start_month", "start_year", "end_month", "end_year"):
        if key in period:
            num = _int(period[key], 0, 3000)
            if num is not None:
                out[key] = num
    for key in ("start", "end"):
        if key in period:
            out[key] = str(period[key])[:10] if period[key] else None
    return out


def validate_calc(calc: Any) -> dict:
    if not isinstance(calc, dict):
        raise ActionError("Некорректное описание вычисления")
    kind = _pick(calc.get("kind"), CALC_KINDS)
    if not kind:
        raise ActionError("Неизвестный тип вычисления")
    out: dict[str, Any] = {"kind": kind}
    if kind == "expr":
        expr = _str(calc.get("expr"), 300)
        if not expr:
            raise ActionError("Нет формулы")
        out["expr"] = expr
        out["label"] = _str(calc.get("label"), 80) or "Расчёт"
        out["unit"] = _pick(calc.get("unit"), {"%", "₽", "шт", ""}, "")
    else:
        out["measure"] = _str(calc.get("measure"), 40) or "fact"
        if calc.get("by"):
            out["by"] = _pick(calc.get("by"), DIMS - {"period"})
        if kind == "change":
            out["lag"] = _pick(calc.get("lag"), {"prev", "month", "year"}, "prev")
            out["mode"] = _pick(calc.get("mode"), {"pct", "abs"}, "pct")
        if kind == "rank":
            out["desc"] = bool(calc.get("desc", True))
        if kind == "group_stat":
            out["stat"] = _pick(calc.get("stat"), {"mean", "median", "min", "max", "sum"}, "mean")
        if calc.get("label"):
            out["label"] = _str(calc.get("label"), 80)
    return out


def validate_action(action: Any) -> dict:
    if not isinstance(action, dict):
        raise ActionError("Действие должно быть объектом")
    kind = action.get("type")
    if kind not in ACTION_TYPES:
        raise ActionError(f"Неизвестное действие: {kind}")
    out: dict[str, Any] = {"type": kind}
    if kind == "build":
        out["dataset"] = _pick(action.get("dataset"), DATASETS, "sales")
        out["intent"] = _pick(action.get("intent"), BUILD_INTENTS, "dashboard")
        out["period"] = validate_period(action.get("period"))
        dims = [d for d in _str_list(action.get("dims")) if d in DIMS - {"period"}]
        out["dims"] = dims
        out["metric"] = _pick(action.get("metric"), METRICS)
        out["grain"] = _pick(action.get("grain"), GRAINS)
        out["top_n"] = _int(action.get("top_n"), 1, 50) if action.get("top_n") else None
        out["lagging"] = bool(action.get("lagging", False))
        out["structure_dim"] = _pick(action.get("structure_dim"), DIMS - {"period"})
        filters = action.get("filters") or {}
        out["filters"] = {k: _str_list(v) for k, v in filters.items() if k in DIMS - {"period"}} if isinstance(filters, dict) else {}
        out["client"] = _str(action.get("client"), 120)
        out["query"] = _str(action.get("query"), 500)
        out["sort_by"] = _pick(action.get("sort_by"), {"exec_pct", "deviation", "fact"})
    elif kind == "custom_sql":
        out["sql"] = _str(action.get("sql"), 8000)
        out["title"] = _str(action.get("title"), 120) or "Результат запроса"
        if not out["sql"]:
            raise ActionError("Нет SQL")
    elif kind == "show_table":
        out["group_by"] = [d for d in _str_list(action.get("group_by")) if d in DIMS]
        out["sort_by"] = _str(action.get("sort_by"), 60)
        out["sort_dir"] = _pick(action.get("sort_dir"), {"asc", "desc"}, "desc")
        out["top_n"] = _int(action.get("top_n"), 1, 1000) if action.get("top_n") else None
    elif kind == "add_calc":
        calcs = action.get("calcs") or ([action.get("calc")] if action.get("calc") else [])
        out["calcs"] = [validate_calc(c) for c in calcs]
        if not out["calcs"]:
            raise ActionError("Не указано вычисление")
        out["rebuild"] = _int(action.get("rebuild"), 0, 6) if action.get("rebuild") is not None else None
    elif kind in ("remove_calc", "modify_calc"):
        out["ref"] = _str(action.get("ref"), 120) or "last"
        if kind == "modify_calc":
            out["calc"] = validate_calc(action.get("calc"))
    elif kind in ("rebuild_chart", "add_chart", "remove_chart"):
        out["chart"] = action.get("chart") if action.get("chart") in ("all", "last") else _int(action.get("chart"), 1, 6)
        changes = action.get("changes") or {}
        clean: dict[str, Any] = {}
        if isinstance(changes, dict):
            if changes.get("type") in CHART_TYPES:
                clean["type"] = changes["type"]
            for key in ("dim", "split", "structure_dim"):
                if changes.get(key) in DIMS or changes.get(key) == "@switch":
                    clean[key] = changes[key]
            for key in ("sort_by", "value", "title"):
                if changes.get(key) is not None:
                    clean[key] = _str(changes.get(key), 120)
            if changes.get("sort_dir") in ("asc", "desc"):
                clean["sort_dir"] = changes["sort_dir"]
            if changes.get("top_n"):
                clean["top_n"] = _int(changes["top_n"], 1, 50)
            if changes.get("grain") in GRAINS:
                clean["grain"] = changes["grain"]
            if changes.get("view") in VIEWS:
                clean["view"] = changes["view"]
            if changes.get("variant") in ("heatmap", None, "line"):
                if "variant" in changes:
                    clean["variant"] = changes.get("variant")
            if changes.get("metric") in METRICS:
                clean["metric"] = changes["metric"]
            if changes.get("mode") in ("bar", "line"):
                clean["mode"] = changes["mode"]
            if changes.get("use_last_calc"):
                clean["use_last_calc"] = True
        out["changes"] = clean
    elif kind == "set_filter":
        out["field"] = _pick(action.get("field"), DIMS)
        out["values"] = _str_list(action.get("values"), 400)
        out["mode"] = _pick(action.get("mode"), {"only", "add", "exclude", "clear", "clear_all"}, "only")
        if action.get("period"):
            out["period"] = validate_period(action.get("period"))
        if not out["field"] and out["mode"] != "clear_all":
            raise ActionError("Не указано поле фильтра")
    elif kind == "set_switch":
        for key, allowed in (("dim", DIMS - {"period"}), ("metric", METRICS), ("grain", GRAINS), ("view", VIEWS), ("series", SERIES)):
            if action.get(key) in allowed:
                out[key] = action[key]
        if len(out) == 1:
            raise ActionError("Не указано, что переключить")
    elif kind in ("save", "rename", "open", "delete", "copy", "restore"):
        out["name"] = _str(action.get("name"), 120)
        out["new_name"] = _str(action.get("new_name"), 120)
        out["as_new"] = bool(action.get("as_new", False))
        out["id"] = _str(action.get("id"), 40)
        out["force"] = bool(action.get("force", False))
    elif kind == "list":
        out["query"] = _str(action.get("query"), 120)
        out["sort"] = _pick(action.get("sort"), {"updated", "created", "name"}, "updated")
    elif kind == "export":
        out["format"] = _pick(action.get("format"), EXPORT_FORMATS, "html")
    elif kind == "set_period":
        out["period"] = validate_period(action.get("period"))
        if not out["period"]:
            raise ActionError("Не указан период")
    elif kind == "answer":
        out["kind"] = _pick(action.get("kind"), ANSWER_KINDS)
        if not out["kind"]:
            raise ActionError("Неизвестный тип вопроса")
        out["period"] = validate_period(action.get("period"))
        out["dim"] = _pick(action.get("dim"), DIMS - {"period"})
        out["field"] = _pick(action.get("field"), DIMS - {"period"})
        out["value"] = _str(action.get("value"), 200)
        out["what"] = _pick(action.get("what"), {"fact", "plan", "qty", "deals", "discount", "client", "outlet", "product", "region", "category"})
        out["measure"] = _pick(action.get("measure"), {"fact", "qty", "discount", "exec", "growth"})
        out["lagging"] = bool(action.get("lagging"))
        out["n"] = _int(action.get("n"), 1, 20) if action.get("n") else None
        out["query"] = _str(action.get("query"), 500)
    elif kind == "help":
        out["greeting"] = bool(action.get("greeting"))
    elif kind == "revert":
        out["version"] = _int(action.get("version"), 1, 100000)
    elif kind in ("reply", "ui_info"):
        out["text"] = _str(action.get("text"), 2000) or ""
        out["topic"] = _str(action.get("topic"), 40)
    elif kind == "clarify":
        out["question"] = _str(action.get("question"), 500) or "Уточните, пожалуйста, запрос"
        options = action.get("options") or []
        out["options"] = [o for o in options if isinstance(o, dict) and o.get("label")][:8]
        out["pending"] = action.get("pending") if isinstance(action.get("pending"), dict) else None
    return out
