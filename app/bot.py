"""Диалоговая логика бота: принимает сообщение, понимает его, выполняет
действия и возвращает блоки ответа (текст, SQL, таблица, дашборд, список…)."""
from __future__ import annotations

import concurrent.futures
import copy
import html
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from . import fmt, semantic, storage
from .calc import CalcError, aggregate, make_calc, materialize
from .charts import CHART_TYPE_LABELS
from .config import settings
from .dashboard import DashboardState, default_charts, default_switches, now_iso, render
from .db import QueryError
from .nlu import llm, rules
from .nlu.actions import ActionError, validate_action
from .semantic import (DERIVED, DIM_LABELS, DIM_LABELS_PLURAL, MONTHS_NOM, Period, column_label,
                       column_unit)
from .sql_guard import UnsafeSQLError

log = logging.getLogger(__name__)

MAX_UNDO = 40
MAX_CHARTS = 6
PAGE_SIZE = 50


def esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


def text_block(text: str, tone: str = "normal") -> dict:
    return {"type": "text", "text": text, "tone": tone}


def error_block(text: str) -> dict:
    return {"type": "text", "text": text, "tone": "error"}


class BotError(Exception):
    """Ошибка, которую нужно показать пользователю как обычный ответ."""


# ---------------------------------------------------------------------------
# Сессия
# ---------------------------------------------------------------------------


@dataclass
class Session:
    id: str
    messages: list[dict] = field(default_factory=list)
    draft: DashboardState | None = None
    undo: list[DashboardState | None] = field(default_factory=list)
    pending: dict | None = None
    llm_history: list[dict] = field(default_factory=list)
    last_chart: int | None = None
    table_view: dict = field(default_factory=dict)
    last_deleted: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def push_undo(self) -> None:
        self.undo.append(self.draft.clone() if self.draft else None)
        if len(self.undo) > MAX_UNDO:
            self.undo.pop(0)

    def persist(self) -> None:
        try:
            frames = (self.draft.table, self.draft.prev, self.draft.links) if self.draft else None
            meta = self.draft.meta() if self.draft else None
            if meta and self.draft:
                meta["saved_id"] = self.draft.saved_id
                meta["saved_version"] = self.draft.saved_version
            storage.save_session(self.id, self.messages[-200:], meta, frames)
        except Exception:  # noqa: BLE001 — автосохранение не должно ломать ответ
            log.exception("Не удалось сохранить черновик сессии")


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str | None) -> Session:
    session_id = (session_id or "").strip()[:64] or uuid.uuid4().hex
    session_id = re.sub(r"[^A-Za-z0-9_-]", "", session_id) or uuid.uuid4().hex
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            session = Session(session_id)
            loaded = storage.load_session(session_id)
            if loaded:
                messages, meta, frames = loaded
                session.messages = messages
                if meta and frames:
                    try:
                        state = DashboardState.from_meta(meta, *frames)
                        state.saved_id = meta.get("saved_id")
                        state.saved_version = meta.get("saved_version")
                        session.draft = state
                    except Exception:  # noqa: BLE001
                        log.exception("Не удалось восстановить черновик")
            _sessions[session_id] = session
        return session


def reset_session(session_id: str) -> Session:
    with _sessions_lock:
        session = Session(session_id)
        _sessions[session_id] = session
    session.persist()
    return session


# ---------------------------------------------------------------------------
# Периоды
# ---------------------------------------------------------------------------


def resolve_period(spec: dict | None) -> Period | None:
    if not spec:
        return None
    kind = spec.get("kind")
    anchor = semantic.anchor_date()
    last = semantic.last_complete_month(anchor)
    if kind == "last_quarter":
        period = semantic.period_last_quarter(anchor)
    elif kind == "last_month":
        period = semantic.period_last_months(1, anchor, f"{MONTHS_NOM[last.month - 1]} {last.year}", "month")
    elif kind == "last_year":
        period = semantic.period_last_months(12, anchor, "последние 12 месяцев", "last_year")
    elif kind == "last_n_months":
        n = max(1, min(int(spec.get("n") or 12), 120))
        if n % 12 == 0 and n >= 24:
            years = n // 12
            label = f"последние {years} {fmt.plural(years, 'год', 'года', 'лет')}"
        elif n == 12:
            label = "последние 12 месяцев"
        else:
            label = f"последние {n} {fmt.plural(n, 'месяц', 'месяца', 'месяцев')}"
        period = semantic.period_last_months(n, anchor, label, "last_n_months")
    elif kind == "ytd":
        period = semantic.period_ytd(anchor)
    elif kind == "all":
        period = semantic.period_all()
    elif kind == "year":
        period = semantic.period_year(int(spec.get("year") or last.year))
    elif kind == "prev_year":
        period = semantic.period_year(last.year - 1)
    elif kind == "quarter":
        q = int(spec.get("quarter") or 1)
        year = spec.get("year")
        if not year:
            year = last.year if 3 * q <= last.month else last.year - 1
        period = semantic.period_quarter(q, int(year))
    elif kind == "month":
        m = int(spec.get("month") or 1)
        year = spec.get("year")
        if not year:
            year = last.year if m <= last.month else last.year - 1
        period = semantic.period_month(m, int(year))
    elif kind == "range":
        if spec.get("start_month"):
            sy = spec.get("start_year") or (last.year if spec["start_month"] <= last.month else last.year - 1)
            ey = spec.get("end_year") or sy if spec.get("end_month", 12) >= spec["start_month"] else (spec.get("end_year") or sy + 1)
            start = date(int(sy), int(spec["start_month"]), 1)
            end = semantic.month_end(date(int(ey), int(spec.get("end_month") or 12), 1))
        else:
            start = date.fromisoformat(spec["start"])
            end = date.fromisoformat(spec["end"]) if spec.get("end") else semantic.month_end(last)
        period = Period(semantic.month_start(start), semantic.month_end(end), "", "range")
        period.label = period.range_label()
        if period.start.year == period.end.year and (period.start.month, period.end.month) in ((1, 6), (7, 12)):
            period.label = f"{1 if period.start.month == 1 else 2} полугодие {period.start.year}"
        elif period.start.year == period.end.year and (period.start.month, period.end.month) == (1, 12):
            period.label = f"{period.start.year} год"
    else:
        return None
    clamped = semantic.clamp_to_data(period)
    return clamped


PERIOD_OPTIONS = [
    ("Последний квартал", {"kind": "last_quarter"}),
    ("Последние 12 месяцев", {"kind": "last_year"}),
    ("С начала года", {"kind": "ytd"}),
    ("Прошлый год", {"kind": "prev_year"}),
    ("Всё время", {"kind": "all"}),
]


# ---------------------------------------------------------------------------
# Главная точка входа
# ---------------------------------------------------------------------------


def nlu_context(session: Session) -> rules.NLUContext:
    draft = session.draft
    ctx = rules.NLUContext(catalog=semantic.catalog())
    if draft:
        ctx.has_dashboard = True
        ctx.dataset = draft.dataset
        ctx.charts = [{"type": c["type"], "title": c.get("title", "")} for c in draft.charts]
        ctx.calcs = [{"id": c["id"], "label": c["label"], "kind": c["kind"]} for c in draft.calcs]
        ctx.switches = dict(draft.switches)
        ctx.current_name = draft.name if draft.saved_id else None
        ctx.last_chart = session.last_chart
        ctx.columns = list(draft.table.columns)
    return ctx


def understand(session: Session, text: str) -> list[dict]:
    ctx = nlu_context(session)
    mode = settings.nlu_mode
    use_llm = settings.llm_enabled and mode in ("hybrid", "llm")
    if use_llm and mode == "llm":
        try:
            return llm.understand(text, ctx, session.llm_history)
        except llm.LLMError as err:
            log.warning("LLM: %s — используем встроенный разборщик", err)
    result = rules.parse(text, copy.deepcopy(ctx))
    result.actions = [validate_action(a) for a in result.actions]
    if result.actions and (result.confident or not use_llm):
        return result.actions
    if use_llm:
        try:
            return llm.understand(text, ctx, session.llm_history)
        except llm.LLMError as err:
            log.warning("LLM: %s", err)
    return result.actions


def handle_message(session_id: str | None, text: str) -> dict:
    session = get_session(session_id)
    text = (text or "").strip()[:2000]
    with session.lock:
        session.messages.append({"role": "user", "text": text, "ts": now_iso()})
        blocks: list[dict] = []
        try:
            actions = _pending_answer(session, text)
            if actions is None:
                session.pending = None
                actions = understand(session, text)
            if not actions:
                blocks = [text_block(
                    "Не совсем понял запрос. Попробуйте, например: «Собери дашборд по продажам за последний год» "
                    "или «Покажи план-факт по регионам за последний квартал». Напишите «помощь», чтобы увидеть все возможности."),
                ]
                suggestions = starter_suggestions()
            else:
                blocks, suggestions = execute_actions(session, actions)
        except BotError as err:
            blocks, suggestions = [error_block(str(err))], []
        except (QueryError, UnsafeSQLError) as err:
            blocks, suggestions = [error_block(f"Не удалось получить данные: {esc(err)}")], []
        except CalcError as err:
            blocks, suggestions = [error_block(f"Не получилось посчитать: {esc(err)}")], []
        except Exception as err:  # noqa: BLE001
            log.exception("Ошибка обработки сообщения")
            blocks, suggestions = [error_block(f"Что-то пошло не так: {esc(err)}. Попробуйте переформулировать.")], []
        return _finish(session, blocks, suggestions, user_text=text)


def handle_action(session_id: str | None, action: dict) -> dict:
    session = get_session(session_id)
    with session.lock:
        try:
            clean = validate_action(action)
            session.pending = None
            label = action.get("_label")
            if label:
                session.messages.append({"role": "user", "text": str(label)[:200], "ts": now_iso()})
            blocks, suggestions = execute_actions(session, [clean])
        except (ActionError, BotError) as err:
            blocks, suggestions = [error_block(str(err))], []
        except (QueryError, UnsafeSQLError) as err:
            blocks, suggestions = [error_block(f"Не удалось получить данные: {esc(err)}")], []
        except CalcError as err:
            blocks, suggestions = [error_block(f"Не получилось посчитать: {esc(err)}")], []
        except Exception as err:  # noqa: BLE001
            log.exception("Ошибка выполнения действия")
            blocks, suggestions = [error_block(f"Что-то пошло не так: {esc(err)}")], []
        return _finish(session, blocks, suggestions)


def _finish(session: Session, blocks: list[dict], suggestions: list[str], user_text: str | None = None) -> dict:
    message = {"role": "bot", "blocks": blocks, "suggestions": suggestions, "ts": now_iso(), "id": uuid.uuid4().hex[:10]}
    stored = {**message, "blocks": [_stub(b) for b in blocks]}
    session.messages.append(stored)
    if user_text:
        summary = " ".join(re.sub(r"<[^>]+>", "", b.get("text", "")) for b in blocks if b["type"] == "text")[:600]
        session.llm_history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": summary or "ok"}])
        session.llm_history = session.llm_history[-12:]
    session.persist()
    return {"session_id": session.id, "message": message, "draft_uid": session.draft.uid if session.draft else None}


def _stub(block: dict) -> dict:
    if block["type"] in ("dashboard", "dashboard_update"):
        dash = block["dashboard"]
        return {"type": block["type"], "stub": True, "uid": dash["uid"], "name": dash.get("name"),
                "period_label": dash.get("period_label"), "charts": len(dash.get("charts", []))}
    return block


def _pending_answer(session: Session, text: str) -> list[dict] | None:
    pending = session.pending
    if not pending:
        return None
    t = rules.norm(text)
    options = pending.get("options") or []
    for i, option in enumerate(options, 1):
        label = rules.norm(option["label"])
        if t == label or t == str(i) or (len(t) > 3 and (label.startswith(t) or t in label)):
            session.pending = None
            return [validate_action(option["action"])]
    slot = pending.get("slot")
    if slot and pending.get("action"):
        action = dict(pending["action"])
        if slot == "period":
            period = rules.parse_period(t)
            if period and rules.is_period_only(t):
                action["period"] = period
                session.pending = None
                return [validate_action(action)]
            return None
        if slot in ("new_name", "name"):
            action[slot] = text.strip().strip("«»\"")
            session.pending = None
            return [validate_action(action)]
    return None


# ---------------------------------------------------------------------------
# Выполнение действий
# ---------------------------------------------------------------------------


def execute_actions(session: Session, actions: list[dict]) -> tuple[list[dict], list[str]]:
    blocks: list[dict] = []
    suggestions: list[str] = []
    dashboard_shown = False
    for i, action in enumerate(actions):
        handler = HANDLERS.get(action["type"])
        if handler is None:
            blocks.append(error_block("Это действие пока не поддерживается."))
            continue
        result = handler(session, action)
        new_blocks, new_suggestions = result if isinstance(result, tuple) else (result, [])
        # если несколько действий подряд меняют дашборд — показываем его один раз, в конце
        if i < len(actions) - 1 and any(b["type"] == "dashboard" for b in new_blocks):
            new_blocks = [b for b in new_blocks if b["type"] != "dashboard"]
            dashboard_shown = True
        blocks.extend(new_blocks)
        suggestions = new_suggestions or suggestions
        if any(b["type"] == "clarify" for b in new_blocks):
            break
    if dashboard_shown and session.draft and not any(b["type"] == "dashboard" for b in blocks):
        blocks.append(dashboard_block(session))
    return blocks, suggestions[:6]


def require_draft(session: Session) -> DashboardState:
    if session.draft is None:
        raise BotError("Сначала нужны данные. Напишите, например: «Собери дашборд по продажам за последний год».")
    return session.draft


def dashboard_block(session: Session) -> dict:
    draft = require_draft(session)
    payload = render(draft)
    payload["name"] = draft.name
    return {"type": "dashboard", "dashboard": payload}


def dashboard_update_block(session: Session) -> dict:
    block = dashboard_block(session)
    block["type"] = "dashboard_update"
    return block


def run_with_timeout(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=settings.calc_timeout_sec)
        except concurrent.futures.TimeoutError as err:
            raise CalcError(f"вычисление заняло больше {settings.calc_timeout_sec:.0f} с и было остановлено") from err


# --- построение -------------------------------------------------------------

INTENT_TEXT = {
    "dashboard": "дашборд из четырёх графиков",
    "compare": "сравнение и поиск отстающих",
    "structure": "структуру (доли)",
    "planfact_dim": "план-факт в разрезе",
    "trend": "динамику по периодам",
    "top": "рейтинг (ТОП)",
    "exec": "динамику выполнения плана",
    "chart": "график",
}


def do_build(session: Session, action: dict) -> tuple[list[dict], list[str]]:
    period = resolve_period(action.get("period"))
    dataset = action.get("dataset") or "sales"
    inherited = False
    if period is None and session.draft is not None and session.draft.dataset != "custom":
        # период не назван — берём период текущего дашборда (продолжение разговора)
        period = session.draft.period
        inherited = True
    if period is None:
        options = [{"label": label, "action": {**action, "period": spec}} for label, spec in PERIOD_OPTIONS]
        session.pending = {"options": options, "slot": "period", "action": action}
        what = {"discounts": "скидки", "clients": "продажи по клиентам"}.get(dataset, "продажи")
        return [{"type": "clarify", "question": f"За какой период показать {what}? Можно выбрать вариант или написать свой — например, «2 квартал 2026» или «с марта по август 2025».",
                 "options": options}], []

    filters = {k: list(v) for k, v in (action.get("filters") or {}).items() if v}
    catalog = semantic.catalog()
    if action.get("client"):
        fragment = action["client"]
        matches = rules.match_clients(fragment, catalog["client"])
        if len(matches) == 1:
            filters["client"] = matches
        else:
            candidates = matches[:6] if matches else _top_clients(dataset, period)
            options = [{"label": c, "action": {**action, "client": None, "filters": {**filters, "client": [c]}}} for c in candidates]
            options.append({"label": "Все клиенты", "action": {**action, "client": None}})
            session.pending = {"options": options}
            question = (f"Нашёл несколько клиентов по запросу «{esc(fragment)}». Кого показать?" if matches
                        else f"Клиента «{esc(fragment)}» в базе нет. Выберите из крупнейших или покажу по всем клиентам:")
            return [{"type": "clarify", "question": question, "options": options}], []

    fetched = semantic.fetch_dataset(dataset, period)
    if fetched.table.empty:
        dmin, dmax = semantic.db.data_bounds()
        raise BotError(f"За период «{esc(period.label)}» ({period.range_label()}) данных нет. "
                       f"В базе есть продажи с {semantic.month_label(dmin)} по {semantic.month_label(dmax)}.")

    state = DashboardState(dataset=dataset, query=action.get("query") or "", period=period, table=fetched.table,
                           prev=fetched.prev, links=fetched.links, sql=fetched.sql)
    dims = [d for d in (action.get("dims") or []) if d in state.table.columns]
    if dataset == "discounts":
        # разрез «клиент» бессмысленен, когда выбран один клиент — начинаем с причин
        dims = [d for d in dims if d != "client" or not filters.get("client")] or ["reason"]
    intent = action.get("intent") or "dashboard"
    state.switches = default_switches(state, dims[0] if dims else None, action.get("metric"), action.get("grain"))
    if intent == "compare" and not dims:
        state.switches["dim"] = "outlet" if dataset == "sales" else state.switches["dim"]
    state.switches["series"] = "both"
    state.charts = default_charts(dataset, intent, dim=dims[0] if dims else None,
                                  structure_dim=action.get("structure_dim"), top_n=action.get("top_n"),
                                  lagging=bool(action.get("lagging")), sort_by=action.get("sort_by"))
    state.filters = _normalize_filters(state, filters)
    state.name = suggest_name(state, intent)
    state.log(f"Построен по запросу «{state.query}»" if state.query else "Построен дашборд")
    session.push_undo()
    session.draft = state
    session.last_chart = None
    session.table_view = {}

    spec = state.spec
    dims_text = ", ".join(DIM_LABELS_PLURAL.get(d, d).lower() for d in dims) if dims else ""
    understood = f"Понял запрос: <b>{INTENT_TEXT.get(intent, 'визуализация')}</b>"
    if dims_text:
        understood += f" — {dims_text}"
    understood += f", период — <b>{esc(period.label)}</b>" + (f" ({period.range_label()})" if period.label != period.range_label() else "")
    understood += " — как в предыдущем дашборде; другой период: «за 2025 год»." if inherited else "."
    if filters:
        understood += " Фильтры: " + "; ".join(f"{DIM_LABELS.get(k, k)} = {', '.join(esc(v) for v in vals[:4])}{'…' if len(vals) > 4 else ''}"
                                               for k, vals in filters.items()) + "."
    loaded = (f"Выгрузил из базы единую таблицу «{spec.label}»: <b>{fmt.num(len(state.table))}</b> строк "
              f"({spec.description.lower()}) за {fetched.seconds:.1f} с. Все вычисления дальше — на стороне бота, без повторных запросов к базе.")
    blocks = [
        text_block(understood + "<br>" + loaded),
        {"type": "sql", "sql": fetched.sql, "tables": spec.tables, "rows": int(len(state.table))},
        table_block(session, page=0, preview=True),
        dashboard_block(session),
    ]
    return blocks, after_build_suggestions(state)


def _top_clients(dataset: str, period: Period) -> list[str]:
    try:
        frame = semantic.fetch_dataset("discounts" if dataset == "discounts" else "clients", period).table
        col = "discount" if dataset == "discounts" else "fact"
        return list(frame.groupby("client")[col].sum().sort_values(ascending=False).head(6).index)
    except Exception:  # noqa: BLE001
        return semantic.catalog()["client"][:6]


def _normalize_filters(state: DashboardState, filters: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    table = state.table
    for key, values in filters.items():
        if not values:
            continue
        if key in ("city", "outlet_type") and "outlet" in table.columns and key in table.columns:
            outlets = sorted(table.loc[table[key].isin(values), "outlet"].unique())
            out["outlet"] = sorted(set(out.get("outlet", [])) | set(outlets))
            continue
        if key in table.columns:
            present = set(table[key].astype(str).unique())
            kept = [v for v in values if v in present]
            if kept:
                out[key] = kept
    return out


def suggest_name(state: DashboardState, intent: str = "dashboard") -> str:
    period = state.period.label
    if state.dataset == "discounts":
        client = state.filters.get("client")
        who = f": {client[0]}" if client and len(client) == 1 else ""
        return f"Скидки{who} · {period}"
    if state.dataset == "clients":
        return f"Продажи по клиентам · {period}"
    if state.dataset == "custom":
        return f"Запрос · {period}"
    dim = DIM_LABELS_PLURAL.get(state.switches.get("dim", ""), "").lower()
    base = {"compare": f"Сравнение: {dim}", "structure": "Структура продаж", "top": f"ТОП: {dim}",
            "exec": "Выполнение плана", "trend": "Продажи: динамика", "planfact_dim": f"План-факт: {dim}"}.get(intent, f"Продажи: план-факт по {semantic_dative(state.switches.get('dim', 'region'))}")
    return f"{base} · {period}"


def semantic_dative(dim: str) -> str:
    return {"region": "регионам", "outlet": "точкам", "category": "категориям", "product": "продуктам"}.get(dim, "разрезам")


def starter_suggestions() -> list[str]:
    return [
        "Собери дашборд по продажам: план-факт по регионам и торговым точкам за последний год",
        "Покажи план-факт продаж по регионам за последний квартал",
        "Сравни продажи по торговым точкам за год, выдели отстающих",
        "Покажи по клиенту Сибирский Продукт за последние 3 года, какие скидки делались и почему",
        "Покажи мои дашборды",
    ]


def after_build_suggestions(state: DashboardState) -> list[str]:
    if state.dataset == "sales":
        return ["Добавь колонку с процентом выполнения плана", "Оставь только Центральный регион",
                "Переключи разрез на точки", "Добавь динамику к предыдущему месяцу", "Сохрани дашборд"]
    if state.dataset == "discounts":
        return ["Покажи таблицу", "Посчитай долю каждой причины в общем объёме", "Разбивка по кварталам", "Сохрани дашборд"]
    return ["Покажи таблицу", "Сохрани дашборд"]


def do_custom_sql(session: Session, action: dict) -> tuple[list[dict], list[str]]:
    frame, sql = semantic.fetch_custom(action["sql"])
    if frame.empty:
        return [{"type": "sql", "sql": sql, "tables": [], "rows": 0}, text_block("Запрос выполнен, но строк не вернул.")], []
    frame = frame.copy()
    date_cols = [c for c in frame.columns if re.search(r"date|month|period|дата|месяц", str(c), re.I)]
    if date_cols and "period" not in frame.columns:
        frame = frame.rename(columns={date_cols[0]: "period"})
    if "period" in frame.columns:
        frame["period"] = pd.to_datetime(frame["period"].astype(str).str[:10], errors="coerce")
        frame = frame.dropna(subset=["period"])
    measures = [c for c in frame.columns if c != "period" and pd.api.types.is_numeric_dtype(frame[c])]
    dims = [c for c in frame.columns if c not in measures]
    for d in dims:
        if d != "period":
            frame[d] = frame[d].astype(str)
    if not measures:
        frame["строк"] = 1.0
        measures = ["строк"]
    start = frame["period"].min().date() if "period" in frame.columns and len(frame) else semantic.anchor_date()
    end = frame["period"].max().date() if "period" in frame.columns and len(frame) else semantic.anchor_date()
    period = Period(semantic.month_start(start), semantic.month_end(end), "по запросу", "custom")
    state = DashboardState(dataset="custom", query=action.get("title", ""), period=period, table=frame, sql=sql,
                           custom_spec={"dims": dims, "measures": measures})
    state.switches = {"dim": next((d for d in dims if d != "period"), "period"), "metric": measures[0], "grain": "month", "view": "abs"}
    state.charts = ([{"id": uuid.uuid4().hex[:8], "type": "trend", "mode": "line"}] if "period" in frame.columns else []) + \
                   ([{"id": uuid.uuid4().hex[:8], "type": "bar_dim", "dim": "@switch", "top_n": 15}] if len(dims) > (1 if "period" in dims else 0) else [])
    state.name = action.get("title") or "Произвольный запрос"
    state.log("Построен по SQL от языковой модели")
    session.push_undo()
    session.draft = state
    return [text_block(f"Выполнил SQL (проверен: только чтение, разрешённые таблицы, лимит строк). Строк: {len(frame)}."),
            {"type": "sql", "sql": sql, "tables": [], "rows": int(len(frame))},
            table_block(session, page=0, preview=True), dashboard_block(session)], ["Покажи таблицу", "Сохрани дашборд"]


# --- таблица ----------------------------------------------------------------


def table_frame(session: Session, group_by: list[str] | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    draft = require_draft(session)
    spec = draft.spec
    frame = draft.filtered(draft.table)
    calc_ids = [c["id"] for c in draft.calcs]
    if group_by:
        grain = draft.switches.get("grain", "month")
        needed = [m for m in spec.measures if m in frame.columns and m != "pct_sum"] + calc_ids
        data = aggregate(frame, group_by, needed, draft.calcs, measures=spec.measures, grain=grain,
                         prev=draft.filtered_prev())
        cols = group_by + needed
    else:
        data = run_with_timeout(materialize, frame, draft.calcs, dims=draft.dim_columns(), measures=spec.measures,
                                prev=draft.filtered_prev())
        cols = [c for c in data.columns if c != "pct_sum"]
    return data, cols, calc_ids


def table_block(session: Session, page: int = 0, preview: bool = False, group_by: list[str] | None = None,
                sort_by: str | None = None, sort_dir: str = "desc", top_n: int | None = None,
                highlight: list[str] | None = None) -> dict:
    draft = require_draft(session)
    view = {"group_by": group_by or [], "sort_by": sort_by, "sort_dir": sort_dir, "top_n": top_n}
    session.table_view = view
    data, cols, calc_ids = table_frame(session, group_by)
    if sort_by and sort_by in data.columns:
        data = data.sort_values(sort_by, ascending=sort_dir == "asc", na_position="last")
    if top_n:
        data = data.head(top_n)
    total = len(data)
    size = 8 if preview else PAGE_SIZE
    chunk = data.iloc[page * size: page * size + size]
    grain = draft.switches.get("grain", "month") if group_by else "month"
    rows = []
    for _, row in chunk.iterrows():
        cells = []
        for col in cols:
            value = row.get(col)
            if col == "period":
                cells.append(_period_cell(value, grain))
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                unit = column_unit(col, draft.calcs)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    cells.append("—")
                elif unit == "%":
                    cells.append(fmt.pct(value))
                elif col.startswith("rank"):
                    cells.append(fmt.num(value))
                else:
                    cells.append(fmt.num(value, 0 if abs(value) >= 100 or float(value).is_integer() else 2))
            else:
                cells.append(str(value))
        rows.append(cells)
    columns = [{"key": c, "label": column_label(c, draft.calcs), "calc": c in calc_ids,
                "numeric": c in data.columns and pd.api.types.is_numeric_dtype(data[c]),
                "highlight": bool(highlight and c in highlight)} for c in cols]
    title = "Единая таблица для визуализации"
    if group_by:
        title = "Таблица, сгруппированная по: " + ", ".join(DIM_LABELS.get(g, g).lower() for g in group_by)
    return {"type": "table", "title": title, "columns": columns, "rows": rows, "total": int(total), "page": page,
            "page_size": size, "preview": preview, "group_by": group_by or [], "sort_by": sort_by, "sort_dir": sort_dir,
            "top_n": top_n, "filters": {k: v for k, v in draft.filters.items() if v},
            "calcs": [{"id": c["id"], "label": c["label"], "formula": c.get("formula", "")} for c in draft.calcs]}


def _period_cell(value, grain: str) -> str:
    try:
        from .charts import period_label

        return period_label(pd.Timestamp(value), grain)
    except Exception:  # noqa: BLE001
        return str(value)


def do_show_table(session: Session, action: dict):
    draft = require_draft(session)
    group_by = [g for g in action.get("group_by") or [] if g in draft.table.columns]
    sort_by = action.get("sort_by")
    if sort_by and sort_by not in draft.table.columns and sort_by not in [c["id"] for c in draft.calcs] and sort_by not in DERIVED:
        sort_by = None
    block = table_block(session, page=0, group_by=group_by, sort_by=sort_by, sort_dir=action.get("sort_dir", "desc"),
                        top_n=action.get("top_n"))
    note = f"Таблица: {fmt.num(block['total'])} строк"
    if draft.calcs:
        note += f", вычислений: {len(draft.calcs)} (подсвечены)"
    if draft.filters and any(draft.filters.values()):
        note += ", с учётом текущих фильтров"
    return [text_block(note + ". Колонки можно сортировать кликом, таблицу — скачать в CSV или Excel."), block], \
        ["Добавь колонку с отклонением факта от плана", "Покажи таблицу по регионам", "Выгрузи таблицу в Excel"]


# --- вычисления ---------------------------------------------------------------


def _available_columns(draft: DashboardState) -> set[str]:
    cols = set(draft.table.columns) | {c["id"] for c in draft.calcs}
    cols |= {d for d in DERIVED if all(n in draft.table.columns for n in re.findall(r"[a-z_]+", DERIVED[d]["expr"]))}
    return cols


def do_add_calc(session: Session, action: dict):
    draft = require_draft(session)
    available = _available_columns(draft)
    added: list[dict] = []
    new_state = draft.clone()
    for spec in action["calcs"]:
        params = {k: v for k, v in spec.items() if k != "kind"}
        if spec["kind"] == "expr":
            from .calc import expr_names

            missing = [n for n in expr_names(spec["expr"]) if n not in available]
            if missing:
                if any(m in ("plan", "plan_qty") for m in missing):
                    raise BotError("В этих данных нет плана — посчитать отклонение или выполнение плана не получится. "
                                   "План есть в наборе «Продажи: план и факт».")
                raise BotError(f"Не нашёл колонки: {', '.join(missing)}. Доступны: "
                               + ", ".join(column_label(c, draft.calcs) for c in sorted(available) if c != "period")[:600])
        else:
            measure = spec.get("measure") or "fact"
            if measure not in available:
                fam = draft.spec.metrics.get(draft.switches.get("metric") or draft.spec.default_metric)
                measure = fam.fact if fam else draft.spec.measures[0]
            params["measure"] = measure
            by = spec.get("by")
            if by and by not in draft.table.columns:
                raise BotError(f"В таблице нет разреза «{DIM_LABELS.get(by, by)}».")
            if spec["kind"] in ("share", "rank", "group_stat") and not by:
                params["by"] = draft.switches.get("dim") if draft.switches.get("dim") in draft.table.columns else None
        # не добавляем дубликат
        duplicate = next((c for c in new_state.calcs if c["kind"] == spec["kind"] and all(
            c.get(k) == params.get(k) for k in ("expr", "measure", "by", "lag", "mode", "stat") if k in params)), None)
        if duplicate:
            added.append({**duplicate, "_dup": True})
            continue
        calc = make_calc(spec["kind"], existing=new_state.calcs, source_text=spec.get("label", ""), **params)
        new_state.calcs.append(calc)
        available.add(calc["id"])
        added.append(calc)
    run_with_timeout(materialize, new_state.filtered(new_state.table), new_state.calcs, dims=new_state.dim_columns(),
                     measures=new_state.spec.measures, prev=new_state.filtered_prev())
    real = [c for c in added if not c.get("_dup")]
    for calc in real:
        new_state.log(f"Добавлена колонка «{calc['label']}» = {calc['formula']}")
    session.push_undo()
    session.draft = new_state
    lines = []
    for calc in added:
        if calc.get("_dup"):
            lines.append(f"Колонка «{esc(calc['label'])}» уже есть в таблице.")
        else:
            lines.append(f"Добавил колонку <b>«{esc(calc['label'])}»</b> = {esc(calc['formula'])}.")
    blocks = [text_block(" ".join(lines) + " Графики пока не менял — скажите, какой перестроить, или «перестрой график с учётом новой колонки».")]
    highlight = [c["id"] for c in added]
    group = [draft.switches["dim"]] if draft.switches.get("dim") in draft.table.columns and draft.dataset != "custom" else []
    blocks.append(table_block(session, page=0, preview=True, group_by=group, highlight=highlight,
                              sort_by=highlight[-1] if group and highlight else None))
    blocks[-1]["title"] = f"Итоги по {semantic_dative(group[0]) if group else ''} с новыми колонками (полная таблица — «покажи таблицу»)" if group else blocks[-1]["title"]
    blocks.append(dashboard_update_block(session))
    last = added[-1]
    tips = [_rebuild_hint(last), "Покажи таблицу", "Отмени последнее вычисление"]
    return blocks, [t for t in tips if t]


def _rebuild_hint(calc: dict) -> str:
    kind = calc["kind"]
    if kind == "change":
        return "Покажи четвёртый график с учётом динамики"
    if kind == "expr" and "выполн" in calc["label"].lower():
        return "Перестрой третий график — отсортируй по проценту выполнения"
    if kind == "cumsum":
        return "Перестрой первый график с учётом новой колонки"
    return "Перестрой график с учётом новой колонки"


def _find_calc(draft: DashboardState, ref: str | None) -> dict | None:
    if not draft.calcs:
        return None
    if not ref or ref == "last":
        return draft.calcs[-1]
    for calc in draft.calcs:
        if calc["id"] == ref:
            return calc
    r = rules.norm(ref)
    for calc in reversed(draft.calcs):
        if rules.norm(calc["label"]) == r:
            return calc
    stems = [rules._stem_word(w) for w in re.findall(r"[а-яa-z]+", r) if len(w) > 2]
    best, best_score = None, 0
    for calc in draft.calcs:
        label = rules.norm(calc["label"] + " " + calc.get("formula", ""))
        score = sum(1 for s in stems if s in label)
        if score > best_score:
            best, best_score = calc, score
    return best


def do_remove_calc(session: Session, action: dict):
    draft = require_draft(session)
    calc = _find_calc(draft, action.get("ref"))
    if calc is None:
        raise BotError("Вычислений пока нет — удалять нечего." if not draft.calcs else f"Не нашёл колонку «{esc(action.get('ref'))}».")
    new_state = draft.clone()
    dependents = [c["label"] for c in new_state.calcs if c["id"] != calc["id"] and calc["id"] in _deps(c)]
    if dependents:
        raise BotError(f"Колонку «{esc(calc['label'])}» используют другие вычисления: {', '.join(dependents)}. Сначала удалите их.")
    new_state.calcs = [c for c in new_state.calcs if c["id"] != calc["id"]]
    touched = []
    for i, chart in enumerate(new_state.charts, 1):
        for key in ("value", "sort_by"):
            if chart.get(key) == calc["id"]:
                chart.pop(key, None)
                touched.append(i)
    new_state.log(f"Удалена колонка «{calc['label']}»")
    session.push_undo()
    session.draft = new_state
    text = f"Убрал колонку «{esc(calc['label'])}»."
    if touched:
        text += f" Графики {', '.join(map(str, sorted(set(touched))))} вернул к исходному виду."
        return [text_block(text), dashboard_block(session)], ["Отмени"]
    return [text_block(text), dashboard_update_block(session)], ["Отмени", "Покажи таблицу"]


def _deps(calc: dict) -> list[str]:
    from .calc import calc_dependencies

    try:
        return calc_dependencies(calc)
    except CalcError:
        return []


def do_modify_calc(session: Session, action: dict):
    draft = require_draft(session)
    calc = _find_calc(draft, action.get("ref"))
    if calc is None:
        raise BotError("Не нашёл вычисление, которое нужно изменить.")
    new_spec = action["calc"]
    new_state = draft.clone()
    params = {k: v for k, v in new_spec.items() if k != "kind"}
    if new_spec["kind"] == "expr":
        params.setdefault("label", calc["label"])
        params.setdefault("unit", calc.get("unit", ""))
        if params["label"].startswith("Расчёт:"):
            params["label"] = calc["label"]
    others = [c for c in new_state.calcs if c["id"] != calc["id"]]
    rebuilt = make_calc(new_spec["kind"], existing=others, **params)
    rebuilt["id"] = calc["id"]
    new_state.calcs = [rebuilt if c["id"] == calc["id"] else c for c in new_state.calcs]
    run_with_timeout(materialize, new_state.table, new_state.calcs, dims=new_state.dim_columns(),
                     measures=new_state.spec.measures, prev=new_state.prev)
    new_state.log(f"Изменена формула «{rebuilt['label']}»: {rebuilt['formula']}")
    session.push_undo()
    session.draft = new_state
    return [text_block(f"Изменил формулу «{esc(rebuilt['label'])}»: теперь {esc(rebuilt['formula'])}. Графики и таблица пересчитаны."),
            dashboard_block(session)], ["Покажи таблицу", "Отмени"]


# --- графики ------------------------------------------------------------------


def _target_chart(session: Session, draft: DashboardState, ref: Any, changes: dict) -> list[int]:
    n = len(draft.charts)
    if ref == "all":
        return list(range(n))
    if isinstance(ref, int):
        if not 1 <= ref <= n:
            raise BotError(f"На дашборде {n} {fmt.plural(n, 'график', 'графика', 'графиков')}, графика №{ref} нет.")
        return [ref - 1]
    if ref == "last" and session.last_chart:
        return [session.last_chart - 1]
    types = [c["type"] for c in draft.charts]
    value = changes.get("value") or (draft.calcs[-1]["id"] if changes.get("use_last_calc") and draft.calcs else None)
    calc = next((c for c in draft.calcs if c["id"] == value), None) if value else None

    def first(*wanted: str) -> int | None:
        return next((i for i, t in enumerate(types) if t in wanted), None)

    idx = None
    if calc is not None:
        if calc["kind"] == "change":
            idx = first("exec_line", "trend", "planfact_period")
        elif calc["kind"] == "cumsum":
            idx = first("planfact_period", "trend")
        elif calc.get("unit") == "%" or calc["kind"] in ("rank", "share", "group_stat") or calc["kind"] == "expr":
            idx = first("top", "bar_dim", "planfact_dim")
    elif changes.get("sort_by") or changes.get("top_n"):
        idx = first("top", "bar_dim", "planfact_dim")
    elif changes.get("variant") == "heatmap":
        idx = first("exec_line")
    if idx is None:
        idx = (session.last_chart - 1) if session.last_chart and session.last_chart <= n else 0
    return [idx]


def _apply_chart_changes(draft: DashboardState, chart: dict, changes: dict) -> list[str]:
    notes: list[str] = []
    available = _available_columns(draft)
    changes = dict(changes)
    for key in ("value", "sort_by"):
        ref = changes.get(key)
        if isinstance(ref, str) and ref.startswith("__pending_"):
            try:
                changes[key] = draft.calcs[int(ref.rsplit("_", 1)[1])]["id"]
            except (IndexError, ValueError):
                changes.pop(key)
    if changes.pop("use_last_calc", False) and draft.calcs and "value" not in changes:
        changes["value"] = draft.calcs[-1]["id"]
    for key in ("value", "sort_by"):
        if key in changes and changes[key] and changes[key] not in available:
            if changes[key] in DERIVED and draft.dataset == "sales":
                continue
            raise BotError(f"Показателя «{esc(column_label(changes[key], draft.calcs))}» в таблице нет. "
                           "Сначала добавьте такую колонку, например: «Посчитай процент выполнения плана».")
    if "type" in changes and changes["type"] != chart["type"]:
        chart["type"] = changes["type"]
        chart.pop("variant", None)
        notes.append(f"тип — {CHART_TYPE_LABELS.get(chart['type'], chart['type']).lower()}")
    if "variant" in changes:
        chart["variant"] = changes["variant"]
        if changes["variant"] == "heatmap":
            notes.append("вид — тепловая карта")
    if changes.get("value"):
        value = changes["value"]
        calc = next((c for c in draft.calcs if c["id"] == value), None)
        if calc and calc["kind"] in ("cumsum", "change") and calc.get("lag") != "year" \
                and chart["type"] not in ("planfact_period", "trend", "exec_line"):
            chart["type"] = "planfact_period" if draft.dataset == "sales" else "trend"
            chart.pop("sort_by", None)
            notes.append("тип — по периодам (этот расчёт идёт во времени)")
        if chart["type"] in ("top", "bar_dim", "planfact_dim") and calc and calc["kind"] in ("rank",):
            chart["sort_by"] = value
            chart["sort_dir"] = "asc"
            notes.append(f"сортировка по «{column_label(value, draft.calcs)}»")
        elif chart["type"] == "structure" and calc and calc.get("unit") == "%":
            chart["type"] = "top"
            chart["value"] = value
            chart["sort_by"] = value
            notes.append(f"показывает «{column_label(value, draft.calcs)}» (ТОП вместо круговой — доли в процентах)")
        elif chart["type"] == "planfact_dim":
            chart["type"] = "top"
            chart["value"] = value
            chart["sort_by"] = value
            notes.append(f"показывает «{column_label(value, draft.calcs)}»")
        else:
            chart["value"] = value
            if chart["type"] in ("top", "bar_dim"):
                chart["sort_by"] = value
            notes.append(f"показывает «{column_label(value, draft.calcs)}»")
    if changes.get("sort_by"):
        chart["sort_by"] = changes["sort_by"]
        if chart["type"] in ("top", "bar_dim") and column_unit(changes["sort_by"], draft.calcs) == "%" and not chart.get("value"):
            chart["value"] = changes["sort_by"]
        if chart["type"] not in ("top", "bar_dim", "planfact_dim"):
            chart["type"] = "top"
            notes.append("тип — ТОП (сортировка возможна только у рейтинга)")
        notes.append(f"сортировка по «{column_label(changes['sort_by'], draft.calcs)}» "
                     f"({'по возрастанию' if changes.get('sort_dir') == 'asc' else 'по убыванию'})")
    if "sort_dir" in changes:
        chart["sort_dir"] = changes["sort_dir"]
    if changes.get("top_n"):
        chart["top_n"] = changes["top_n"]
        if chart["type"] not in ("top", "bar_dim", "planfact_dim", "structure"):
            chart["type"] = "top"
        notes.append(f"ТОП-{changes['top_n']}")
    if changes.get("dim"):
        chart["dim"] = changes["dim"]
        notes.append(f"разрез — {DIM_LABELS_PLURAL.get(changes['dim'], changes['dim']).lower()}")
    for key, label in (("grain", "период"), ("view", "вид"), ("metric", "метрика"), ("mode", "вид"), ("split", "разбивка")):
        if changes.get(key):
            chart[key] = changes[key]
            value_label = SWITCH_VALUE_LABELS.get(changes[key]) or DIM_LABELS_PLURAL.get(changes[key], "").lower() or \
                {"bar": "столбцы", "line": "линия", "amount": "объём", "qty": "количество", "discount": "скидка", "deals": "сделки"}.get(changes[key], changes[key])
            notes.append(f"{label} — {value_label}")
    if changes.get("title"):
        chart["title"] = changes["title"]
    else:
        chart.pop("title", None)
    return notes


def do_rebuild_chart(session: Session, action: dict):
    draft = require_draft(session)
    changes = action.get("changes") or {}
    if not changes:
        raise BotError("Что изменить в графике? Например: «отсортируй по проценту выполнения», «покажи ТОП-10», «сделай тепловой картой».")
    new_state = draft.clone()
    targets = _target_chart(session, new_state, action.get("chart"), changes)
    all_notes = []
    for idx in targets:
        notes = _apply_chart_changes(new_state, new_state.charts[idx], changes)
        all_notes.append(f"график {idx + 1}: {', '.join(notes) if notes else 'перестроен'}")
        new_state.log(f"Перестроен график {idx + 1}: {', '.join(notes) if notes else 'обновлён'}")
    session.push_undo()
    session.draft = new_state
    session.last_chart = targets[-1] + 1
    return [text_block("Перестроил " + "; ".join(all_notes) + "."), dashboard_block(session)], \
        ["Отмени", "Сохрани дашборд", "Покажи таблицу"]


def do_add_chart(session: Session, action: dict):
    draft = require_draft(session)
    if len(draft.charts) >= MAX_CHARTS:
        raise BotError(f"На дашборде уже {len(draft.charts)} графиков — больше не добавляю, чтобы он оставался читаемым. Уберите лишний: «убери второй график».")
    changes = dict(action.get("changes") or {})
    new_state = draft.clone()
    chart_type = changes.pop("type", None) or "top"
    if draft.dataset != "sales" and chart_type in ("planfact_period", "exec_line", "scatter_planfact", "planfact_dim", "top"):
        chart_type = "bar_dim" if chart_type in ("top", "planfact_dim") else "trend"
    chart = {"id": uuid.uuid4().hex[:8], "type": chart_type, "dim": changes.pop("dim", "@switch")}
    new_state.charts.append(chart)
    notes = _apply_chart_changes(new_state, chart, changes)
    new_state.log(f"Добавлен график {len(new_state.charts)}: {CHART_TYPE_LABELS.get(chart['type'], chart['type'])}")
    session.push_undo()
    session.draft = new_state
    session.last_chart = len(new_state.charts)
    return [text_block(f"Добавил график {len(new_state.charts)}: {CHART_TYPE_LABELS.get(chart['type'], chart['type']).lower()}"
                       + (f" ({', '.join(notes)})" if notes else "") + "."), dashboard_block(session)], ["Отмени"]


def do_remove_chart(session: Session, action: dict):
    draft = require_draft(session)
    if len(draft.charts) <= 1:
        raise BotError("Это последний график — убрать его нельзя.")
    idx = _target_chart(session, draft, action.get("chart"), {})[0]
    new_state = draft.clone()
    removed = new_state.charts.pop(idx)
    new_state.log(f"Убран график {idx + 1}")
    session.push_undo()
    session.draft = new_state
    session.last_chart = None
    return [text_block(f"Убрал график {idx + 1} ({CHART_TYPE_LABELS.get(removed['type'], removed['type']).lower()})."),
            dashboard_block(session)], ["Отмени"]


# --- фильтры и переключатели ----------------------------------------------------


def apply_filter(state: DashboardState, field: str | None, values: list[str], mode: str, period: dict | None = None) -> str:
    table = state.table
    if mode == "clear_all":
        state.filters = {}
        return "Сбросил все фильтры"
    if field == "period" and period:
        p = resolve_period(period)
        if p is None:
            raise BotError("Не понял, какой период оставить.")
        months = pd.date_range(p.start, p.end, freq="MS").strftime("%Y-%m").tolist()
        present = set(table["period"].dt.strftime("%Y-%m"))
        values = [m for m in months if m in present]
        if not values:
            raise BotError(f"Период «{esc(p.label)}» не входит в данные дашборда ({state.period.range_label()}). "
                           f"Постройте новый запрос за нужный период.")
    if field in ("city", "outlet_type") and "outlet" in table.columns and field in table.columns:
        outlets = sorted(table.loc[table[field].isin(values), "outlet"].unique())
        prefix = f"{'Город' if field == 'city' else 'Тип точки'}: {', '.join(esc(v) for v in values)} → "
        scope_label = f"{'город' if field == 'city' else 'тип точки'} {', '.join(esc(v) for v in values)}"
        if not outlets:
            raise BotError(f"В данных дашборда нет точек: {', '.join(esc(v) for v in values)}.")
        if mode == "exclude":
            present_outlets = set(state.filters.get("outlet") or table["outlet"].unique())
            state.filters["outlet"] = sorted(present_outlets - set(outlets))
            return f"{prefix}исключил {len(outlets)} {fmt.plural(len(outlets), 'точку', 'точки', 'точек')}"
        field, values = "outlet", outlets
    else:
        prefix = scope_label = ""
    if field not in table.columns:
        raise BotError(f"В этом дашборде нет разреза «{DIM_LABELS.get(field or '', field)}».")
    if mode == "clear":
        state.filters.pop(field, None)
        return f"Сбросил фильтр «{DIM_LABELS.get(field, field)}»"
    present = set(table[field].astype(str).unique()) if field != "period" else set(table["period"].dt.strftime("%Y-%m"))
    found = [v for v in values if v in present]
    missing = [v for v in values if v not in present]
    if not found:
        raise BotError(f"Значений {', '.join(esc(v) for v in values[:5])} нет в данных дашборда.")
    current = set(state.filters.get(field, []))
    if mode == "add":
        new = sorted(current | set(found))
    elif mode == "exclude":
        base = current or present
        new = sorted(base - set(found))
    else:
        new = found
    state.filters[field] = new
    shown = ", ".join(esc(_period_cell(pd.Timestamp(v + "-01"), "month") if field == "period" else v) for v in new[:6])
    extra = f" и ещё {len(new) - 6}" if len(new) > 6 else ""
    verb = {"exclude": "Исключил", "add": "Добавил в фильтр"}.get(mode, "Фильтр")
    text = f"{prefix}{verb}: {DIM_LABELS.get(field, field)} = {shown}{extra}"
    if prefix:
        text = f"Фильтр: {scope_label} — {len(new)} {fmt.plural(len(new), 'точка', 'точки', 'точек')}"
    if mode == "exclude":
        text = f"{prefix}Исключил {', '.join(esc(v) for v in found[:5])} — осталось {len(new)} знач."
    if missing:
        text += f" (не нашёл: {', '.join(esc(v) for v in missing[:3])})"
    # каскад: при выборе региона убираем точки из других регионов
    if field == "region" and state.filters.get("outlet"):
        allowed = set(table.loc[table["region"].isin(new), "outlet"])
        state.filters["outlet"] = [o for o in state.filters["outlet"] if o in allowed]
    return text


def do_set_filter(session: Session, action: dict):
    draft = require_draft(session)
    new_state = draft.clone()
    text = apply_filter(new_state, action.get("field"), action.get("values") or [], action.get("mode", "only"), action.get("period"))
    new_state.log(text)
    session.push_undo()
    session.draft = new_state
    return [text_block(text + ". Все графики и KPI пересчитаны."), dashboard_block(session)], \
        ["Сбрось фильтры", "Переключи разрез на точки", "Сохрани дашборд"]


SWITCH_VALUE_LABELS = {"month": "месяц", "quarter": "квартал", "year": "год", "abs": "абсолютные значения",
                       "pct": "проценты", "dyn": "динамика", "both": "план и факт", "plan": "только план", "fact": "только факт"}


def do_set_switch(session: Session, action: dict):
    draft = require_draft(session)
    new_state = draft.clone()
    notes = []
    spec = new_state.spec
    if action.get("dim"):
        if action["dim"] not in new_state.table.columns:
            raise BotError(f"Разреза «{DIM_LABELS.get(action['dim'], action['dim'])}» в этих данных нет.")
        new_state.switches["dim"] = action["dim"]
        notes.append(f"разрез — {DIM_LABELS_PLURAL.get(action['dim'], action['dim']).lower()}")
    if action.get("metric"):
        metric = action["metric"]
        if metric not in spec.metrics:
            if metric == "discount" and new_state.dataset == "discounts":
                metric = "discount"
            else:
                raise BotError(f"Метрики «{metric}» в этом наборе нет. Доступны: {', '.join(m.label for m in spec.metrics.values())}.")
        new_state.switches["metric"] = metric
        notes.append(f"метрика — {spec.metrics[metric].label.lower()}")
    for key in ("grain", "view", "series"):
        if action.get(key):
            new_state.switches[key] = action[key]
            notes.append(f"{ {'grain': 'период', 'view': 'вид', 'series': 'ряды'}[key] } — {SWITCH_VALUE_LABELS.get(action[key], action[key])}")
    if new_state.switches == draft.switches:
        return [text_block("Так уже и есть: " + ", ".join(notes) + ". Другие варианты — на переключателях над графиками.")], []
    new_state.log("Переключено: " + ", ".join(notes))
    session.push_undo()
    session.draft = new_state
    return [text_block("Переключил: " + ", ".join(notes) + ". Графики перестроены."), dashboard_block(session)], \
        ["Отмени", "Сохрани дашборд"]


def update_draft_ui(session_id: str, uid: str | None, filters: dict | None, switches: dict | None) -> dict:
    """Изменение фильтров/переключателей из интерфейса (без сообщения в чат)."""
    session = get_session(session_id)
    with session.lock:
        draft = require_draft(session)
        if uid and uid != draft.uid:
            raise BotError("Этот дашборд устарел — работайте с последним в чате.")
        new_state = draft.clone()
        notes = []
        if filters is not None:
            clean = {}
            for key, values in filters.items():
                if key in new_state.table.columns and isinstance(values, list):
                    vals = [str(v)[:300] for v in values][:500]
                    if vals:
                        clean[key] = vals
            if clean != {k: v for k, v in new_state.filters.items() if v}:
                new_state.filters = clean
                notes.append("фильтры: " + ("; ".join(f"{DIM_LABELS.get(k, k)} ({len(v)})" for k, v in clean.items()) or "сброшены"))
        if switches is not None:
            allowed = {"dim": set(new_state.table.columns), "metric": set(new_state.spec.metrics),
                       "grain": {"month", "quarter", "year"}, "view": {"abs", "pct", "dyn"}, "series": {"both", "plan", "fact"}}
            for key, value in switches.items():
                if key in allowed and value in allowed[key] and new_state.switches.get(key) != value:
                    new_state.switches[key] = value
                    notes.append(f"{SWITCH_LABELS_RU.get(key, key)} — {SWITCH_VALUE_LABELS.get(value, DIM_LABELS_PLURAL.get(value, value)).lower()}")
        if notes:
            new_state.log("Интерфейс: " + ", ".join(notes))
            session.push_undo()
            session.draft = new_state
        payload = render(session.draft)
        session.persist()
        return payload


SWITCH_LABELS_RU = {"dim": "разрез", "metric": "метрика", "grain": "период", "view": "вид", "series": "ряды"}


# --- сохранение и список ---------------------------------------------------------


def _frames(state: DashboardState) -> tuple:
    return (state.table, state.prev, state.links)


def _describe(state: DashboardState) -> str:
    parts = [state.spec.label, state.period.range_label()]
    if state.calcs:
        parts.append("вычисления: " + ", ".join(c["label"] for c in state.calcs[:4]))
    return " · ".join(parts)


def do_save(session: Session, action: dict):
    draft = require_draft(session)
    name = (action.get("name") or "").strip()
    linked = storage.get_info(draft.saved_id) if draft.saved_id else None
    if linked and linked.get("deleted"):
        linked = None
        draft.saved_id = None
    save_as_new = action.get("as_new") or (linked is not None and name and rules.norm(name) != rules.norm(linked["name"]))
    if linked and not save_as_new:
        if name:
            draft.name = name
        draft.log(f"Сохранено (версия {linked['version'] + 1})")
        info = storage.save_version(linked["id"], draft.meta(), _frames(draft), name=draft.name, description=_describe(draft),
                                    note="Сохранены изменения")
        draft.saved_version = info["version"]
        return [text_block(f"Обновил дашборд <b>«{esc(info['name'])}»</b> — версия {info['version']}. "
                           "Прошлые версии доступны в истории («история версий»)."), dashboard_update_block(session)], \
            ["Покажи мои дашборды", "История версий", "Скачай дашборд в HTML"]
    target_name = name or (f"{draft.name} (копия)" if linked else draft.name) or "Дашборд"
    existing = storage.find_by_name(target_name)
    if existing and not action.get("force"):
        alt = storage.unique_name(target_name)
        options = [
            {"label": f"Перезаписать «{target_name}»", "action": {"type": "save", "name": target_name, "force": True, "as_new": bool(action.get("as_new"))}},
            {"label": f"Сохранить как «{alt}»", "action": {"type": "save", "name": alt, "as_new": True}},
        ]
        session.pending = {"options": options}
        return [{"type": "clarify", "question": f"Дашборд «{esc(target_name)}» уже есть. Перезаписать его или сохранить под другим именем?",
                 "options": options}], []
    new_state = draft.clone() if linked else draft
    new_state.name = target_name
    if existing and action.get("force"):
        new_state.log(f"Сохранено поверх «{target_name}»")
        info = storage.save_version(existing["id"], new_state.meta(), _frames(new_state), name=target_name,
                                    description=_describe(new_state), note="Перезаписан")
    else:
        new_state.log(f"Сохранено как «{target_name}»")
        info = storage.save_new(target_name, new_state.meta(), _frames(new_state), description=_describe(new_state))
    new_state.saved_id = info["id"]
    new_state.saved_version = info["version"]
    if linked:
        session.push_undo()
        session.draft = new_state
    text = (f"Сохранил дашборд <b>«{esc(info['name'])}»</b>. Сохранены запрос, единая таблица с вычислениями, графики, "
            "фильтры, переключатели и KPI. Вернуться к нему: «Открой дашборд «" + esc(info["name"]) + "»» или «Покажи мои дашборды».")
    if linked:
        text = f"Сохранил как новый дашборд <b>«{esc(info['name'])}»</b>. Оригинал «{esc(linked['name'])}» не изменён."
    return [text_block(text), dashboard_update_block(session)], ["Покажи мои дашборды", "Скачай дашборд в HTML", "Поделись ссылкой"]


def list_block(query: str = "", sort: str = "updated") -> dict:
    items = storage.list_dashboards(query, sort)
    return {"type": "dashboards", "items": items, "query": query, "sort": sort}


def do_list(session: Session, action: dict):
    query = action.get("query") or ""
    block = list_block(query, action.get("sort") or "updated")
    n = len(block["items"])
    if query:
        if not n:
            return [text_block(f"Не нашёл дашбордов по запросу «{esc(query)}». Все дашборды — «Покажи мои дашборды».")], ["Покажи мои дашборды"]
        text = f"Нашёл {n} {fmt.plural(n, 'дашборд', 'дашборда', 'дашбордов')} по запросу «{esc(query)}» (ищу по названию и тексту запроса):"
    elif not n:
        return [text_block("Сохранённых дашбордов пока нет. Постройте дашборд и скажите «Сохрани дашборд».")], starter_suggestions()[:2]
    else:
        text = f"Ваши дашборды ({n}). Можно открыть, переименовать, скопировать или удалить; список сортируется и ищется."
    return [text_block(text), block], []


def _resolve_saved(session: Session, name: str | None, *, include_deleted: bool = False) -> dict | list[dict] | None:
    if not name:
        if session.draft and session.draft.saved_id:
            return storage.get_info(session.draft.saved_id)
        return None
    exact = storage.find_by_name(name, include_deleted=include_deleted)
    if exact:
        return exact
    found = storage.list_dashboards(name)
    if len(found) == 1:
        return found[0]
    return found or None


def do_open(session: Session, action: dict):
    target = storage.get_info(action["id"]) if action.get("id") else _resolve_saved(session, action.get("name"))
    if target is None:
        name = action.get("name") or ""
        stems = [storage._stem(w) for w in re.findall(r"[\wё-]+", name.lower()) if len(w) > 2]
        deleted = [d for d in storage.deleted_dashboards() if stems and all(s in storage._norm(d["name"]) for s in stems)]
        if deleted:
            options = [{"label": f"Восстановить и открыть «{d['name']}»", "action": {"type": "open", "id": d["id"]}} for d in deleted[:3]]
            session.pending = {"options": options}
            return [{"type": "clarify", "question": f"Дашборд «{esc(deleted[0]['name'])}» был удалён. Восстановить его?", "options": options}], []
        return [text_block(f"Не нашёл дашборд «{esc(name)}». Список — «Покажи мои дашборды».")], ["Покажи мои дашборды"]
    if isinstance(target, list):
        options = [{"label": item["name"], "action": {"type": "open", "id": item["id"]}} for item in target[:6]]
        session.pending = {"options": options}
        return [{"type": "clarify", "question": "Нашёл несколько подходящих дашбордов. Какой открыть?", "options": options}], []
    if target.get("deleted"):
        storage.restore(target["id"])
    info, meta, frames = storage.load(target["id"])
    state = DashboardState.from_meta(meta, *frames)
    state.saved_id = info["id"]
    state.saved_version = info["loaded_version"]
    state.name = info["name"]
    session.push_undo()
    session.draft = state
    session.last_chart = None
    saved_at = info["version_created_at"].replace("T", " ")[:16]
    text = (f"Открыл дашборд <b>«{esc(info['name'])}»</b> (версия {info['loaded_version']} от {saved_at}). "
            "Восстановлены таблица с вычислениями, графики, фильтры, переключатели и KPI. "
            "Можно продолжать работу; «Сохрани» обновит этот дашборд, «Сохрани как новый» — создаст копию.")
    blocks = [text_block(text)]
    if state.calcs:
        blocks.append(table_block(session, page=0, preview=True))
    blocks.append(dashboard_block(session))
    return blocks, ["Добавь колонку с отклонением факта от плана", "История версий", "Обнови данные", "Сделай копию дашборда"]


def do_rename(session: Session, action: dict):
    new_name = (action.get("new_name") or "").strip()
    if not new_name:
        session.pending = {"slot": "new_name", "action": {"type": "rename", "name": action.get("name")}}
        return [{"type": "clarify", "question": "Какое название дать дашборду?", "options": []}], []
    if action.get("id"):
        target = storage.get_info(action["id"])
    else:
        target = _resolve_saved(session, action.get("name"))
    if target is None and session.draft and not action.get("name"):
        session.draft.name = new_name
        session.draft.log(f"Переименован в «{new_name}»")
        return [text_block(f"Назвал текущий дашборд «{esc(new_name)}». Он ещё не сохранён — скажите «Сохрани»."),
                dashboard_update_block(session)], ["Сохрани дашборд"]
    if target is None or isinstance(target, list):
        return [text_block("Не понял, какой дашборд переименовать. Откройте его или укажите название в кавычках.")], ["Покажи мои дашборды"]
    clash = storage.find_by_name(new_name)
    if clash and clash["id"] != target["id"]:
        raise BotError(f"Дашборд с названием «{esc(new_name)}» уже есть — выберите другое название.")
    storage.rename(target["id"], new_name)
    blocks = [text_block(f"Переименовал «{esc(target['name'])}» → <b>«{esc(new_name)}»</b>.")]
    if session.draft and session.draft.saved_id == target["id"]:
        session.draft.name = new_name
        session.draft.log(f"Переименован в «{new_name}»")
        blocks.append(dashboard_update_block(session))
    return blocks, ["Покажи мои дашборды"]


def do_delete(session: Session, action: dict):
    target = storage.get_info(action["id"]) if action.get("id") else _resolve_saved(session, action.get("name"))
    if target is None:
        return [text_block(f"Не нашёл дашборд «{esc(action.get('name') or '')}» — удалять нечего.")], ["Покажи мои дашборды"]
    if isinstance(target, list):
        options = [{"label": f"Удалить «{item['name']}»", "action": {"type": "delete", "id": item["id"]}} for item in target[:6]]
        session.pending = {"options": options}
        return [{"type": "clarify", "question": "Какой из дашбордов удалить?", "options": options}], []
    storage.delete(target["id"])
    session.last_deleted = target["id"]
    if session.draft and session.draft.saved_id == target["id"]:
        session.draft.saved_id = None
        session.draft.saved_version = None
    options = [{"label": "Восстановить", "action": {"type": "restore", "id": target["id"]}}]
    return [text_block(f"Удалил дашборд «{esc(target['name'])}»."),
            {"type": "clarify", "question": "Передумали? Его можно вернуть:", "options": options, "soft": True}], ["Покажи мои дашборды"]


def do_restore(session: Session, action: dict):
    target_id = action.get("id") or session.last_deleted
    if action.get("name"):
        found = storage.find_by_name(action["name"], include_deleted=True)
        target_id = found["id"] if found else None
    if not target_id and not action.get("name"):
        deleted = storage.deleted_dashboards()
        target_id = deleted[0]["id"] if deleted else None
    if not target_id:
        raise BotError("Удалённых дашбордов нет — восстанавливать нечего.")
    storage.restore(target_id)
    info = storage.get_info(target_id)
    return [text_block(f"Восстановил дашборд «{esc(info['name'])}»."), list_block()], []


def do_copy(session: Session, action: dict):
    if action.get("name") or action.get("id"):
        target = storage.get_info(action["id"]) if action.get("id") else _resolve_saved(session, action.get("name"))
        if not target or isinstance(target, list):
            raise BotError("Не нашёл дашборд для копирования.")
        new_name = action.get("new_name") or storage.unique_name(f"{target['name']} (копия)")
        info = storage.copy_dashboard(target["id"], new_name)
        return [text_block(f"Сделал копию: <b>«{esc(info['name'])}»</b>. Оригинал «{esc(target['name'])}» не тронут."),
                list_block()], [f"Открой дашборд «{info['name']}»"]
    draft = require_draft(session)
    base = draft.name or "Дашборд"
    new_name = action.get("new_name") or storage.unique_name(f"{base} (копия)")
    new_state = draft.clone()
    new_state.name = new_name
    original = draft.saved_id and storage.get_info(draft.saved_id)
    new_state.log(f"Копия дашборда «{base}»")
    info = storage.save_new(new_name, new_state.meta(), _frames(new_state), description=_describe(new_state),
                            note=f"Копия «{base}»")
    new_state.saved_id = info["id"]
    new_state.saved_version = info["version"]
    session.push_undo()
    session.draft = new_state
    text = f"Сделал копию <b>«{esc(new_name)}»</b> и переключился на неё — экспериментируйте"
    text += f", оригинал «{esc(original['name'])}» не изменится." if original else "."
    return [text_block(text), dashboard_update_block(session)], ["Покажи мои дашборды"]


def do_history(session: Session, action: dict):
    draft = require_draft(session)
    versions = storage.versions(draft.saved_id) if draft.saved_id else []
    return [{"type": "history", "items": draft.history[-30:], "versions": versions, "saved": bool(draft.saved_id),
             "name": draft.name, "calcs": [{"id": c["id"], "label": c["label"], "formula": c.get("formula", "")} for c in draft.calcs]}], \
        (["Откати к версии 1"] if len(versions) > 1 else []) + ["Отмени"]


def do_revert(session: Session, action: dict):
    draft = require_draft(session)
    if not draft.saved_id:
        raise BotError("История версий есть только у сохранённых дашбордов. Для отмены последнего шага скажите «Отмени».")
    version = action.get("version")
    info, meta, frames = storage.load(draft.saved_id, version)
    state = DashboardState.from_meta(meta, *frames)
    state.saved_id = info["id"]
    state.saved_version = info["version"]
    state.name = info["name"]
    state.log(f"Возврат к версии {version}")
    session.push_undo()
    session.draft = state
    return [text_block(f"Вернул дашборд «{esc(info['name'])}» к версии {version}. Чтобы закрепить, скажите «Сохрани» — появится новая версия, старые останутся."),
            dashboard_block(session)], ["Сохрани дашборд", "История версий"]


def do_undo(session: Session, action: dict):
    if not session.undo:
        raise BotError("Отменять нечего — это исходное состояние.")
    previous = session.undo.pop()
    session.draft = previous
    if previous is None:
        return [text_block("Отменил. Дашборда сейчас нет.")], starter_suggestions()[:3]
    return [text_block("Отменил последнее действие."), dashboard_block(session)], ["Отмени", "История версий"]


def do_refresh(session: Session, action: dict):
    draft = require_draft(session)
    if draft.dataset == "custom":
        frame, _ = semantic.fetch_custom(draft.sql)
        fetched_table, prev, links = frame, None, None
    else:
        fetched = semantic.fetch_dataset(draft.dataset, draft.period)
        fetched_table, prev, links = fetched.table, fetched.prev, fetched.links
    new_state = draft.clone()
    before = len(draft.table)
    new_state.table, new_state.prev, new_state.links = fetched_table, prev, links
    new_state.log("Данные обновлены из базы")
    session.push_undo()
    session.draft = new_state
    return [text_block(f"Перезагрузил данные из базы: было {fmt.num(before)} строк, стало {fmt.num(len(fetched_table))}. "
                       "Вычисления, графики и фильтры применены к свежим данным."), dashboard_block(session)], ["Сохрани дашборд"]


# --- экспорт и прочее -----------------------------------------------------------


def do_export(session: Session, action: dict):
    draft = require_draft(session)
    fmt_ = action.get("format", "html")
    pdf_note = ""
    if fmt_ == "pdf":
        fmt_ = "html"
        pdf_note = " PDF соберётся из HTML: откройте файл и нажмите Ctrl+P → «Сохранить как PDF»."
    if fmt_ == "png":
        return [text_block("Собираю картинку дашборда прямо в браузере…"), {"type": "download", "format": "png", "client_side": True,
                                                                             "uid": draft.uid, "filename": _file_stem(draft) + ".png"}], []
    labels = {"html": "интерактивный HTML (графики с зумом и подсказками, KPI, резюме, таблица)",
              "csv": "CSV (разделитель «;», открывается в Excel)", "xlsx": "Excel: данные с вычислениями, формулы, параметры"}
    return [text_block(f"Готово: {labels[fmt_]}. Выгрузка учитывает текущие фильтры.{pdf_note}"),
            {"type": "download", "format": fmt_, "url": f"/api/export?session_id={session.id}&format={fmt_}&uid={draft.uid}",
             "filename": _file_stem(draft) + "." + fmt_}], []


def _file_stem(draft: DashboardState) -> str:
    name = re.sub(r"[^\wа-яА-ЯёЁ.-]+", "_", draft.name or "dashboard").strip("_")
    return name[:60] or "dashboard"


def do_share(session: Session, action: dict):
    draft = require_draft(session)
    if not draft.saved_id:
        options = [{"label": "Сохранить и поделиться", "action": {"type": "save", "name": draft.name}}]
        session.pending = {"options": options}
        return [{"type": "clarify", "question": "Ссылкой можно поделиться только сохранённым дашбордом. Сохранить его?",
                 "options": options}], []
    info = storage.get_info(draft.saved_id)
    return [text_block(f"Ссылка на дашборд «{esc(info['name'])}» (только просмотр, с живыми фильтрами; показывает последнюю сохранённую версию):"),
            {"type": "share", "url": f"/share/{info['share_token']}", "name": info["name"]}], []


HELP_TEXT = """Я строю аналитику прямо в чате. Что умею:
<ul>
<li><b>Данные и графики:</b> «Собери дашборд по продажам за последний год», «Покажи план-факт по регионам за последний квартал», «Сравни точки за год, выдели отстающих», «Структура продаж по продуктам за последний месяц», «Какие скидки делались клиенту … и почему».</li>
<li><b>Таблица и вычисления:</b> «Покажи таблицу», «Добавь колонку с отклонением факта от плана», «Посчитай процент выполнения плана», «Добавь динамику к предыдущему месяцу», «Долю каждого региона», «Накопительный итог», «Факт минус план, делить на план, умножить на 100», «Убери последнюю колонку».</li>
<li><b>Перестройка:</b> «Перестрой третий график — отсортируй по проценту выполнения», «Покажи четвёртый график с учётом динамики», «Сделай четвёртый график тепловой картой», «Покажи ТОП-5».</li>
<li><b>Фильтры и переключатели:</b> «Оставь только Центральный регион», «Исключи Москву», «Переключи на точки», «По кварталам», «В процентах», «Метрика — скидка». Их же можно менять мышью над графиками.</li>
<li><b>Сохранение:</b> «Сохрани как «Продажи Q3»», «Покажи мои дашборды», «Открой …», «Переименуй в …», «Сделай копию», «Удали …», «Найди дашборд про скидки», «История версий», «Откати к версии 2», «Отмени».</li>
<li><b>Выгрузка:</b> «Скачай дашборд в HTML», «Выгрузи таблицу в Excel/CSV», «Скачай картинкой», «Поделись ссылкой».</li>
</ul>"""


def do_help(session: Session, action: dict):
    greeting = "Здравствуйте! " if action.get("greeting") else ""
    return [text_block(greeting + HELP_TEXT)], starter_suggestions()


UI_INFO = {
    "filter": "Фильтры уже есть над графиками: выпадающие списки с поиском (начните вводить название) и множественным выбором — период, регион, точка, категория, продукт. Фильтр действует на все графики и KPI сразу. В чате тоже можно: «оставь только Урал», «исключи Москву», «сбрось фильтры».",
    "cascade": "Каскад уже работает: выберите регион — в списке точек останутся только точки этого региона (так же категория → продукты).",
    "series": "Переключатель «Ряды: План и факт / План / Факт» уже есть над графиками — он действует на графики план-факта. В чате: «покажи только факт».",
    "grain": "Переключатель периода «Месяц / Квартал / Год» уже есть над графиками. В чате: «по кварталам».",
    "switch": "Над графиками есть переключатели: разрез (регионы / точки / категории / продукты), метрика (объём / скидка / количество / сделки), период (месяц / квартал / год) и вид (абсолютные / проценты / динамика).",
    "enlarge": "Нажмите «⤢» над нужным графиком — он развернётся на всю ширину дашборда; повторное нажатие вернёт обратно. "
               "Ещё можно выделить мышью область на графике — он приблизится.",
    "kpi": "Итоговые цифры уже наверху дашборда: план, факт, выполнение, отклонение, активные точки и клиенты, динамика к прошлому году. Зелёный — выполнение/рост, красный — отставание; карточки пересчитываются при любом фильтре.",
}


def do_ui_info(session: Session, action: dict):
    topic = action.get("topic") or "switch"
    blocks = [text_block(UI_INFO.get(topic, UI_INFO["switch"]))]
    if session.draft:
        blocks.append({"type": "highlight", "target": topic})
    return blocks, []


def do_reply(session: Session, action: dict):
    return [text_block(esc(action.get("text") or ""))], []


def do_clarify(session: Session, action: dict):
    options = []
    for option in action.get("options") or []:
        try:
            options.append({"label": str(option["label"])[:80], "action": validate_action(option["action"])})
        except (ActionError, KeyError, TypeError):
            continue
    session.pending = {"options": options} if options else None
    return [{"type": "clarify", "question": esc(action.get("question")), "options": options}], []


# --- продолжение разговора: смена периода, ответы на вопросы, описание данных ---------


def do_set_period(session: Session, action: dict):
    draft = require_draft(session)
    if draft.dataset == "custom":
        raise BotError("Этот дашборд построен по произвольному SQL — период в нём не меняется. Задайте новый запрос.")
    period = resolve_period(action.get("period"))
    if period is None:
        raise BotError("Не понял, какой период показать.")
    fetched = semantic.fetch_dataset(draft.dataset, period)
    if fetched.table.empty:
        dmin, dmax = semantic.db.data_bounds()
        raise BotError(f"За «{esc(period.label)}» данных нет: в базе продажи с {semantic.month_label(dmin)} по {semantic.month_label(dmax)}.")
    new_state = draft.clone()
    old_label = draft.period.label
    new_state.period = period
    new_state.table, new_state.prev, new_state.links, new_state.sql = fetched.table, fetched.prev, fetched.links, fetched.sql
    new_state.filters = {k: v for k, v in new_state.filters.items() if k != "period"}
    if period.months > 24 and new_state.switches.get("grain") == "month":
        new_state.switches["grain"] = "quarter"
    elif period.months <= 3 and new_state.switches.get("grain") != "month":
        new_state.switches["grain"] = "month"
    if not new_state.saved_id and old_label in (new_state.name or ""):
        new_state.name = new_state.name.replace(old_label, period.label)
    new_state.log(f"Период изменён: {old_label} → {period.label}")
    session.push_undo()
    session.draft = new_state
    return [text_block(f"Тот же дашборд за <b>{esc(period.label)}</b> ({period.range_label()}) — было: {esc(old_label)}. "
                       f"Фильтры, вычисления и графики сохранены; выгружено {fmt.num(len(fetched.table))} строк."),
            dashboard_block(session)], ["Кто отстаёт?", "Сохрани дашборд", "Отмени"]


def _answer_dataset(action: dict, draft: DashboardState | None) -> str:
    query = rules.norm(action.get("query") or "")
    if action.get("field") == "reason" or (action.get("measure") == "discount" and re.search(r"причин", query)):
        return "discounts"
    if action.get("field") == "client" or action.get("dim") == "client":
        return draft.dataset if draft and draft.dataset in ("clients", "discounts") else "clients"
    if draft and draft.dataset != "custom":
        needs_sales = action.get("what") in ("fact", "plan", "qty", "deals", "outlet", "product") or action.get("measure") in ("exec", "qty")
        if draft.dataset != "sales" and needs_sales and action.get("what") != "discount":
            if not (draft.dataset == "clients" and action.get("what") in ("fact", "qty", "deals", "product")):
                return "sales"
        return draft.dataset
    return "sales"


def _filters_note(state: DashboardState) -> str:
    active = {k: v for k, v in state.filters.items() if v}
    if not active:
        return ""
    parts = "; ".join(f"{DIM_LABELS.get(k, k).lower()}: {', '.join(v[:3])}{'…' if len(v) > 3 else ''}" for k, v in active.items())
    return f" <span style='color:#898781'>(с учётом фильтров — {esc(parts)})</span>"


def do_answer(session: Session, action: dict):
    draft = session.draft
    dataset = _answer_dataset(action, draft)
    period = resolve_period(action.get("period")) if action.get("period") else None
    need_new = draft is None or draft.dataset != dataset or (
        period is not None and (period.start, period.end) != (draft.period.start, draft.period.end))
    built: list[dict] = []
    suggestions: list[str] = []
    if need_new:
        spec = action.get("period") or ({"kind": "last_year"} if draft is None else None)
        build = {"type": "build", "dataset": dataset, "intent": "dashboard", "period": spec,
                 "dims": [action["dim"]] if action.get("dim") else [], "filters": {}, "query": action.get("query") or ""}
        built, suggestions = do_build(session, validate_action(build))
        if any(b["type"] == "clarify" for b in built):
            return built, suggestions
        built = [b for b in built if b["type"] != "text"]  # вместо длинного пояснения — ответ на вопрос
    state = require_draft(session)
    frame = state.filtered(state.table)
    scope = f"за {esc(state.period.label)}" + _filters_note(state)
    field, value = action.get("field"), action.get("value")
    if field and field not in frame.columns:
        raise BotError(f"В этих данных нет разреза «{DIM_LABELS.get(field, field)}».")
    sub = frame[frame[field] == value] if field and value else frame
    where = f" — {esc(value)}" if field and value else ""
    kind = action["kind"]
    has_plan = "plan" in frame.columns
    lines: list[str] = []

    def total(col: str, data: pd.DataFrame = sub) -> float:
        return float(data[col].sum()) if col in data.columns else float("nan")

    if kind == "count" and action.get("lagging"):
        what = action.get("what") or "outlet"
        if not has_plan:
            raise BotError("Для этого вопроса нужен план — он есть в наборе продаж: «Собери дашборд по продажам за последний год».")
        if what not in sub.columns:
            what = "outlet"
        by = sub.groupby(what)[["fact", "plan"]].sum()
        by = by[by["plan"] > 0]
        by["e"] = by["fact"] / by["plan"] * 100
        lag = by[by["e"] < 100].sort_values("e")
        noun_many = {"outlet": "точек", "region": "регионов", "category": "категорий", "product": "продуктов", "client": "клиентов"}.get(what, "позиций")
        text = f"План не выполнили <b>{len(lag)} из {len(by)}</b> {noun_many} {scope}."
        if len(lag):
            text += " Сильнее всего отстают: " + "; ".join(f"{esc(n)} — {fmt.pct(v)}" for n, v in lag["e"].head(3).items()) + "."
        return [text_block(text)] + built, ["Сравни продажи по торговым точкам за год, выдели отстающих", "Сохрани дашборд"]
    if kind == "yoy":
        prev = state.filtered_prev()
        col = "fact" if "fact" in sub.columns else ("discount" if "discount" in sub.columns else None)
        if prev is None or not len(prev) or col is None:
            raise BotError("Данных за тот же период прошлого года нет — сравнить не с чем.")
        psub = prev[prev[field] == value] if field and value and field in prev.columns else prev
        cur, old = total(col), float(psub[col].sum())
        change = (cur / old - 1) * 100 if old else float("nan")
        unit = "₽"
        lines.append(f"{'Продажи' if col == 'fact' else 'Скидки'}{where} {scope}: <b>{fmt.short(cur, unit)}</b> против "
                     f"{fmt.short(old, unit)} годом ранее — <b>{fmt.pct(change, signed=True)}</b>.")
        dim = state.switches.get("dim") if state.switches.get("dim") in sub.columns else None
        if dim and dim != field:
            a = sub.groupby(dim)[col].sum()
            b = psub.groupby(dim)[col].sum() if dim in psub.columns else pd.Series(dtype=float)
            growth = ((a / b.reindex(a.index)) - 1).dropna() * 100
            if len(growth) > 1:
                lines.append(f"Быстрее всех растёт {esc(growth.idxmax())} ({fmt.pct(growth.max(), signed=True)}), "
                             f"хуже всех — {esc(growth.idxmin())} ({fmt.pct(growth.min(), signed=True)}).")
        return [text_block(" ".join(lines))] + built, ["Добавь динамику год к году", "Покажи четвёртый график с учётом динамики"]
    if kind == "total":
        what = action.get("what") or "fact"
        if what == "fact" and "fact" in sub.columns:
            fact = total("fact")
            text = f"Продано на <b>{fmt.short(fact, '₽')}</b>{where} {scope}"
            if has_plan:
                plan = total("plan")
                text += f": план {fmt.short(plan, '₽')}, выполнение <b>{fmt.pct(fact / plan * 100 if plan else float('nan'))}</b>"
            lines.append(text + ".")
        elif what == "plan" and has_plan:
            lines.append(f"План{where} {scope}: <b>{fmt.short(total('plan'), '₽')}</b>; факт {fmt.short(total('fact'), '₽')}.")
        elif what == "qty" and "qty" in sub.columns:
            text = f"Продано <b>{fmt.short(total('qty'), 'шт')}</b>{where} {scope}"
            if "plan_qty" in sub.columns:
                text += f" при плане {fmt.short(total('plan_qty'), 'шт')}"
            lines.append(text + ".")
        elif what == "deals" and "deals" in sub.columns:
            lines.append(f"Сделок{where} {scope}: <b>{fmt.num(total('deals'))}</b>.")
        elif what == "discount" and "discount" in sub.columns:
            text = f"Скидок{where} {scope} на <b>{fmt.short(total('discount'), '₽')}</b>"
            if "discounts_cnt" in sub.columns:
                text += f" ({fmt.num(total('discounts_cnt'))} шт.)"
            elif "fact" in sub.columns and total("fact"):
                text += f" — {fmt.pct(total('discount') / total('fact') * 100)} от продаж"
            lines.append(text + ".")
        else:
            raise BotError("В текущих данных нет такого показателя — постройте продажи: «Собери дашборд по продажам за последний год».")
    elif kind == "count":
        what = action.get("what") or "client"
        if what == "client":
            if "client" in sub.columns:
                n = sub["client"].nunique()
            elif state.links is not None:
                links = state.filtered(state.links)
                if field and value and field in links.columns:
                    links = links[links[field] == value]
                n = links["client_id"].nunique()
            else:
                n = 0
            lines.append(f"Активных клиентов{where} {scope}: <b>{fmt.num(n)}</b> из {len(semantic.catalog()['client'])} в справочнике.")
        else:
            col = what
            active = sub[sub["fact"] > 0] if "fact" in sub.columns else sub
            n = active[col].nunique() if col in active.columns else 0
            word = {"outlet": ("Работающих точек", "outlet"), "product": ("Продаваемых продуктов", "product")}[col]
            lines.append(f"{word[0]}{where} {scope}: <b>{fmt.num(n)}</b> из {len(semantic.catalog()[word[1]])}.")
    elif kind in ("top", "bottom"):
        dim = action.get("dim") or state.switches.get("dim") or state.spec.switch_dims[0]
        if dim not in sub.columns:
            dim = next((d for d in state.spec.switch_dims if d in sub.columns), None)
        if state.filters.get(dim) or (dim == "region" and state.filters.get("outlet")):
            # сравниваем со всеми: фильтр по тому же разрезу для рейтинга не применяем
            tmp = state.clone()
            tmp.filters = {k: v for k, v in state.filters.items() if k != dim and not (dim == "region" and k == "outlet")}
            sub = tmp.filtered(tmp.table)
            if field and value:
                sub = sub[sub[field] == value]
            scope = f"за {esc(state.period.label)}" + _filters_note(tmp)
        measure = action.get("measure")
        if measure == "growth":
            prev = state.filtered_prev()
            col = "fact" if "fact" in sub.columns else "discount"
            if prev is None or not len(prev) or dim not in prev.columns:
                raise BotError("Для сравнения нужен тот же период прошлого года, а его в данных нет.")
            psub = prev[prev[field] == value] if field and value and field in prev.columns else prev
            a = sub.groupby(dim)[col].sum()
            b = psub.groupby(dim)[col].sum().reindex(a.index)
            growth = ((a / b) - 1).replace([np.inf, -np.inf], np.nan).dropna() * 100
            growth = growth.sort_values(ascending=kind == "bottom")
            if growth.empty:
                raise BotError("Не с чем сравнить: нет данных за прошлый год.")
            noun_gen = {"region": "регионов", "category": "категорий", "product": "продуктов", "outlet": "точек", "client": "клиентов"}.get(dim, "")
            verb = "Быстрее всех растёт" if kind == "top" else "Сильнее всех падает" if growth.iloc[0] < 0 else "Медленнее всех растёт"
            lines.append(f"{verb} {esc(growth.index[0])}: <b>{fmt.pct(growth.iloc[0], signed=True)}</b> к тому же периоду прошлого года {scope}.")
            rest = "; ".join(f"{esc(n)} {fmt.pct(v, signed=True)}" for n, v in growth.iloc[1:4].items())
            if rest:
                lines.append(f"Далее среди {noun_gen}: {rest}.")
            return [text_block(" ".join(lines))] + built, ["Добавь динамику год к году", "Сохрани дашборд"]
        if state.dataset == "discounts":
            measure = "discount"
        elif measure is None:
            measure = "exec" if has_plan else "fact"
        if measure == "exec" and not has_plan:
            measure = "fact"
        cols = [c for c in ("fact", "plan", "discount", "qty") if c in sub.columns]
        agg = sub.groupby(dim)[cols].sum()
        if measure == "exec":
            agg = agg[agg["plan"] > 0]
            agg["value"] = agg["fact"] / agg["plan"] * 100
        else:
            agg["value"] = agg[measure if measure in agg.columns else cols[0]]
        agg = agg.sort_values("value", ascending=kind == "bottom")
        if agg.empty:
            raise BotError("По выбранным фильтрам данных нет.")
        n = action.get("n") or 3
        unit = "%" if measure == "exec" else ("шт" if measure == "qty" else "₽")
        noun = {"region": "регион", "outlet": "точка", "product": "продукт", "category": "категория", "client": "клиент",
                "city": "город", "reason": "причина", "segment": "сегмент", "outlet_type": "тип точки"}.get(dim, "позиция")

        def show(v: float) -> str:
            return fmt.pct(v) if unit == "%" else fmt.short(v, unit)

        head_name, head = agg.index[0], agg.iloc[0]
        what_word = {"exec": "по выполнению плана", "fact": "по продажам", "discount": "по сумме скидок", "qty": "по количеству"}[measure]
        if kind == "top":
            text = f"Лидер {what_word}{where} {scope} — <b>{esc(head_name)}</b> ({noun}): {show(head['value'])}"
        else:
            text = f"Слабее всех {what_word}{where} {scope} — <b>{esc(head_name)}</b> ({noun}): {show(head['value'])}"
        if measure == "exec":
            text += f" (факт {fmt.short(head['fact'], '₽')} при плане {fmt.short(head['plan'], '₽')})"
        elif measure in ("fact", "discount") and agg["value"].sum():
            text += f" — {fmt.pct(head['value'] / agg['value'].sum() * 100)} от общего"
        lines.append(text + ".")
        rest = [f"{esc(name)} — {show(row['value'])}" for name, row in agg.iloc[1:n].iterrows()]
        if rest:
            lines.append(("Следом: " if kind == "top" else "Далее: ") + "; ".join(rest) + ".")
        if measure == "exec":
            lagging = int((agg["value"] < 100).sum())
            lines.append(f"План не выполняют {lagging} из {len(agg)}.")
        chart_idx = next((i + 1 for i, c in enumerate(state.charts) if c["type"] in ("top", "planfact_dim", "bar_dim")), None)
        suggestions = ([f"Перестрой {ORDINAL_GEN.get(chart_idx, str(chart_idx))} график — отсортируй по проценту выполнения"] if chart_idx and measure == "exec" else []) + \
                      [f"Покажи только {head_name}" if dim in ("region", "category") else "Покажи таблицу по " + DIM_DATIVE.get(dim, "регионам"),
                       "Сохрани дашборд"]
    elif kind == "share":
        measure = "discount" if (action.get("measure") == "discount" or state.dataset == "discounts") and "discount" in frame.columns else "fact"
        whole, part = total(measure, frame), total(measure, sub)
        what = {"discount": "суммы скидок", "fact": "продаж"}[measure]
        lines.append(f"Доля «{esc(value)}» — <b>{fmt.pct(part / whole * 100 if whole else float('nan'))}</b> {what} {scope} "
                     f"({fmt.short(part, '₽')} из {fmt.short(whole, '₽')}).")
    elif kind == "entity":
        if has_plan:
            fact, plan = total("fact"), total("plan")
            exec_pct = fact / plan * 100 if plan else float("nan")
            all_exec = total("fact", frame) / total("plan", frame) * 100 if total("plan", frame) else float("nan")
            compare = "лучше" if exec_pct >= all_exec else "хуже"
            lines.append(f"<b>{esc(value)}</b> {scope}: факт {fmt.short(fact, '₽')} при плане {fmt.short(plan, '₽')} — "
                         f"выполнение <b>{fmt.pct(exec_pct)}</b> ({compare} среднего {fmt.pct(all_exec)}).")
            prev = state.filtered_prev()
            if prev is not None and len(prev) and field in prev.columns:
                prev_fact = float(prev.loc[prev[field] == value, "fact"].sum())
                if prev_fact:
                    lines.append(f"К тому же периоду прошлого года: <b>{fmt.pct((fact / prev_fact - 1) * 100, signed=True)}</b>.")
            inner = {"region": "outlet", "category": "product", "city": "outlet", "outlet_type": "outlet"}.get(field)
            if inner and inner in sub.columns:
                by = sub.groupby(inner)[["fact", "plan"]].sum()
                by = by[by["plan"] > 0]
                if len(by) > 1:
                    by["e"] = by["fact"] / by["plan"] * 100
                    lines.append(f"Внутри: лучше всех — {esc(by['e'].idxmax())} ({fmt.pct(by['e'].max())}), "
                                 f"слабее всех — {esc(by['e'].idxmin())} ({fmt.pct(by['e'].min())}).")
        elif "discount" in sub.columns:
            text = f"<b>{esc(value)}</b> {scope}: скидок на {fmt.short(total('discount'), '₽')}"
            if "reason" in sub.columns and len(sub):
                reasons = sub.groupby("reason")["discount"].sum().sort_values(ascending=False)
                text += f", главная причина — «{esc(reasons.index[0])}» ({fmt.pct(reasons.iloc[0] / reasons.sum() * 100)})"
            lines.append(text + ".")
        else:
            lines.append(f"<b>{esc(value)}</b> {scope}: продажи {fmt.short(total('fact'), '₽')}, сделок {fmt.num(total('deals'))}.")
        suggestions = [f"Оставь только {value}", "Сохрани дашборд"]
    if not sub.shape[0] and kind != "count":
        lines = [f"По «{esc(value)}» {scope} данных нет."]
    return [text_block(" ".join(lines))] + built, suggestions or ["Кто отстаёт?", "Сохрани дашборд"]


ORDINAL_GEN = {1: "первый", 2: "второй", 3: "третий", 4: "четвёртый", 5: "пятый", 6: "шестой"}
DIM_DATIVE = {"region": "регионам", "outlet": "точкам", "category": "категориям", "product": "продуктам", "client": "клиентам",
              "reason": "причинам", "segment": "сегментам", "city": "городам"}


def do_show_sql(session: Session, action: dict):
    draft = require_draft(session)
    return [text_block(f"Данные для «{esc(draft.name)}» получены этим запросом (только чтение, проверен анализатором). "
                       "Все вычисления после выгрузки — на стороне бота."),
            {"type": "sql", "sql": draft.sql or "—", "tables": draft.spec.tables, "rows": int(len(draft.table)), "open": True}], []


def do_about_data(session: Session, action: dict):
    counts = semantic.db.run_select(
        "SELECT (SELECT COUNT(*) FROM sales) AS sales, (SELECT COUNT(*) FROM sales_plan) AS plans, "
        "(SELECT COUNT(*) FROM discounts) AS discounts, (SELECT COUNT(*) FROM contracts) AS contracts", trusted=True).iloc[0]
    cat = semantic.catalog()
    dmin, dmax = semantic.db.data_bounds()
    text = (f"В базе (только чтение) продажи с <b>{semantic.month_label(dmin)}</b> по <b>{semantic.month_label(dmax)}</b>: "
            f"{fmt.num(counts['sales'])} продаж, {fmt.num(counts['plans'])} строк плана, {fmt.num(counts['discounts'])} скидок, "
            f"{fmt.num(counts['contracts'])} договоров.<br>"
            f"Справочники: {len(cat['region'])} регионов, {len(cat['city'])} городов, {len(cat['outlet'])} торговых точек "
            f"({', '.join(cat['outlet_type']).lower()}), {len(cat['product'])} продуктов в {len(cat['category'])} категориях "
            f"({', '.join(cat['category']).lower()}), {len(cat['client'])} клиентов, {len(cat['reason'])} причин скидок.<br>"
            "Готовые наборы: <b>продажи план-факт</b> (месяц × точка × продукт), <b>скидки</b> (клиент × продукт × причина), "
            "<b>продажи по клиентам</b>. Если подключена языковая модель, можно спрашивать и о договорах — она напишет проверяемый SQL.")
    return [text_block(text)], ["Собери дашборд по продажам за последний год", "Покажи скидки за год по причинам", "ТОП-10 клиентов за год"]



HANDLERS = {
    "build": do_build, "custom_sql": do_custom_sql, "show_table": do_show_table, "add_calc": do_add_calc,
    "remove_calc": do_remove_calc, "modify_calc": do_modify_calc, "rebuild_chart": do_rebuild_chart,
    "add_chart": do_add_chart, "remove_chart": do_remove_chart, "set_filter": do_set_filter, "set_switch": do_set_switch,
    "save": do_save, "list": do_list, "open": do_open, "rename": do_rename, "delete": do_delete, "restore": do_restore,
    "copy": do_copy, "history": do_history, "revert": do_revert, "undo": do_undo, "refresh": do_refresh,
    "export": do_export, "share": do_share, "help": do_help, "ui_info": do_ui_info, "reply": do_reply,
    "clarify": do_clarify, "set_period": do_set_period, "answer": do_answer, "about_data": do_about_data,
    "show_sql": do_show_sql,
}
