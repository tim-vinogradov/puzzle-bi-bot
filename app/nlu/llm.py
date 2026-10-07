"""Понимание запросов языковой моделью (необязательно).

Поддерживаются:
* любой OpenAI-совместимый API (OpenAI, YandexGPT через совместимый endpoint,
  DeepSeek, OpenRouter, локальные Ollama/LM Studio и т. п.);
* GigaChat (Сбер) — с получением токена по ключу авторизации.

Модель НЕ пишет код и не выполняет его: она возвращает JSON со списком действий
того же формата, что и встроенный разборщик. Всё проверяется `validate_action`,
произвольный SQL — `sql_guard`.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid

import httpx

from ..config import settings
from .actions import ActionError, validate_action
from .rules import NLUContext

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


SYSTEM_PROMPT = """Ты — модуль понимания запросов в BI-чат-боте. Пользователь пишет по-русски,
ты переводишь сообщение в JSON со списком действий. Отвечай ТОЛЬКО JSON вида {"actions": [...]}.
Никакого текста вне JSON, никакого кода, кроме SQL в действии custom_sql.

## Данные
Набор sales (продажи план/факт), строка = месяц × торговая точка × продукт:
  разрезы: period, region, city, outlet, outlet_type, category, product
  показатели: plan, fact (₽), plan_qty, qty (шт), discount (₽), deals (число сделок)
Набор discounts (скидки), строка = месяц × клиент × продукт × причина:
  разрезы: period, client, segment, region, category, product, reason
  показатели: discount (₽), discounts_cnt, pct_sum (служебный), sale_amount
Набор clients (продажи по клиентам): разрезы period, client, segment, region, category, product;
  показатели fact, qty, deals, discount
Данные есть с {data_min} по {data_max}. «Сегодня» для относительных периодов = {anchor}.

Справочники:
  регионы: {regions}
  категории: {categories}
  причины скидок: {reasons}
  сегменты: {segments}
  продукты: {products}
  клиенты (часть): {clients}

Таблицы базы ({dialect}) для custom_sql:
{schema}

## Текущий дашборд
{dashboard}

## Действия (поле "type")
- build: новый запрос данных и визуализация.
  {"type":"build","dataset":"sales|discounts|clients","intent":"dashboard|compare|structure|planfact_dim|trend|top|exec",
   "period":{"kind":"last_quarter|last_month|last_year|last_n_months|ytd|year|quarter|month|all|range|prev_year",
             "n":6,"year":2025,"quarter":3,"month":9,"start":"2025-01-01","end":"2025-06-30"},
   "dims":["region"],"metric":"amount|qty|discount|deals","grain":"month|quarter|year","top_n":10,
   "lagging":false,"structure_dim":"category","filters":{"region":["Центральный"]},"client":"название клиента"}
  intent: dashboard — 4 графика; planfact_dim — план-факт по разрезу; structure — доли; compare — сравнение и отстающие;
  trend — по периодам; top — рейтинг; exec — динамика выполнения плана.
  Если период не назван — period = null (бот сам уточнит).
- show_table: {"type":"show_table","group_by":["region"],"sort_by":"fact","sort_dir":"desc","top_n":10}
- add_calc: {"type":"add_calc","calcs":[CALC,...]} где CALC:
   {"kind":"expr","expr":"(fact - plan) / plan * 100","label":"Отклонение от плана, %","unit":"%"}
     (в expr только колонки, числа, + - * / скобки, abs(), round(); можно ссылаться на id прежних вычислений)
   {"kind":"change","measure":"fact","lag":"month|year|prev","mode":"pct|abs"}  — динамика
   {"kind":"share","measure":"fact","by":"region"}  — доля в общем объёме
   {"kind":"cumsum","measure":"fact"}  — накопительный итог
   {"kind":"rank","measure":"fact","by":"region","desc":true}
   {"kind":"group_stat","measure":"fact","by":"region","stat":"mean|median|min|max"}
- remove_calc: {"type":"remove_calc","ref":"id или название или last"}
- modify_calc: {"type":"modify_calc","ref":"...","calc":CALC}
- rebuild_chart: {"type":"rebuild_chart","chart":3,"changes":{"sort_by":"exec_pct или id вычисления","sort_dir":"asc|desc",
   "top_n":10,"value":"id вычисления для отображения","dim":"region","type":"planfact_period|structure|top|exec_line|planfact_dim|scatter_planfact|trend|bar_dim",
   "variant":"heatmap","grain":"quarter","view":"pct"}}
- add_chart / remove_chart: {"type":"add_chart","changes":{...}} / {"type":"remove_chart","chart":2}
- set_filter: {"type":"set_filter","field":"region","values":["Центральный"],"mode":"only|add|exclude|clear|clear_all"}
- set_switch: {"type":"set_switch","dim":"outlet","metric":"discount","grain":"quarter","view":"abs|pct|dyn","series":"both|plan|fact"}
- save {"name":..,"as_new":false}; list {"query":"скидки","sort":"updated|created|name"}; open {"name":..};
  rename {"name":null,"new_name":..}; delete {"name":..}; copy {"new_name":..}; restore {"name":..}
- export {"format":"html|csv|xlsx|png"}; share; undo; history; revert {"version":2}; refresh; help
- custom_sql: {"type":"custom_sql","sql":"SELECT ...","title":"..."} — только если вопрос не покрывается наборами выше.
  Только один SELECT, только таблицы из списка, без изменения данных, с LIMIT.
- clarify: {"type":"clarify","question":"...","options":[{"label":"...","action":{...}}]} — если запрос неоднозначен.
- reply: {"type":"reply","text":"..."} — если это не запрос к данным (благодарность, вопрос о возможностях).

Несколько действий подряд допустимы (например, add_calc и затем rebuild_chart).
"""


def _dashboard_context(ctx: NLUContext) -> str:
    if not ctx.has_dashboard:
        return "Дашборда пока нет."
    charts = "; ".join(f"{i + 1}: {c.get('type')} {c.get('title', '')}".strip() for i, c in enumerate(ctx.charts))
    calcs = "; ".join(f"{c['id']} = {c['label']}" for c in ctx.calcs) or "нет"
    return (f"Набор: {ctx.dataset}. Переключатели: {json.dumps(ctx.switches, ensure_ascii=False)}. "
            f"Графики: {charts}. Вычисления: {calcs}. Название: {ctx.current_name or 'не сохранён'}.")


def build_system_prompt(ctx: NLUContext) -> str:
    from .. import db, semantic

    cat = ctx.catalog or semantic.catalog()
    dmin, dmax = db.data_bounds()
    values = {
        "data_min": dmin.date().isoformat(), "data_max": dmax.date().isoformat(),
        "anchor": semantic.anchor_date().isoformat(),
        "regions": ", ".join(cat.get("region", [])), "categories": ", ".join(cat.get("category", [])),
        "reasons": ", ".join(cat.get("reason", [])), "segments": ", ".join(cat.get("segment", [])),
        "products": ", ".join(cat.get("product", [])), "clients": ", ".join(cat.get("client", [])[:80]),
        "dialect": db.dialect_name(), "schema": semantic.schema_description(),
        "dashboard": _dashboard_context(ctx),
    }
    prompt = SYSTEM_PROMPT
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


class _GigaToken:
    lock = threading.Lock()
    token: str | None = None
    expires: float = 0


def _gigachat_token() -> str:
    with _GigaToken.lock:
        if _GigaToken.token and time.time() < _GigaToken.expires - 60:
            return _GigaToken.token
        response = httpx.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={"Authorization": f"Basic {settings.gigachat_auth_key}", "RqUID": str(uuid.uuid4()),
                     "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={"scope": settings.gigachat_scope}, verify=settings.gigachat_verify_ssl, timeout=settings.llm_timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        _GigaToken.token = data["access_token"]
        _GigaToken.expires = data.get("expires_at", 0) / 1000 or time.time() + 1500
        return _GigaToken.token


def chat_completion(messages: list[dict]) -> str:
    if settings.llm_provider == "gigachat":
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {_gigachat_token()}", "Accept": "application/json"}
        body = {"model": settings.gigachat_model, "messages": messages, "temperature": 0.01}
        verify = settings.gigachat_verify_ssl
    else:
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        body = {"model": settings.llm_model, "messages": messages, "temperature": 0,
                "response_format": {"type": "json_object"}}
        verify = True
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_sec, verify=verify)
        if response.status_code == 400 and "response_format" in body:
            body.pop("response_format")
            response = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_sec, verify=verify)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPError as err:
        raise LLMError(f"Языковая модель недоступна: {err}") from err
    except (KeyError, IndexError, ValueError) as err:
        raise LLMError("Неожиданный ответ языковой модели") from err


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise LLMError("Модель вернула не JSON")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as err:
            raise LLMError("Модель вернула некорректный JSON") from err


def understand(text: str, ctx: NLUContext, history: list[dict] | None = None) -> list[dict]:
    """Отправляет запрос в модель и возвращает проверенный список действий."""
    messages = [{"role": "system", "content": build_system_prompt(ctx)}]
    for item in (history or [])[-6:]:
        messages.append(item)
    messages.append({"role": "user", "content": text})
    raw = chat_completion(messages)
    data = extract_json(raw)
    actions_raw = data.get("actions") if isinstance(data, dict) else data
    if isinstance(actions_raw, dict):
        actions_raw = [actions_raw]
    if not isinstance(actions_raw, list) or not actions_raw:
        raise LLMError("Модель не вернула действий")
    actions = []
    for item in actions_raw[:6]:
        try:
            actions.append(validate_action(item))
        except ActionError as err:
            log.warning("Отброшено действие модели %s: %s", item, err)
    if not actions:
        raise LLMError("Модель вернула только некорректные действия")
    for action in actions:
        if action["type"] == "build" and not action.get("query"):
            action["query"] = text
    return actions
