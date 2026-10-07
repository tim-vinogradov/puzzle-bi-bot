"""Понимание запросов: все формулировки из кейса."""
import copy

import pytest

from app.nlu.actions import validate_action
from app.nlu.rules import NLUContext, parse, parse_period


@pytest.fixture()
def empty(catalog):
    return NLUContext(catalog=catalog)


@pytest.fixture()
def dash(catalog):
    return NLUContext(has_dashboard=True, dataset="sales", catalog=catalog,
                      switches={"dim": "region", "metric": "amount", "grain": "month", "view": "abs"},
                      charts=[{"type": "planfact_period"}, {"type": "structure"}, {"type": "top"}, {"type": "exec_line"}],
                      calcs=[{"id": "exec", "label": "Выполнение плана, %", "kind": "expr"},
                             {"id": "mom_fact", "label": "Динамика к предыдущему месяцу, %", "kind": "change"}])


def first(text, ctx):
    result = parse(text, copy.deepcopy(ctx))
    assert result.actions, text
    for action in result.actions:
        validate_action(action)
    return result.actions[0]


@pytest.mark.parametrize("text,period", [
    ("за последний квартал", {"kind": "last_quarter"}),
    ("за последний год", {"kind": "last_year"}),
    ("за последние 3 года", {"kind": "last_n_months", "n": 36}),
    ("за 2 квартал 2026", {"kind": "quarter", "quarter": 2, "year": 2026}),
    ("за сентябрь 2025", {"kind": "month", "month": 9, "year": 2025}),
    ("за 2024 год", {"kind": "year", "year": 2024}),
    ("с начала года", {"kind": "ytd"}),
    ("за всё время", {"kind": "all"}),
    ("за прошлый год", {"kind": "prev_year"}),
])
def test_periods(text, period):
    assert parse_period(text.lower().replace("ё", "е")) == {**period, **({"year": period.get("year")} if "year" in period else {})}


@pytest.mark.parametrize("text,expected", [
    ("Покажи план-факт продаж по регионам и торговым точкам за последний квартал", {"intent": "planfact_dim", "dims": ["region", "outlet"]}),
    ("Собери дашборд по продажам: план-факт по регионам и торговым точкам за последний год", {"intent": "dashboard", "dataset": "sales"}),
    ("Собери дашборд по продажам: динамика, структура, ТОП точек, выполнение плана", {"intent": "dashboard", "dims": ["outlet"]}),
    ("Покажи по клиенту Сибирский Продукт за последние 3 года, какие скидки делались и почему", {"dataset": "discounts"}),
    ("Сравни продажи по торговым точкам за год, выдели отстающих", {"intent": "compare", "lagging": True}),
    ("Покажи структуру продаж по продуктам за последний месяц", {"intent": "structure", "structure_dim": "product"}),
])
def test_build_requests(text, expected, empty):
    action = first(text, empty)
    assert action["type"] == "build"
    for key, value in expected.items():
        assert action[key] == value, (key, action)


@pytest.mark.parametrize("text,kind,extra", [
    ("Добавь колонку с отклонением факта от плана", "expr", {"expr": "fact - plan"}),
    ("Посчитай процент выполнения плана", "expr", {"expr": "fact / plan * 100"}),
    ("Добавь динамику к предыдущему месяцу", "change", {"lag": "month"}),
    ("Посчитай долю каждого региона в общем объёме", "share", {"by": "region"}),
    ("Добавь накопительный итог по месяцам", "cumsum", {}),
    ("посчитай медиану факта по точкам", "group_stat", {"stat": "median", "by": "outlet"}),
    ("ранжируй регионы по факту", "rank", {"by": "region"}),
    ("факт минус план, делить на план, умножить на 100 — посчитай", "expr", {"expr": "((fact - plan) / plan) * 100"}),
])
def test_calcs(text, kind, extra, dash):
    action = first(text, dash)
    assert action["type"] == "add_calc"
    calc = action["calcs"][0]
    assert calc["kind"] == kind
    for key, value in extra.items():
        assert calc[key] == value


@pytest.mark.parametrize("text,atype,check", [
    ("Перестрой третий график — отсортируй по проценту выполнения", "rebuild_chart", lambda a: a["chart"] == 3 and a["changes"]["sort_by"] == "exec"),
    ("Покажи четвёртый график с учётом динамики", "rebuild_chart", lambda a: a["chart"] == 4 and a["changes"]["value"] == "mom_fact"),
    ("Перестрой график с учётом новой колонки", "rebuild_chart", lambda a: a["changes"].get("value") == "mom_fact"),
    ("Отсортируй по проценту выполнения и покажи ТОП-10", "rebuild_chart", lambda a: a["chart"] == 3),
    ("покажи только Северо-Запад", "set_filter", lambda a: a["values"] == ["Северо-Западный"]),
    ("Оставь только Центральный регион", "set_filter", lambda a: a["values"] == ["Центральный"]),
    ("исключи Москву", "set_filter", lambda a: a["mode"] == "exclude"),
    ("добавь разбивку по кварталам", "set_switch", lambda a: a["grain"] == "quarter"),
    ("переключи на точки", "set_switch", lambda a: a["dim"] == "outlet"),
    ("сохрани как «Продажи Q3»", "save", lambda a: a["name"] == "Продажи Q3"),
    ("Покажи мои дашборды", "list", lambda a: True),
    ("Открой дашборд «Продажи Q3»", "open", lambda a: a["name"] == "Продажи Q3"),
    ("Переименуй дашборд в «Продажи Q3 финал»", "rename", lambda a: a["new_name"] == "Продажи Q3 финал"),
    ("Удали дашборд «Старый отчёт»", "delete", lambda a: a["name"] == "Старый отчёт"),
    ("Найди дашборд про скидки", "list", lambda a: a["query"] == "скидки"),
    ("Сделай копию дашборда", "copy", lambda a: True),
    ("Скачай дашборд в HTML", "export", lambda a: a["format"] == "html"),
    ("Выгрузи таблицу в Excel", "export", lambda a: a["format"] == "xlsx"),
    ("Отмени", "undo", lambda a: True),
    ("Откати к версии 2", "revert", lambda a: a["version"] == 2),
    ("Добавь фильтр по региону с поиском", "ui_info", lambda a: True),
    ("Сделай переключатель план/факт", "ui_info", lambda a: a["topic"] == "series"),
    ("Сделай каскадный фильтр: регион → точка", "ui_info", lambda a: a["topic"] == "cascade"),
])
def test_dialog_commands(text, atype, check, dash):
    action = first(text, dash)
    assert action["type"] == atype, action
    assert check(action), action


def test_compound_request(dash):
    result = parse("добавь колонку с процентом выполнения плана и перестрой третий график", copy.deepcopy(dash))
    assert [a["type"] for a in result.actions] == ["add_calc", "rebuild_chart"]
