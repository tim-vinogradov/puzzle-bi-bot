"""Полный демо-сценарий из кейса через HTTP API."""
import re

import pytest

SID = "pytest-demo"


def say(client, text, sid=SID):
    response = client.post("/api/chat", json={"session_id": sid, "text": text})
    assert response.status_code == 200, response.text
    blocks = response.json()["message"]["blocks"]
    errors = [b["text"] for b in blocks if b["type"] == "text" and b.get("tone") == "error"]
    assert not errors, (text, errors)
    return blocks


def dashboard(blocks):
    for block in reversed(blocks):
        if block["type"] in ("dashboard", "dashboard_update"):
            return block["dashboard"]
    return None


def kpi(dash, key):
    return next(k for k in dash["kpis"] if k["key"] == key)


def test_full_demo(client):
    # 1. запрос без периода — бот уточняет
    blocks = say(client, "Собери дашборд по продажам: план-факт по регионам и торговым точкам")
    assert blocks[0]["type"] == "clarify"
    # 2. ответ на уточнение
    blocks = say(client, "за последний год")
    types = [b["type"] for b in blocks]
    assert types == ["text", "sql", "table", "dashboard"]
    dash = dashboard(blocks)
    assert len(dash["charts"]) == 4
    assert all(not c.get("error") for c in dash["charts"])
    assert kpi(dash, "plan")["value"] == pytest.approx(1_249_993_200, rel=1e-6)
    assert kpi(dash, "exec")["status"] == "bad"
    assert {s["key"] for s in dash["switches"]} >= {"dim", "metric", "grain", "view"}
    assert {f["key"] for f in dash["filters"]} == {"period", "region", "outlet", "category", "product"}
    outlet_filter = next(f for f in dash["filters"] if f["key"] == "outlet")
    assert outlet_filter["parent"] == "region" and len(outlet_filter["options"]) >= 290
    assert dash["summary"]

    # 3. вычисления и перестройка
    blocks = say(client, "Добавь колонку с процентом выполнения плана")
    assert any(b["type"] == "table" and any(c["calc"] for c in b["columns"]) for b in blocks)
    blocks = say(client, "Перестрой третий график — отсортируй по проценту выполнения")
    chart3 = dashboard(blocks)["charts"][2]
    assert "выполнению" in chart3["title"]
    say(client, "Добавь вычисление динамики к прошлому месяцу")
    blocks = say(client, "Покажи четвёртый график с учётом динамики")
    assert "Динамика" in dashboard(blocks)["charts"][3]["title"]

    # 4. фильтр и переключатель
    blocks = say(client, "Оставь только Центральный регион")
    dash = dashboard(blocks)
    assert dash["active_filters"] == {"region": ["Центральный"]}
    assert kpi(dash, "exec")["value"] > 100
    blocks = say(client, "Переключи разрез на точки")
    assert next(s for s in dashboard(blocks)["switches"] if s["key"] == "dim")["value"] == "outlet"

    # 5. сохранение, список, открытие
    say(client, "Сохрани дашборд под названием «Продажи Q3»")
    blocks = say(client, "Покажи мои дашборды")
    names = [i["name"] for b in blocks if b["type"] == "dashboards" for i in b["items"]]
    assert "Продажи Q3" in names
    say(client, "Сбрось фильтры")
    blocks = say(client, "Открой дашборд «Продажи Q3»")
    dash = dashboard(blocks)
    assert dash["active_filters"] == {"region": ["Центральный"]}
    assert [c["label"] for c in dash["calcs"]] == ["Выполнение плана, %", "Динамика к предыдущему месяцу, %"]
    assert next(s for s in dash["switches"] if s["key"] == "dim")["value"] == "outlet"

    # 6. переименование, копия, поиск, удаление
    say(client, "Переименуй дашборд в «Продажи Q3 финал»")
    say(client, "Сделай копию дашборда")
    blocks = say(client, "Найди дашборд про продажи")
    names = [i["name"] for b in blocks if b["type"] == "dashboards" for i in b["items"]]
    assert "Продажи Q3 финал" in names and "Продажи Q3 финал (копия)" in names
    say(client, "Удали дашборд «Продажи Q3 финал (копия)»")
    blocks = say(client, "Покажи мои дашборды")
    names = [i["name"] for b in blocks if b["type"] == "dashboards" for i in b["items"]]
    assert "Продажи Q3 финал (копия)" not in names

    # 7. выгрузки
    for fmt, marker in (("html", b"<!doctype html>"), ("csv", "Период".encode("utf-8")), ("xlsx", b"PK")):
        response = client.get("/api/export", params={"session_id": SID, "format": fmt})
        assert response.status_code == 200
        assert marker in response.content[:4000]


def test_ui_filter_update_and_empty_data(client):
    sid = "pytest-ui"
    say(client, "Покажи план-факт продаж по регионам за последний квартал", sid)
    session = client.get("/api/session", params={"session_id": sid}).json()
    uid = session["draft"]["uid"]
    response = client.post("/api/draft/update", json={"session_id": sid, "uid": uid, "filters": {"region": ["Нет такого"]}})
    dash = response.json()["dashboard"]
    assert dash["rows_filtered"] == 0
    assert all(not c.get("error") for c in dash["charts"])  # пустые данные не роняют графики
    response = client.post("/api/draft/update", json={"session_id": sid, "uid": uid, "filters": {}, "switches": {"grain": "quarter", "view": "pct"}})
    assert response.json()["dashboard"]["rows_filtered"] > 0


def test_discounts_and_clarification(client):
    sid = "pytest-disc"
    blocks = say(client, "Покажи по клиенту Х за последние 3 года, какие скидки делались и почему", sid)
    assert blocks[0]["type"] == "clarify"
    blocks = say(client, "1", sid)
    dash = dashboard(blocks)
    assert dash["dataset"] == "discounts"
    assert len(dash["active_filters"]["client"]) == 1
    assert any("причина" in s.lower() for s in dash["summary"])


def test_all_example_requests(client):
    sid = "pytest-examples"
    for text in [
        "Покажи план-факт продаж по регионам за последний квартал",
        "Покажи структуру продаж по продуктам за последний месяц",
        "Сравни продажи по торговым точкам за год, выдели отстающих",
        "Собери дашборд по продажам: динамика, структура, ТОП точек, выполнение плана за последний год",
        "Добавь колонку с отклонением факта от плана",
        "Посчитай процент выполнения плана",
        "Добавь динамику к предыдущему месяцу",
        "Посчитай долю каждого региона в общем объёме",
        "Добавь накопительный итог по месяцам",
        "Отсортируй по проценту выполнения и покажи ТОП-10",
        "Перестрой график с учётом новой колонки",
        "Добавь фильтр по региону с поиском",
        "Сделай переключатель план/факт",
        "Покажи итоговые цифры сверху",
        "Сделай каскадный фильтр: регион → точка",
        "Добавь переключатель периода: месяц/квартал/год",
        "Сохрани дашборд",
        "Покажи мои дашборды",
        "История версий",
        "Отмени",
        "Сделай четвёртый график тепловой картой",
        "Покажи таблицу по регионам",
        "Скачай картинкой",
    ]:
        blocks = say(client, text, sid)
        dash = dashboard(blocks)
        if dash:
            assert all(not c.get("error") for c in dash["charts"]), (text, [c.get("error") for c in dash["charts"]])


def test_share_link(client):
    sid = "pytest-share"
    say(client, "Покажи план-факт продаж по регионам за последний квартал", sid)
    say(client, "Сохрани как «Для коллеги»", sid)
    blocks = say(client, "Поделись ссылкой", sid)
    url = next(b["url"] for b in blocks if b["type"] == "share")
    token = re.sub(r"^/share/", "", url)
    data = client.get(f"/api/share/{token}").json()
    assert data["name"] == "Для коллеги"
    filtered = client.post(f"/api/share/{token}/render", json={"filters": {"region": ["Уральский"]}}).json()
    assert filtered["dashboard"]["rows_filtered"] < data["dashboard"]["rows_filtered"]
