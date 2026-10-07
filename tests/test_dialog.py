"""Живой диалог: вопросы с ответом числом, продолжение разговора о текущем дашборде."""
import re

SID = "pytest-dialog"


def say(client, text, sid=SID):
    blocks = client.post("/api/chat", json={"session_id": sid, "text": text}).json()["message"]["blocks"]
    errors = [b["text"] for b in blocks if b["type"] == "text" and b.get("tone") == "error"]
    assert not errors, (text, errors)
    return blocks


def first_text(blocks):
    return re.sub(r"<[^>]+>", "", next(b["text"] for b in blocks if b["type"] == "text"))


def dash(blocks):
    return next((b["dashboard"] for b in blocks if b["type"] in ("dashboard", "dashboard_update")), None)


def test_meta_questions(client):
    assert "Что умею" in first_text(say(client, "какие дашборды ты можешь собрать?", "dlg-meta"))
    assert "В базе" in first_text(say(client, "какие данные у тебя есть?", "dlg-meta"))
    assert "Пожалуйста" in first_text(say(client, "спасибо", "dlg-meta"))


def test_questions_and_followups(client):
    blocks = say(client, "сколько продали за сентябрь 2026?")
    assert first_text(blocks).startswith("Продано на")
    assert dash(blocks)["period_label"] == "сен 2026"
    assert "Лидер по выполнению плана" in first_text(say(client, "какой регион лучший?"))
    assert "Слабее всех" in first_text(say(client, "кто отстаёт?"))
    assert "План не выполнили" in first_text(say(client, "сколько точек не выполнили план?"))
    assert "Уральский" in first_text(say(client, "как дела у Урала?"))
    # продолжение: тот же дашборд за другой период
    blocks = say(client, "а за прошлый год?")
    assert "Тот же дашборд" in first_text(blocks)
    assert dash(blocks)["period"]["label"] == "2025 год"
    # вопрос без периода не переспрашивает период, а берёт текущий
    blocks = say(client, "топ 5 точек по выручке")
    assert not any(b["type"] == "clarify" for b in blocks)
    # город через «а в …?»
    blocks = say(client, "покажи продажи в Москве")
    assert dash(blocks)["active_filters"].get("outlet")
    blocks = say(client, "а в Казани?")
    assert "Казань" in first_text(blocks)
    assert "Сбросил" in first_text(say(client, "верни все города"))
    assert "годом ранее" in first_text(say(client, "сравни с прошлым годом"))
    assert "растёт" in first_text(say(client, "какая категория растёт быстрее всех?"))


def test_calc_pronouns_and_unknown_items(client):
    sid = "dlg-calc"
    say(client, "дашборд по продажам за последний год", sid)
    say(client, "добавь отклонение", sid)
    assert "Перестроил" in first_text(say(client, "покажи его на графике", sid))
    assert "Убрал колонку" in first_text(say(client, "удали эту колонку", sid))
    assert "себестоимости" in first_text(say(client, "посчитай маржу", sid))
    blocks = say(client, "оставь только сыр", sid)
    assert "Сыр" in first_text(blocks)


def test_discount_questions(client):
    sid = "dlg-disc"
    say(client, "какие скидки мы даём и почему за 2025", sid)
    assert "(клиент)" in first_text(say(client, "кому больше всего скидок?", sid))
    assert "Доля «Лояльность клиента»" in first_text(say(client, "доля лояльности клиентов в скидках", sid))


def test_restore_and_open_by_short_name(client):
    sid = "dlg-save"
    say(client, "дашборд по продажам за 2 квартал 2026", sid)
    say(client, "сохрани как «Квартальный разбор»", sid)
    say(client, "удали дашборд «Квартальный разбор»", sid)
    assert "Восстановил" in first_text(say(client, "восстанови", sid))
    assert "Открыл" in first_text(say(client, "открой квартальный", sid))
