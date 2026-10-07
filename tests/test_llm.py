"""Режим языковой модели: проверяем на локальном «макете» OpenAI-совместимого API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

REPLIES = {
    "какие договоры с отсрочкой больше 30 дней по регионам": {"actions": [{
        "type": "custom_sql", "title": "Договоры с отсрочкой > 30 дней",
        "sql": "SELECT r.region_name AS region, COUNT(*) AS contracts FROM contracts c JOIN clients cl ON cl.client_id = c.client_id "
               "JOIN regions r ON r.region_id = cl.region_id WHERE c.payment_terms_days > 30 GROUP BY r.region_name"}]},
    "выполни служебную чистку": {"actions": [{"type": "custom_sql", "sql": "DELETE FROM sales", "title": "x"}]},
    "расклад по столице на текущий год": {"actions": [{
        "type": "build", "dataset": "sales", "intent": "dashboard", "period": {"kind": "ytd"}, "dims": ["outlet"],
        "filters": {"city": ["Москва"]}, "evil": "__import__('os')"}]},
    "спасибо": {"actions": [{"type": "reply", "text": "Пожалуйста! Чем ещё помочь?"}]},
}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "system"
        question = body["messages"][-1]["content"].strip().lower()
        reply = REPLIES.get(question, {"actions": [{"type": "reply", "text": "?"}]})
        data = json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(reply, ensure_ascii=False) + "\n```"}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_llm(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from app.config import settings

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_base_url", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(settings, "llm_api_key", "test")
    monkeypatch.setattr(settings, "nlu_mode", "hybrid")
    yield
    server.shutdown()


def say(client, text, sid):
    return client.post("/api/chat", json={"session_id": sid, "text": text}).json()["message"]["blocks"]


def test_llm_custom_sql(client, fake_llm):
    blocks = say(client, "Какие договоры с отсрочкой больше 30 дней по регионам", "llm-1")
    assert [b["type"] for b in blocks] == ["text", "sql", "table", "dashboard"]
    assert "LIMIT" in blocks[1]["sql"] or blocks[1]["rows"] > 0


def test_llm_cannot_write(client, fake_llm):
    blocks = say(client, "Выполни служебную чистку", "llm-2")
    assert blocks[0].get("tone") == "error"
    blocks = say(client, "Удали все продажи", "llm-2")
    assert "только на чтение" in blocks[0]["text"]


def test_llm_build_is_validated(client, fake_llm):
    blocks = say(client, "Расклад по столице на текущий год", "llm-3")
    dash = next(b["dashboard"] for b in blocks if b["type"] == "dashboard")
    assert dash["active_filters"]["outlet"]  # город превращён в фильтр по точкам


def test_llm_reply(client, fake_llm):
    blocks = say(client, "Спасибо", "llm-4")
    assert "Пожалуйста" in blocks[0]["text"]
