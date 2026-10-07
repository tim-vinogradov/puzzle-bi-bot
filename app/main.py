"""Веб-сервер: чат-интерфейс и API.

Запуск:  python -m app.main   (или start.bat в Windows)
"""
from __future__ import annotations

import logging
import webbrowser
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import bot, export, storage
from .config import BASE_DIR, settings
from .dashboard import DashboardState, render

import os

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bibot")

STATIC = BASE_DIR / "static"

app = FastAPI(title="Puzzle BI-бот", version="1.0.0", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatIn(BaseModel):
    session_id: str | None = None
    text: str = Field(default="", max_length=2000)


class ActionIn(BaseModel):
    session_id: str | None = None
    action: dict[str, Any]
    label: str | None = None


class DraftUpdateIn(BaseModel):
    session_id: str
    uid: str | None = None
    filters: dict[str, list[str]] | None = None
    switches: dict[str, str] | None = None


class ShareRenderIn(BaseModel):
    filters: dict[str, list[str]] | None = None
    switches: dict[str, str] | None = None


def _index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index() -> FileResponse:
    return _index()


@app.get("/share/{token}")
def share_page(token: str) -> FileResponse:
    return _index()


@app.get("/api/health")
def health() -> dict:
    from . import db

    dmin, dmax = db.data_bounds()
    return {"status": "ok", "database": db.dialect_name(), "data_from": str(dmin.date()), "data_to": str(dmax.date()),
            "llm": settings.llm_provider if settings.llm_enabled else "выключена", "nlu_mode": settings.nlu_mode}


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    return bot.handle_message(body.session_id, body.text)


@app.post("/api/action")
def action(body: ActionIn) -> dict:
    payload = dict(body.action)
    if body.label:
        payload["_label"] = body.label
    return bot.handle_action(body.session_id, payload)


@app.post("/api/draft/update")
def draft_update(body: DraftUpdateIn) -> dict:
    try:
        return {"dashboard": bot.update_draft_ui(body.session_id, body.uid, body.filters, body.switches)}
    except bot.BotError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err


@app.get("/api/session")
def session_state(session_id: str) -> dict:
    session = bot.get_session(session_id)
    with session.lock:
        draft = render(session.draft) if session.draft else None
        return {"session_id": session.id, "messages": session.messages[-120:], "draft": draft}


@app.post("/api/session/new")
def session_new(body: ChatIn) -> dict:
    session = bot.reset_session(body.session_id or "")
    return {"session_id": session.id}


@app.get("/api/table")
def table(session_id: str, page: int = 0, group_by: str = "", sort_by: str = "", sort_dir: str = "desc",
          top_n: int | None = None) -> dict:
    session = bot.get_session(session_id)
    with session.lock:
        try:
            groups = [g for g in group_by.split(",") if g]
            return bot.table_block(session, page=max(0, page), group_by=groups, sort_by=sort_by or None,
                                   sort_dir="asc" if sort_dir == "asc" else "desc", top_n=top_n)
        except bot.BotError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err


@app.get("/api/dashboards")
def dashboards(q: str = "", sort: str = "updated") -> dict:
    return {"items": storage.list_dashboards(q, sort if sort in ("updated", "created", "name") else "updated")}


@app.get("/api/export")
def export_file(session_id: str, format: str = Query("html", pattern="^(html|csv|xlsx)$"), uid: str | None = None) -> Response:
    session = bot.get_session(session_id)
    with session.lock:
        draft = session.draft
        if draft is None:
            raise HTTPException(status_code=404, detail="Нет дашборда для выгрузки")
        stem = bot._file_stem(draft)
        if format == "csv":
            content, media = export.to_csv(draft), "text/csv; charset=utf-8"
        elif format == "xlsx":
            content, media = export.to_xlsx(draft), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            content, media = export.to_html(draft), "text/html; charset=utf-8"
    from urllib.parse import quote

    filename = f"{stem}.{format}"
    headers = {"Content-Disposition": f"attachment; filename=\"dashboard.{format}\"; filename*=UTF-8''{quote(filename)}"}
    return Response(content=content, media_type=media, headers=headers)


def _shared_state(token: str) -> tuple[dict, DashboardState]:
    info = storage.by_share_token(token)
    if info is None:
        raise HTTPException(status_code=404, detail="Ссылка недействительна или дашборд удалён")
    info, meta, frames = storage.load(info["id"])
    state = DashboardState.from_meta(meta, *frames)
    state.name = info["name"]
    return info, state


@app.get("/api/share/{token}")
def share_data(token: str) -> dict:
    info, state = _shared_state(token)
    return {"dashboard": render(state), "name": info["name"], "updated_at": info["updated_at"], "version": info["version"]}


@app.post("/api/share/{token}/render")
def share_render(token: str, body: ShareRenderIn) -> dict:
    info, state = _shared_state(token)
    if body.filters is not None:
        state.filters = {k: [str(x) for x in v][:500] for k, v in body.filters.items() if k in state.table.columns and v}
    if body.switches is not None:
        for key, value in body.switches.items():
            if key in ("dim", "metric", "grain", "view", "series"):
                state.switches[key] = value
    return {"dashboard": render(state)}


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover - страховка
    log.exception("Необработанная ошибка: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка: {exc}"})


def main() -> None:
    import threading

    import uvicorn

    # прогреваем базу (при первом запуске собирается SQLite из CSV)
    from . import db, semantic

    db.data_bounds()
    semantic.catalog()
    url = f"http://{settings.host}:{settings.port}"
    print(f"\n  Puzzle BI-бот запущен: {url}\n  Остановить: Ctrl+C\n")
    if settings.host in ("127.0.0.1", "localhost"):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
