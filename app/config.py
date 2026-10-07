"""Настройки приложения.

Все параметры читаются из переменных окружения или из файла `.env`
в корне проекта (формат `КЛЮЧ=значение`). Пример — в `.env.example`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Минимальный загрузчик .env без внешних зависимостей."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "1" if default else "0").lower()
    return value in {"1", "true", "yes", "да", "on"}


@dataclass
class Settings:
    # Источник данных. По умолчанию — встроенная SQLite, собранная из CSV кейса.
    # Для PostgreSQL: postgresql+psycopg2://bi_reader:пароль@localhost:5432/puzzle
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR") or BASE_DIR / "data"))
    # Хранилище дашбордов, черновиков и истории версий (отдельно от базы с данными,
    # потому что к базе с данными у бота доступ только на чтение).
    app_db_path: Path = field(default_factory=lambda: Path(_env("APP_DB") or BASE_DIR / "storage" / "app.db"))

    # Понимание запросов: rules — встроенный разборщик русского языка,
    # llm — только языковая модель, hybrid — сначала правила, непонятное — в модель.
    nlu_mode: str = field(default_factory=lambda: _env("NLU_MODE", "hybrid").lower())
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none").lower())  # none | openai | gigachat
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_sec: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_SEC", "40")))
    gigachat_auth_key: str = field(default_factory=lambda: _env("GIGACHAT_AUTH_KEY"))
    gigachat_scope: str = field(default_factory=lambda: _env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"))
    gigachat_model: str = field(default_factory=lambda: _env("GIGACHAT_MODEL", "GigaChat-2-Max"))
    gigachat_verify_ssl: bool = field(default_factory=lambda: _env_bool("GIGACHAT_VERIFY_SSL", False))

    # Ограничения безопасности
    query_timeout_sec: int = field(default_factory=lambda: int(_env("QUERY_TIMEOUT_SEC", "20")))
    max_rows: int = field(default_factory=lambda: int(_env("MAX_ROWS", "400000")))
    calc_timeout_sec: float = field(default_factory=lambda: float(_env("CALC_TIMEOUT_SEC", "10")))

    # «Сегодня» для относительных периодов («последний квартал»). Пусто — берётся
    # последняя дата продаж в базе, чтобы демо не зависело от текущей даты.
    anchor_date: str = field(default_factory=lambda: _env("ANCHOR_DATE"))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'sales.db').as_posix()}"

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "gigachat":
            return bool(self.gigachat_auth_key)
        if self.llm_provider == "openai":
            return bool(self.llm_api_key) or "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url
        return False


settings = Settings()
