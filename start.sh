#!/usr/bin/env bash
# Запуск на macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/installed.ok ]; then
  echo "Первый запуск: создаю окружение и устанавливаю библиотеки…"
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/installed.ok
fi
if [ -f .env ] && grep -q '^DATABASE_URL=postgresql' .env; then
  .venv/bin/python -m pip install -q -r requirements-postgres.txt
fi
exec .venv/bin/python -m app.main
