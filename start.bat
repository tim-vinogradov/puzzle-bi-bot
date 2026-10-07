@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Puzzle BI-bot

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo  Не найден Python 3.10 или новее.
  echo  Установите его с https://www.python.org/downloads/ ^(при установке отметьте "Add python.exe to PATH"^),
  echo  затем снова запустите start.bat.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\installed.ok" (
  echo  Первый запуск: создаю окружение и устанавливаю библиотеки. Это займёт 2-4 минуты...
  if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
  if not exist ".venv\Scripts\python.exe" goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
  echo ok> ".venv\installed.ok"
)

if exist ".env" (
  findstr /b /c:"DATABASE_URL=postgresql" ".env" >nul 2>nul && ".venv\Scripts\python.exe" -m pip install -q -r requirements-postgres.txt
)

echo.
echo  Запускаю бота. Браузер откроется сам: http://127.0.0.1:8000
echo  Чтобы остановить — закройте это окно или нажмите Ctrl+C.
echo.
".venv\Scripts\python.exe" -m app.main
pause
exit /b 0

:fail
echo.
echo  Не удалось установить библиотеки. Проверьте интернет и запустите start.bat ещё раз.
echo  Если ошибка повторится — удалите папку .venv и попробуйте снова.
pause
exit /b 1
