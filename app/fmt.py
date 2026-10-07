"""Форматирование чисел по-русски: 1 087 500 000 ₽, 1,09 млрд ₽, 87,0%."""
from __future__ import annotations

import math

NBSP = " "
THIN = " "


def _is_missing(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except TypeError:
        return True


def num(value, digits: int = 0) -> str:
    if _is_missing(value):
        return "—"
    text = f"{value:,.{digits}f}"
    return text.replace(",", NBSP).replace(".", ",")


def full(value, unit: str = "") -> str:
    """Полное число с разделителями разрядов."""
    if _is_missing(value):
        return "—"
    digits = 0 if unit in ("₽", "шт", "") and abs(value) >= 100 else (1 if unit == "%" else 2)
    if unit == "%":
        return pct(value)
    text = num(value, digits if not float(value).is_integer() else 0)
    return f"{text}{NBSP}{unit}" if unit else text


def short(value, unit: str = "") -> str:
    """Короткая запись: 1,09 млрд ₽, 162,5 млн ₽, 45,2 тыс шт."""
    if _is_missing(value):
        return "—"
    if unit == "%":
        return pct(value)
    sign = "−" if value < 0 else ""
    v = abs(value)
    for threshold, word in ((1e9, "млрд"), (1e6, "млн"), (1e3, "тыс.")):
        if v >= threshold:
            scaled = v / threshold
            digits = 2 if scaled < 10 else 1
            text = f"{scaled:.{digits}f}".replace(".", ",")
            if "," in text:
                text = text.rstrip("0").rstrip(",")
            return f"{sign}{text}{NBSP}{word}{(NBSP + unit) if unit else ''}"
    digits = 0 if v >= 100 or float(v).is_integer() else 1
    return f"{sign}{num(v, digits)}{(NBSP + unit) if unit else ''}"


def pct(value, digits: int = 1, signed: bool = False) -> str:
    if _is_missing(value):
        return "—"
    text = f"{abs(value):.{digits}f}".replace(".", ",")
    sign = ""
    if value < 0:
        sign = "−"
    elif signed and value > 0:
        sign = "+"
    return f"{sign}{text}%"


def signed_short(value, unit: str = "") -> str:
    if _is_missing(value):
        return "—"
    base = short(value, unit)
    return base if value < 0 else f"+{base}"


def axis_scale(max_abs: float) -> tuple[float, str]:
    """Масштаб для осей: делитель и подпись («млн»)."""
    if max_abs >= 1e9:
        return 1e9, "млрд"
    if max_abs >= 1e6:
        return 1e6, "млн"
    if max_abs >= 1e4:
        return 1e3, "тыс."
    return 1.0, ""


def plural(n: int, one: str, few: str, many: str) -> str:
    n_abs = abs(int(n)) % 100
    if 11 <= n_abs <= 14:
        return many
    last = n_abs % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many
