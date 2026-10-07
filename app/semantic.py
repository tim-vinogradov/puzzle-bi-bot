"""Семантический слой: какие наборы данных есть, из чего они собираются,
как называются колонки по-русски и как превратить «последний квартал» в даты.

Бот не пишет SQL «с нуля» для типовых вопросов: он выбирает набор данных и
период, а конструктор ниже собирает проверенный SQL. Для нестандартных вопросов
языковая модель может предложить свой SELECT — он проходит `sql_guard`.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from . import db
from .config import settings

MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
              "сентябрь", "октябрь", "ноябрь", "декабрь"]
MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

# ---------------------------------------------------------------------------
# Колонки и их русские названия
# ---------------------------------------------------------------------------

DIM_LABELS = {
    "period": "Период",
    "region": "Регион",
    "city": "Город",
    "outlet": "Торговая точка",
    "outlet_type": "Тип точки",
    "category": "Категория",
    "product": "Продукт",
    "client": "Клиент",
    "segment": "Сегмент клиента",
    "reason": "Причина скидки",
}

DIM_LABELS_PLURAL = {
    "region": "Регионы", "city": "Города", "outlet": "Точки", "outlet_type": "Типы точек",
    "category": "Категории", "product": "Продукты", "client": "Клиенты", "segment": "Сегменты",
    "reason": "Причины",
}

# Родительный падеж множественного числа — «ТОП регионов», «доля продуктов»
DIM_GENITIVE_PLURAL = {
    "region": "регионов", "city": "городов", "outlet": "точек", "outlet_type": "типов точек",
    "category": "категорий", "product": "продуктов", "client": "клиентов", "segment": "сегментов",
    "reason": "причин",
}

MEASURE_META: dict[str, dict[str, str]] = {
    "plan": {"label": "План, ₽", "unit": "₽"},
    "fact": {"label": "Факт, ₽", "unit": "₽"},
    "plan_qty": {"label": "План, шт", "unit": "шт"},
    "qty": {"label": "Факт, шт", "unit": "шт"},
    "discount": {"label": "Скидка, ₽", "unit": "₽"},
    "deals": {"label": "Сделки, шт", "unit": "шт"},
    "discounts_cnt": {"label": "Кол-во скидок", "unit": "шт"},
    "pct_sum": {"label": "Сумма % скидок (служебная)", "unit": ""},
    "sale_amount": {"label": "Сумма сделок со скидкой, ₽", "unit": "₽"},
}

# Производные показатели: считаются из сумм после агрегации (как calculated field в BI)
DERIVED: dict[str, dict[str, str]] = {
    "exec_pct": {"expr": "fact / plan * 100", "label": "Выполнение плана, %", "unit": "%"},
    "deviation": {"expr": "fact - plan", "label": "Отклонение от плана, ₽", "unit": "₽"},
    "deviation_pct": {"expr": "(fact - plan) / plan * 100", "label": "Отклонение от плана, %", "unit": "%"},
    "exec_qty_pct": {"expr": "qty / plan_qty * 100", "label": "Выполнение плана в шт, %", "unit": "%"},
    "avg_check": {"expr": "fact / deals", "label": "Средний чек, ₽", "unit": "₽"},
    "discount_share": {"expr": "discount / fact * 100", "label": "Доля скидок от продаж, %", "unit": "%"},
    "avg_pct": {"expr": "pct_sum / discounts_cnt", "label": "Средний размер скидки, %", "unit": "%"},
    "discount_to_sales": {"expr": "discount / sale_amount * 100", "label": "Скидки к сумме сделок, %", "unit": "%"},
}


@dataclass
class MetricFamily:
    key: str
    label: str
    fact: str
    plan: str | None
    unit: str


@dataclass
class DatasetSpec:
    key: str
    label: str
    tables: list[str]
    dims: list[str]
    measures: list[str]
    derived: list[str]
    filters: list[str]
    switch_dims: list[str]
    metrics: dict[str, MetricFamily]
    default_metric: str
    description: str
    extra: dict[str, Any] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "sales": DatasetSpec(
        key="sales",
        label="Продажи: план и факт",
        tables=["sales", "sales_plan", "discounts", "outlets", "regions", "products"],
        dims=["period", "region", "city", "outlet", "outlet_type", "category", "product"],
        measures=["plan", "fact", "plan_qty", "qty", "discount", "deals"],
        derived=["exec_pct", "deviation", "deviation_pct", "exec_qty_pct", "avg_check", "discount_share"],
        filters=["period", "region", "outlet", "category", "product"],
        switch_dims=["region", "outlet", "category", "product"],
        metrics={
            "amount": MetricFamily("amount", "Объём, ₽", "fact", "plan", "₽"),
            "qty": MetricFamily("qty", "Количество, шт", "qty", "plan_qty", "шт"),
            "discount": MetricFamily("discount", "Скидка, ₽", "discount", None, "₽"),
            "deals": MetricFamily("deals", "Сделки", "deals", None, "шт"),
        },
        default_metric="amount",
        description="Месяц × торговая точка × продукт: план, факт, количество, скидки, число сделок",
    ),
    "discounts": DatasetSpec(
        key="discounts",
        label="Скидки",
        tables=["discounts", "sales", "clients", "outlets", "regions", "products"],
        dims=["period", "client", "segment", "region", "category", "product", "reason"],
        measures=["discount", "discounts_cnt", "pct_sum", "sale_amount"],
        derived=["avg_pct", "discount_to_sales"],
        filters=["period", "client", "reason", "region", "category"],
        switch_dims=["reason", "category", "product", "region", "client"],
        metrics={
            "discount": MetricFamily("discount", "Сумма скидок, ₽", "discount", None, "₽"),
            "count": MetricFamily("count", "Кол-во скидок", "discounts_cnt", None, "шт"),
            "avg_pct": MetricFamily("avg_pct", "Средний % скидки", "avg_pct", None, "%"),
        },
        default_metric="discount",
        description="Месяц × клиент × продукт × причина: сумма и количество скидок",
    ),
    "clients": DatasetSpec(
        key="clients",
        label="Продажи по клиентам",
        tables=["sales", "discounts", "clients", "outlets", "regions", "products"],
        dims=["period", "client", "segment", "region", "category", "product"],
        measures=["fact", "qty", "deals", "discount"],
        derived=["avg_check", "discount_share"],
        filters=["period", "client", "segment", "region", "category"],
        switch_dims=["client", "segment", "region", "category", "product"],
        metrics={
            "amount": MetricFamily("amount", "Объём, ₽", "fact", None, "₽"),
            "qty": MetricFamily("qty", "Количество, шт", "qty", None, "шт"),
            "discount": MetricFamily("discount", "Скидка, ₽", "discount", None, "₽"),
            "deals": MetricFamily("deals", "Сделки", "deals", None, "шт"),
        },
        default_metric="amount",
        description="Месяц × клиент × продукт: продажи, количество, сделки, скидки",
    ),
}


def column_label(col: str, calcs: list[dict] | None = None) -> str:
    if col in DIM_LABELS:
        return DIM_LABELS[col]
    if col in MEASURE_META:
        return MEASURE_META[col]["label"]
    if col in DERIVED:
        return DERIVED[col]["label"]
    for calc in calcs or []:
        if calc["id"] == col:
            return calc["label"]
    return col


def column_unit(col: str, calcs: list[dict] | None = None) -> str:
    if col in MEASURE_META:
        return MEASURE_META[col]["unit"]
    if col in DERIVED:
        return DERIVED[col]["unit"]
    for calc in calcs or []:
        if calc["id"] == col:
            return calc.get("unit", "")
    return ""


# ---------------------------------------------------------------------------
# Периоды
# ---------------------------------------------------------------------------


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def add_months(d: date, months: int) -> date:
    idx = d.year * 12 + d.month - 1 + months
    year, month = divmod(idx, 12)
    day = min(d.day, calendar.monthrange(year, month + 1)[1])
    return date(year, month + 1, day)


def month_label(d: date | pd.Timestamp) -> str:
    return f"{MONTHS_SHORT[d.month - 1]} {d.year}"


@dataclass
class Period:
    start: date
    end: date
    label: str
    kind: str = "custom"  # last_quarter, last_year, ytd, year, quarter, month, all, custom

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "label": self.label, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict) -> "Period":
        return cls(date.fromisoformat(data["start"]), date.fromisoformat(data["end"]), data["label"], data.get("kind", "custom"))

    @property
    def months(self) -> int:
        return (self.end.year - self.start.year) * 12 + self.end.month - self.start.month + 1

    def range_label(self) -> str:
        if self.months == 1:
            return month_label(self.start)
        return f"{month_label(self.start)} – {month_label(self.end)}"

    def comparison(self) -> "Period":
        """Тот же период годом ранее — для динамики «к прошлому году»."""
        start = add_months(self.start, -12)
        end = month_end(add_months(month_start(self.end), -12))
        return Period(start, end, f"{self.label} (год назад)", "comparison")


def anchor_date() -> date:
    if settings.anchor_date:
        return date.fromisoformat(settings.anchor_date)
    return db.data_bounds()[1].date()


def last_complete_month(anchor: date | None = None) -> date:
    anchor = anchor or anchor_date()
    if anchor == month_end(anchor):
        return month_start(anchor)
    return month_start(add_months(month_start(anchor), -1))


def period_last_months(n: int, anchor: date | None = None, label: str | None = None, kind: str = "custom") -> Period:
    last = last_complete_month(anchor)
    start = add_months(last, -(n - 1))
    return Period(start, month_end(last), label or f"последние {n} мес.", kind)


def period_last_quarter(anchor: date | None = None) -> Period:
    last = last_complete_month(anchor)
    # последний полностью закрытый квартал
    if last.month % 3 == 0:
        year, q_end_month = last.year, last.month
    elif last.month < 3:
        year, q_end_month = last.year - 1, 12
    else:
        year, q_end_month = last.year, last.month - last.month % 3
    start = date(year, q_end_month - 2, 1)
    q = (q_end_month - 1) // 3 + 1
    return Period(start, month_end(date(year, q_end_month, 1)), f"{q} квартал {year}", "last_quarter")


def period_quarter(q: int, year: int) -> Period:
    start = date(year, 3 * (q - 1) + 1, 1)
    return Period(start, month_end(date(year, 3 * q, 1)), f"{q} квартал {year}", "quarter")


def period_year(year: int) -> Period:
    return Period(date(year, 1, 1), date(year, 12, 31), f"{year} год", "year")


def period_month(month: int, year: int) -> Period:
    start = date(year, month, 1)
    return Period(start, month_end(start), f"{MONTHS_NOM[month - 1]} {year}", "month")


def period_ytd(anchor: date | None = None) -> Period:
    last = last_complete_month(anchor)
    return Period(date(last.year, 1, 1), month_end(last), f"с начала {last.year} года", "ytd")


def period_all() -> Period:
    dmin, dmax = db.data_bounds()
    last = last_complete_month()
    return Period(month_start(dmin.date()), month_end(last), "всё время", "all")


def clamp_to_data(period: Period) -> Period:
    """Обрезает период по фактически имеющимся данным (факт есть не дальше последнего месяца)."""
    dmin, _ = db.data_bounds()
    last = last_complete_month()
    start = max(period.start, month_start(dmin.date()))
    end = min(period.end, month_end(last))
    if end < start:
        return period
    return Period(start, end, period.label, period.kind)


# ---------------------------------------------------------------------------
# Конструктор SQL
# ---------------------------------------------------------------------------


def _month_expr(column: str, dialect: str) -> str:
    if dialect == "postgresql":
        return f"CAST(date_trunc('month', {column}) AS date)"
    return f"(substr({column}, 1, 7) || '-01')"


def build_sql(dataset: str, dialect: str) -> str:
    m_sale = _month_expr("s.sale_date", dialect)
    m_disc = _month_expr("dc.discount_date", dialect)
    m_plan = _month_expr("sp.plan_month", dialect)
    if dataset == "sales":
        return f"""
WITH f AS (
    SELECT {m_sale} AS period, s.outlet_id, s.product_id,
           SUM(s.amount) AS fact, SUM(s.quantity) AS qty, COUNT(*) AS deals
    FROM sales s
    WHERE s.sale_date BETWEEN :start AND :end
    GROUP BY {m_sale}, s.outlet_id, s.product_id
), p AS (
    SELECT {m_plan} AS period, sp.outlet_id, sp.product_id,
           SUM(sp.plan_amount) AS plan, SUM(sp.plan_qty) AS plan_qty
    FROM sales_plan sp
    WHERE sp.plan_month BETWEEN :start AND :end
    GROUP BY {m_plan}, sp.outlet_id, sp.product_id
), d AS (
    SELECT {m_sale} AS period, s.outlet_id, s.product_id, SUM(dc.discount_amount) AS discount
    FROM discounts dc
    JOIN sales s ON s.sale_id = dc.sale_id
    WHERE s.sale_date BETWEEN :start AND :end
    GROUP BY {m_sale}, s.outlet_id, s.product_id
), k AS (
    SELECT period, outlet_id, product_id,
           SUM(plan) AS plan, SUM(plan_qty) AS plan_qty,
           SUM(fact) AS fact, SUM(qty) AS qty, SUM(deals) AS deals
    FROM (
        SELECT period, outlet_id, product_id, plan, plan_qty, 0 AS fact, 0 AS qty, 0 AS deals FROM p
        UNION ALL
        SELECT period, outlet_id, product_id, 0, 0, fact, qty, deals FROM f
    ) u
    GROUP BY period, outlet_id, product_id
)
SELECT k.period, r.region_name AS region, o.city, o.outlet_name AS outlet, o.outlet_type,
       pr.category, pr.product_name AS product,
       k.plan, k.fact, k.plan_qty, k.qty, COALESCE(d.discount, 0) AS discount, k.deals
FROM k
JOIN outlets o ON o.outlet_id = k.outlet_id
JOIN regions r ON r.region_id = o.region_id
JOIN products pr ON pr.product_id = k.product_id
LEFT JOIN d ON d.period = k.period AND d.outlet_id = k.outlet_id AND d.product_id = k.product_id
ORDER BY k.period, r.region_name, o.outlet_name, pr.product_name
""".strip()
    if dataset == "sales_links":
        return f"""
SELECT {m_sale} AS period, r.region_name AS region, o.outlet_name AS outlet,
       pr.category, pr.product_name AS product, s.client_id AS client_id
FROM sales s
JOIN outlets o ON o.outlet_id = s.outlet_id
JOIN regions r ON r.region_id = o.region_id
JOIN products pr ON pr.product_id = s.product_id
WHERE s.sale_date BETWEEN :start AND :end
GROUP BY {m_sale}, r.region_name, o.outlet_name, pr.category, pr.product_name, s.client_id
""".strip()
    if dataset == "discounts":
        return f"""
SELECT {m_disc} AS period, c.client_name AS client, c.segment, r.region_name AS region,
       pr.category, pr.product_name AS product, dc.reason,
       SUM(dc.discount_amount) AS discount, COUNT(*) AS discounts_cnt,
       SUM(dc.discount_pct) AS pct_sum, SUM(s.amount) AS sale_amount
FROM discounts dc
JOIN sales s ON s.sale_id = dc.sale_id
JOIN clients c ON c.client_id = dc.client_id
JOIN outlets o ON o.outlet_id = s.outlet_id
JOIN regions r ON r.region_id = o.region_id
JOIN products pr ON pr.product_id = s.product_id
WHERE dc.discount_date BETWEEN :start AND :end
GROUP BY {m_disc}, c.client_name, c.segment, r.region_name, pr.category, pr.product_name, dc.reason
ORDER BY 1, 2
""".strip()
    if dataset == "clients":
        return f"""
WITH d AS (
    SELECT dc.sale_id, SUM(dc.discount_amount) AS discount FROM discounts dc GROUP BY dc.sale_id
)
SELECT {m_sale} AS period, c.client_name AS client, c.segment, r.region_name AS region,
       pr.category, pr.product_name AS product,
       SUM(s.amount) AS fact, SUM(s.quantity) AS qty, COUNT(*) AS deals,
       COALESCE(SUM(d.discount), 0) AS discount
FROM sales s
JOIN clients c ON c.client_id = s.client_id
JOIN outlets o ON o.outlet_id = s.outlet_id
JOIN regions r ON r.region_id = o.region_id
JOIN products pr ON pr.product_id = s.product_id
LEFT JOIN d ON d.sale_id = s.sale_id
WHERE s.sale_date BETWEEN :start AND :end
GROUP BY {m_sale}, c.client_name, c.segment, r.region_name, pr.category, pr.product_name
ORDER BY 1, 2
""".strip()
    raise ValueError(f"Неизвестный набор данных: {dataset}")


def _normalize(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if frame.empty and len(frame.columns) == 0:
        return frame
    frame = frame.copy()
    if "period" in frame.columns:
        frame["period"] = pd.to_datetime(frame["period"].astype(str).str[:10])
    spec = DATASETS.get(dataset)
    if spec:
        for col in spec.measures:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)
        for col in spec.dims:
            if col in frame.columns and col != "period":
                frame[col] = frame[col].astype(object).where(frame[col].notna(), "—").astype(str)
    return frame


def _params(period: Period, dialect: str) -> dict:
    if dialect == "postgresql":
        return {"start": period.start, "end": period.end}
    return {"start": period.start.isoformat(), "end": period.end.isoformat()}


@dataclass
class FetchResult:
    table: pd.DataFrame
    prev: pd.DataFrame | None
    links: pd.DataFrame | None
    sql: str
    seconds: float


def fetch_dataset(dataset: str, period: Period) -> FetchResult:
    """Выгружает единую плоскую таблицу набора за период + тот же период год назад."""
    import time

    dialect = db.dialect_name()
    sql = build_sql(dataset, dialect)
    started = time.monotonic()
    # Собственный SQL тоже проходит проверку (validate), хотя и собран нами.
    table = _normalize(db.run_select(sql, _params(period, dialect)), dataset)
    comparison = period.comparison()
    prev = None
    if comparison.start >= month_start(db.data_bounds()[0].date()):
        prev = _normalize(db.run_select(sql, _params(comparison, dialect)), dataset)
    links = None
    if dataset == "sales":
        links = _normalize(db.run_select(build_sql("sales_links", dialect), _params(period, dialect)), "sales")
        links["client_id"] = links["client_id"].astype(int)
    return FetchResult(table, prev, links, sql, time.monotonic() - started)


def fetch_custom(sql: str) -> tuple[pd.DataFrame, str]:
    """Выполняет SELECT, предложенный языковой моделью (после проверки безопасности)."""
    frame = db.run_select(sql)
    for col in frame.columns:
        if frame[col].dtype == object:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().all() and len(frame):
                frame[col] = converted
    return frame, sql


# ---------------------------------------------------------------------------
# Справочники для распознавания сущностей в тексте
# ---------------------------------------------------------------------------

_catalog_cache: dict[str, list[str]] = {}


def catalog() -> dict[str, list[str]]:
    if not _catalog_cache:
        queries = {
            "region": "SELECT region_name AS v FROM regions ORDER BY region_id",
            "category": "SELECT DISTINCT category AS v FROM products ORDER BY 1",
            "product": "SELECT product_name AS v FROM products ORDER BY product_id",
            "outlet": "SELECT outlet_name AS v FROM outlets ORDER BY outlet_id",
            "city": "SELECT DISTINCT city AS v FROM outlets ORDER BY 1",
            "outlet_type": "SELECT DISTINCT outlet_type AS v FROM outlets ORDER BY 1",
            "client": "SELECT client_name AS v FROM clients ORDER BY client_id",
            "segment": "SELECT DISTINCT segment AS v FROM clients ORDER BY 1",
            "reason": "SELECT DISTINCT reason AS v FROM discounts ORDER BY 1",
        }
        for key, sql in queries.items():
            _catalog_cache[key] = [str(v) for v in db.run_select(sql, trusted=True)["v"].tolist()]
    return _catalog_cache


def outlet_regions() -> dict[str, str]:
    frame = db.run_select(
        "SELECT o.outlet_name AS outlet, r.region_name AS region FROM outlets o JOIN regions r ON r.region_id = o.region_id",
        trusted=True,
    )
    return dict(zip(frame["outlet"], frame["region"]))


def product_categories() -> dict[str, str]:
    frame = db.run_select("SELECT product_name AS product, category FROM products", trusted=True)
    return dict(zip(frame["product"], frame["category"]))


def schema_description() -> str:
    """Описание схемы для языковой модели."""
    from .loader import TABLES

    lines = []
    for table, cols in TABLES.items():
        lines.append(f"{table}(" + ", ".join(c[0] for c in cols) + ")")
    return "\n".join(lines)


def last_day(d: date) -> date:
    return month_end(d)


def days_between(a: date, b: date) -> int:
    return (b - a).days + 1


def shift_days(d: date, n: int) -> date:
    return d + timedelta(days=n)
