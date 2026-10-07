"""Состояние дашборда и его отрисовка: KPI-карточки, фильтры, переключатели,
графики и текстовое резюме. Все элементы считаются из одной отфильтрованной
единой таблицы, поэтому фильтры и переключатели действуют на всё сразу."""
from __future__ import annotations

import copy
import html
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from . import fmt
from .calc import aggregate, bucket
from .charts import OTHER_COLOR, SERIES, ChartContext, build_chart, period_label
from .semantic import (DATASETS, DIM_LABELS, DIM_LABELS_PLURAL, DatasetSpec, Period,
                       column_label, month_label)

SWITCH_LABELS = {
    "dim": "Разрез",
    "metric": "Метрика",
    "grain": "Период",
    "view": "Вид",
}
GRAIN_OPTIONS = [("month", "Месяц"), ("quarter", "Квартал"), ("year", "Год")]
VIEW_OPTIONS = [("abs", "Абсолютные"), ("pct", "Проценты"), ("dyn", "Динамика")]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class DashboardState:
    dataset: str
    query: str
    period: Period
    table: pd.DataFrame
    prev: pd.DataFrame | None = None
    links: pd.DataFrame | None = None
    sql: str = ""
    name: str = ""
    charts: list[dict] = field(default_factory=list)
    calcs: list[dict] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    switches: dict[str, str] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    saved_id: str | None = None
    saved_version: int | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    custom_spec: dict | None = None
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # --- описание набора ---
    @property
    def spec(self) -> DatasetSpec:
        if self.dataset == "custom" and self.custom_spec:
            from .semantic import MetricFamily as MF

            cs = self.custom_spec
            metrics = {m: MF(m, column_label(m), m, None, "") for m in cs["measures"]}
            return DatasetSpec(key="custom", label="Произвольный запрос", tables=[], dims=cs["dims"],
                               measures=cs["measures"], derived=[], filters=[d for d in cs["dims"]][:4],
                               switch_dims=[d for d in cs["dims"] if d != "period"] or ["period"],
                               metrics=metrics or {"value": MF("value", "Значение", "value", None, "")},
                               default_metric=(cs["measures"] or ["value"])[0], description="SQL от языковой модели")
        return DATASETS[self.dataset]

    def log(self, text: str) -> None:
        self.history.append({"ts": now_iso(), "text": text})
        self.updated_at = now_iso()

    def clone(self) -> "DashboardState":
        """Копия метаданных; таблицы общие (они не изменяются на месте)."""
        other = copy.copy(self)
        other.charts = copy.deepcopy(self.charts)
        other.calcs = copy.deepcopy(self.calcs)
        other.filters = copy.deepcopy(self.filters)
        other.switches = dict(self.switches)
        other.history = list(self.history)
        other.custom_spec = copy.deepcopy(self.custom_spec)
        return other

    # --- сериализация (для сохранения) ---
    def meta(self) -> dict:
        return {
            "dataset": self.dataset, "query": self.query, "period": self.period.to_dict(), "sql": self.sql,
            "name": self.name, "charts": self.charts, "calcs": self.calcs, "filters": self.filters,
            "switches": self.switches, "history": self.history, "created_at": self.created_at,
            "updated_at": self.updated_at, "custom_spec": self.custom_spec, "uid": self.uid,
        }

    @classmethod
    def from_meta(cls, meta: dict, table: pd.DataFrame, prev: pd.DataFrame | None,
                  links: pd.DataFrame | None) -> "DashboardState":
        return cls(
            dataset=meta["dataset"], query=meta.get("query", ""), period=Period.from_dict(meta["period"]),
            table=table, prev=prev, links=links, sql=meta.get("sql", ""), name=meta.get("name", ""),
            charts=meta.get("charts", []), calcs=meta.get("calcs", []), filters=meta.get("filters", {}),
            switches=meta.get("switches", {}), history=meta.get("history", []),
            created_at=meta.get("created_at", now_iso()), updated_at=meta.get("updated_at", now_iso()),
            custom_spec=meta.get("custom_spec"), uid=meta.get("uid") or uuid.uuid4().hex[:12],
        )

    # --- фильтрация ---
    def filtered(self, frame: pd.DataFrame | None, *, ignore_period: bool = False) -> pd.DataFrame | None:
        if frame is None:
            return None
        mask = pd.Series(True, index=frame.index)
        for key, values in self.filters.items():
            if not values:
                continue
            if key == "period":
                if ignore_period or "period" not in frame.columns:
                    continue
                labels = frame["period"].dt.strftime("%Y-%m")
                mask &= labels.isin(values)
            elif key in frame.columns:
                mask &= frame[key].isin(values)
        return frame[mask]

    def filtered_prev(self) -> pd.DataFrame | None:
        if self.prev is None:
            return None
        prev = self.filtered(self.prev, ignore_period=True)
        months = self.filters.get("period")
        if months and prev is not None and len(prev):
            shifted = (pd.to_datetime([m + "-01" for m in months]) - pd.DateOffset(years=1)).strftime("%Y-%m")
            prev = prev[prev["period"].dt.strftime("%Y-%m").isin(set(shifted))]
        return prev

    def dim_columns(self) -> list[str]:
        return [d for d in self.spec.dims if d in self.table.columns]


# ---------------------------------------------------------------------------
# Шаблоны графиков под типовые запросы
# ---------------------------------------------------------------------------


def default_charts(dataset: str, intent: str, *, dim: str | None = None, structure_dim: str | None = None,
                   top_n: int | None = None, lagging: bool = False, sort_by: str | None = None) -> list[dict]:
    def chart(**kw: Any) -> dict:
        kw.setdefault("id", uuid.uuid4().hex[:8])
        return kw

    if dataset == "discounts":
        charts = [
            chart(type="trend", split="reason", mode="bar"),
            chart(type="structure", dim="reason", scope="all"),
            chart(type="bar_dim", dim="product", top_n=10),
            chart(type="trend", value="avg_pct", mode="line"),
        ]
        return charts if intent == "dashboard" else charts[:1] if intent == "chart" else charts
    if dataset == "clients":
        charts = [
            chart(type="bar_dim", dim=dim or "client", top_n=top_n or 10),
            chart(type="trend", mode="bar"),
            chart(type="structure", dim=structure_dim or "segment", scope="all"),
            chart(type="trend", split="segment", mode="line"),
        ]
        return charts if intent == "dashboard" else charts[:1]
    if dataset == "custom":
        return [chart(type="bar_dim", dim="@switch", top_n=top_n or 15)]

    # продажи: план-факт
    if intent == "compare":
        return [
            chart(type="top", dim="@switch", sort_by="exec_pct", sort_dir="asc", top_n=top_n or 15),
            chart(type="scatter_planfact", dim="@switch", threshold=90),
            chart(type="planfact_period"),
            chart(type="exec_line", variant="heatmap", dim="@switch"),
        ] if lagging else [
            chart(type="top", dim="@switch", top_n=top_n or 15, sort_by=sort_by),
            chart(type="scatter_planfact", dim="@switch", threshold=90),
            chart(type="planfact_period"),
            chart(type="exec_line", dim="@switch"),
        ]
    if intent == "structure":
        return [chart(type="structure", dim=structure_dim or "category", scope="last")]
    if intent == "planfact_dim":
        return [chart(type="planfact_dim", dim="@switch", top_n=top_n or 12)]
    if intent == "trend":
        return [chart(type="planfact_period")]
    if intent == "top":
        return [chart(type="top", dim="@switch", top_n=top_n or 10, sort_dir="asc" if lagging else "desc",
                      sort_by="exec_pct" if lagging else sort_by)]
    if intent == "exec":
        return [chart(type="exec_line", dim="@switch")]
    return [
        chart(type="planfact_period"),
        chart(type="structure", dim=structure_dim or "category", scope="last"),
        chart(type="top", dim="@switch", sort_by="deviation", sort_dir="asc" if lagging else "desc", top_n=top_n or 10),
        chart(type="exec_line", dim="@switch"),
    ]


# ---------------------------------------------------------------------------
# Цвета сущностей — закреплены за сущностью, а не за местом в рейтинге
# ---------------------------------------------------------------------------


def entity_order(state: DashboardState) -> dict[str, list[str]]:
    spec = state.spec
    fact = spec.metrics[spec.default_metric].fact
    orders: dict[str, list[str]] = {}
    table = state.table
    for dim in state.dim_columns():
        if dim == "period":
            continue
        if dim == "region":
            from .semantic import catalog

            try:
                orders[dim] = [r for r in catalog()["region"] if r in set(table[dim])]
                continue
            except Exception:  # noqa: BLE001
                pass
        if fact in table.columns:
            orders[dim] = list(table.groupby(dim)[fact].sum().sort_values(ascending=False).index.astype(str))
        else:
            orders[dim] = sorted(table[dim].astype(str).unique())
    return orders


def entity_colors(orders: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    colors = {}
    for dim, order in orders.items():
        mapping = {}
        for i, entity in enumerate(order):
            mapping[entity] = SERIES[i] if i < len(SERIES) else OTHER_COLOR
        colors[dim] = mapping
    return colors


# ---------------------------------------------------------------------------
# KPI-карточки
# ---------------------------------------------------------------------------


def _card(key: str, label: str, value: float | None, unit: str, *, status: str = "neutral", sub: str = "",
          display: str | None = None, arrow: str = "") -> dict:
    return {"key": key, "label": label, "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value),
            "display": display if display is not None else fmt.short(value, unit), "full": fmt.full(value, unit) if unit != "%" else fmt.pct(value),
            "unit": unit, "status": status, "sub": sub, "arrow": arrow}


def _status(value: float | None, good_if_positive: bool = True) -> tuple[str, str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == 0:
        return "neutral", ""
    positive = value > 0
    good = positive if good_if_positive else not positive
    return ("good" if good else "bad"), ("▲" if positive else "▼")


def compute_kpis(state: DashboardState, frame: pd.DataFrame, prev: pd.DataFrame | None) -> list[dict]:
    spec = state.spec
    fam = spec.metrics.get(state.switches.get("metric") or spec.default_metric) or spec.metrics[spec.default_metric]
    cards: list[dict] = []
    measures = [m for m in spec.measures if m in frame.columns]

    def total(col: str, source: pd.DataFrame | None = frame) -> float:
        if source is None or source.empty:
            return float("nan")
        if col in source.columns:
            return float(source[col].sum())
        agg = aggregate(source, [], [col], state.calcs, measures=measures)
        return float(agg[col].iloc[0]) if len(agg) else float("nan")

    fact = total(fam.fact)
    prev_fact = total(fam.fact, prev) if prev is not None and len(prev) else float("nan")
    yoy = (fact / prev_fact - 1) * 100 if prev_fact and not np.isnan(prev_fact) else float("nan")
    yoy_label, yoy_sub = "Динамика к прошлому году", None
    if np.isnan(yoy) and len(frame) and "period" in frame.columns:
        # длинный период без данных годом ранее: последние 12 мес. к предыдущим 12 внутри периода
        last = frame["period"].max()
        cur = frame[frame["period"] > last - pd.DateOffset(months=12)]
        before = frame[(frame["period"] <= last - pd.DateOffset(months=12)) & (frame["period"] > last - pd.DateOffset(months=24))]
        if before["period"].nunique() == 12:
            a, b = total(fam.fact, cur), total(fam.fact, before)
            if b:
                yoy = (a / b - 1) * 100
                yoy_label = "Последние 12 мес. к предыдущим"
                yoy_sub = f"было {fmt.short(b, fam.unit)}"

    if state.dataset == "sales" and fam.plan:
        plan = total(fam.plan)
        exec_pct = fact / plan * 100 if plan else float("nan")
        dev = fact - plan
        dev_pct = dev / plan * 100 if plan else float("nan")
        exec_status = "good" if exec_pct >= 100 else "bad"
        cards.append(_card("plan", f"План, {fam.unit}", plan, fam.unit))
        cards.append(_card("fact", f"Факт, {fam.unit}", fact, fam.unit))
        cards.append(_card("exec", "Выполнение плана", exec_pct, "%", status=exec_status,
                           display=fmt.pct(exec_pct), arrow="▲" if exec_pct >= 100 else "▼"))
        st, _ = _status(dev)
        cards.append(_card("deviation", "Отклонение от плана", dev, fam.unit, status=st,
                           display=fmt.signed_short(dev, fam.unit), sub=fmt.pct(dev_pct, signed=True)))
    elif state.dataset == "sales":
        base_fact = total("fact")
        cards.append(_card("value", fam.label, fact, fam.unit))
        if fam.key == "discount":
            share = fact / base_fact * 100 if base_fact else float("nan")
            cards.append(_card("share", "Доля скидок от продаж", share, "%", display=fmt.pct(share)))
            deals = total("deals")
            cards.append(_card("deals", "Сделки", deals, "шт"))
        else:
            avg_check = base_fact / fact if fact else float("nan")
            cards.append(_card("avg_check", "Средний чек", avg_check, "₽"))
            cards.append(_card("fact", "Факт, ₽", base_fact, "₽"))
    elif state.dataset == "discounts":
        cnt = total("discounts_cnt")
        avg_pct = total("pct_sum") / cnt if cnt else float("nan")
        share = fact / total("sale_amount") * 100 if total("sale_amount") else float("nan")
        cards.append(_card("value", "Сумма скидок" if fam.key == "discount" else fam.label, fact if fam.key != "avg_pct" else total("discount"), "₽"))
        cards.append(_card("count", "Количество скидок", cnt, "шт"))
        cards.append(_card("avg_pct", "Средний размер скидки", avg_pct, "%", display=fmt.pct(avg_pct)))
        cards.append(_card("share", "Скидки к сумме сделок", share, "%", display=fmt.pct(share)))
        if len(frame):
            top_reason = frame.groupby("reason")["discount"].sum().sort_values(ascending=False)
            reason_share = top_reason.iloc[0] / top_reason.sum() * 100 if top_reason.sum() else 0
            cards.append(_card("reason", "Главная причина", None, "", display=str(top_reason.index[0]),
                               sub=f"{fmt.pct(reason_share)} суммы"))
    elif state.dataset == "clients":
        deals = total("deals")
        cards.append(_card("value", fam.label, fact, fam.unit))
        cards.append(_card("deals", "Сделки", deals, "шт"))
        base_fact = total("fact")
        cards.append(_card("avg_check", "Средний чек", base_fact / deals if deals else float("nan"), "₽"))
        if len(frame):
            by_client = frame.groupby("client")["fact"].sum().sort_values(ascending=False)
            share = by_client.iloc[0] / by_client.sum() * 100 if by_client.sum() else 0
            cards.append(_card("top_client", "Крупнейший клиент", None, "", display=_cut(str(by_client.index[0]), 26),
                               sub=f"{fmt.pct(share)} продаж"))
    else:  # custom
        cards.append(_card("rows", "Строк", float(len(frame)), ""))
        for m in measures[:3]:
            cards.append(_card(m, column_label(m), total(m), ""))

    # Количество активных точек / клиентов / продуктов
    if state.dataset == "sales" and len(frame):
        active = frame[frame["fact"] > 0]
        outlets = active["outlet"].nunique()
        products = active["product"].nunique()
        clients = _active_clients(state)
        sub = f"клиентов {fmt.num(clients)} · продуктов {products}" if clients is not None else f"продуктов {products}"
        cards.append(_card("active", "Активные точки", float(outlets), "", display=fmt.num(outlets), sub=sub))
    elif state.dataset in ("discounts", "clients") and len(frame) and frame["client"].nunique() > 1:
        n = frame["client"].nunique()
        cards.append(_card("active", "Клиентов", float(n), "", display=fmt.num(n)))

    if state.dataset != "custom":
        st, _ = _status(yoy, good_if_positive=not (state.dataset == "discounts" or fam.key == "discount"))
        sub = yoy_sub or (f"было {fmt.short(prev_fact, fam.unit)}" if not np.isnan(prev_fact) else "нет данных за прошлый год")
        cards.append(_card("yoy", yoy_label, yoy, "%", status=st,
                           display=fmt.pct(yoy, signed=True), sub=sub))
    return cards


def _active_clients(state: DashboardState) -> int | None:
    if state.links is None:
        return None
    links = state.filtered(state.links)
    return int(links["client_id"].nunique()) if links is not None else None


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _e(value) -> str:
    return html.escape(str(value), quote=False)


def _trim(text: str, n: int) -> str:
    text = text if len(text) <= n else text[: n - 1] + "…"
    return html.escape(text, quote=False)


# ---------------------------------------------------------------------------
# Текстовое резюме
# ---------------------------------------------------------------------------


def summarize(state: DashboardState, frame: pd.DataFrame, prev: pd.DataFrame | None) -> list[str]:
    if frame.empty:
        return ["По выбранным фильтрам данных нет — снимите часть фильтров."]
    spec = state.spec
    fam = spec.metrics.get(state.switches.get("metric") or spec.default_metric) or spec.metrics[spec.default_metric]
    dim = state.switches.get("dim") or spec.switch_dims[0]
    grain = state.switches.get("grain", "month")
    out: list[str] = []
    fact = float(frame[fam.fact].sum()) if fam.fact in frame.columns else float("nan")

    if state.dataset == "sales" and fam.plan:
        plan = float(frame[fam.plan].sum())
        exec_pct = fact / plan * 100 if plan else float("nan")
        dev = fact - plan
        verb = "перевыполнен" if exec_pct >= 100 else "выполнен"
        out.append(f"План {verb} на <b>{fmt.pct(exec_pct)}</b>: факт {fmt.short(fact, fam.unit)} при плане "
                   f"{fmt.short(plan, fam.unit)} ({'запас' if dev >= 0 else 'отставание'} {fmt.short(abs(dev), fam.unit)}).")
        if dim in frame.columns:
            by = frame.groupby(dim)[[fam.fact, fam.plan]].sum()
            by = by[by[fam.plan] > 0]
            if len(by) > 1:
                by["exec"] = by[fam.fact] / by[fam.plan] * 100
                best, worst = by["exec"].idxmax(), by["exec"].idxmin()
                noun = {"region": "регион", "outlet": "точка", "category": "категория", "product": "продукт"}.get(dim, "")
                out.append(f"Лидер по выполнению — {_trim(str(best), 60)} ({fmt.pct(by.loc[best, 'exec'])}), "
                           f"{'слабее всего' if noun != 'точка' else 'хуже всего'} — {_trim(str(worst), 60)} ({fmt.pct(by.loc[worst, 'exec'])}).")
                lagging = int((by["exec"] < 100).sum())
                if len(by) >= 5:
                    word = DIM_LABELS_PLURAL.get(dim, "").lower()
                    out.append(f"План не выполнили {lagging} из {len(by)} ({word}).")
        per = frame.assign(period=bucket(frame["period"], grain)).groupby("period")[[fam.fact, fam.plan]].sum()
        per = per[per[fam.plan] > 0]
        if len(per) > 2:
            per["exec"] = per[fam.fact] / per[fam.plan] * 100
            b, w = per["exec"].idxmax(), per["exec"].idxmin()
            out.append(f"Лучший период по выполнению — {period_label(b, grain)} ({fmt.pct(per.loc[b, 'exec'])}), "
                       f"худший — {period_label(w, grain)} ({fmt.pct(per.loc[w, 'exec'])}).")
    elif state.dataset == "discounts":
        cnt = frame["discounts_cnt"].sum()
        avg = frame["pct_sum"].sum() / cnt if cnt else float("nan")
        out.append(f"Выдано {fmt.num(cnt)} {fmt.plural(cnt, 'скидка', 'скидки', 'скидок')} на "
                   f"{fmt.short(frame['discount'].sum(), '₽')}, средний размер — {fmt.pct(avg)}.")
        reasons = frame.groupby("reason")["discount"].sum().sort_values(ascending=False)
        if len(reasons):
            share = reasons.iloc[0] / reasons.sum() * 100
            out.append(f"Главная причина — «{_e(reasons.index[0])}» ({fmt.pct(share)} суммы)"
                       + (f", далее «{_e(reasons.index[1])}» ({fmt.pct(reasons.iloc[1] / reasons.sum() * 100)})." if len(reasons) > 1 else "."))
        prods = frame.groupby("product")["discount"].sum().sort_values(ascending=False)
        if len(prods):
            out.append(f"Больше всего скидок по продукту «{_e(prods.index[0])}» — {fmt.short(prods.iloc[0], '₽')}.")
        per = frame.assign(period=bucket(frame["period"], grain)).groupby("period")["discount"].sum()
        if len(per) > 2:
            out.append(f"Пик скидок — {period_label(per.idxmax(), grain)} ({fmt.short(per.max(), '₽')}).")
    elif state.dataset == "clients":
        out.append(f"Продажи {fmt.short(frame['fact'].sum(), '₽')} по {frame['client'].nunique()} клиентам, "
                   f"{fmt.num(frame['deals'].sum())} сделок.")
        top = frame.groupby("client")["fact"].sum().sort_values(ascending=False)
        if len(top) > 1:
            share5 = top.head(5).sum() / top.sum() * 100
            out.append(f"Крупнейший клиент — {_e(top.index[0])} ({fmt.short(top.iloc[0], '₽')}); на ТОП-5 приходится {fmt.pct(share5)} продаж.")
        seg = frame.groupby("segment")["fact"].sum().sort_values(ascending=False)
        if len(seg):
            out.append(f"Ведущий сегмент — {_e(seg.index[0])} ({fmt.pct(seg.iloc[0] / seg.sum() * 100)}).")
    elif state.dataset == "sales":
        out.append(f"{fam.label.split(',')[0]}: {fmt.short(fact, fam.unit)} за период.")
        if dim in frame.columns:
            top = frame.groupby(dim)[fam.fact].sum().sort_values(ascending=False)
            if len(top):
                out.append(f"Больше всего — {_trim(str(top.index[0]), 60)} ({fmt.short(top.iloc[0], fam.unit)}).")
    else:
        out.append(f"Строк в результате: {len(frame)}.")

    if prev is not None and len(prev) and fam.fact in prev.columns and state.dataset != "custom":
        prev_fact = float(prev[fam.fact].sum())
        if prev_fact:
            yoy = (fact / prev_fact - 1) * 100
            out.append(f"К тому же периоду прошлого года: <b>{fmt.pct(yoy, signed=True)}</b> "
                       f"({fmt.short(prev_fact, fam.unit)} → {fmt.short(fact, fam.unit)}).")
    if state.dataset == "sales" and "category" in frame.columns and fam.fact in frame.columns:
        cats = frame.groupby("category")[fam.fact].sum().sort_values(ascending=False)
        if len(cats) > 1 and cats.sum():
            out.append(f"Крупнейшая категория — {_e(cats.index[0])}: {fmt.pct(cats.iloc[0] / cats.sum() * 100)} объёма.")
    return out


# ---------------------------------------------------------------------------
# Фильтры и переключатели
# ---------------------------------------------------------------------------


def filter_panel(state: DashboardState) -> list[dict]:
    table = state.table
    panel = []
    for key in state.spec.filters:
        if key not in table.columns:
            continue
        if key == "period":
            months = sorted(table["period"].dropna().unique())
            options = [{"value": pd.Timestamp(m).strftime("%Y-%m"), "label": month_label(pd.Timestamp(m))} for m in months]
            panel.append({"key": key, "label": "Период", "options": options, "selected": state.filters.get(key, []),
                          "placeholder": "Все месяцы"})
            continue
        values = sorted(table[key].dropna().astype(str).unique())
        item = {"key": key, "label": DIM_LABELS.get(key, key), "options": [{"value": v, "label": v} for v in values],
                "selected": state.filters.get(key, []), "placeholder": f"Все ({len(values)})"}
        parent = {"outlet": "region", "product": "category"}.get(key)
        if parent and parent in table.columns and parent in state.spec.filters:
            pairs = table[[key, parent]].drop_duplicates()
            item["parent"] = parent
            item["parent_map"] = dict(zip(pairs[key].astype(str), pairs[parent].astype(str)))
        panel.append(item)
    return panel


def switch_panel(state: DashboardState) -> list[dict]:
    spec = state.spec
    dims = [d for d in spec.switch_dims if d in state.table.columns]
    current = state.switches.get("dim")
    if current and current not in dims and current in state.table.columns and current != "period":
        dims.append(current)  # разрез, выбранный в чате («по городам»), тоже виден на панели
    panel = []
    if dims:
        panel.append({"key": "dim", "label": "Разрез", "value": state.switches.get("dim"),
                      "options": [{"value": d, "label": DIM_LABELS_PLURAL.get(d, d)} for d in dims]})
    if len(spec.metrics) > 1:
        panel.append({"key": "metric", "label": "Метрика", "value": state.switches.get("metric"),
                      "options": [{"value": k, "label": m.label.split(",")[0]} for k, m in spec.metrics.items()]})
    if "period" in state.table.columns:
        panel.append({"key": "grain", "label": "Период", "value": state.switches.get("grain", "month"),
                      "options": [{"value": v, "label": lbl} for v, lbl in GRAIN_OPTIONS]})
        if state.dataset == "sales":
            panel.append({"key": "view", "label": "Вид", "value": state.switches.get("view", "abs"),
                          "options": [{"value": v, "label": lbl} for v, lbl in VIEW_OPTIONS]})
            fam = spec.metrics.get(state.switches.get("metric") or spec.default_metric)
            if fam and fam.plan:
                panel.append({"key": "series", "label": "Ряды", "value": state.switches.get("series", "both"),
                              "options": [{"value": "both", "label": "План и факт"}, {"value": "plan", "label": "План"},
                                          {"value": "fact", "label": "Факт"}]})
    return panel


def default_switches(state: DashboardState, dim: str | None = None, metric: str | None = None,
                     grain: str | None = None) -> dict[str, str]:
    spec = state.spec
    dims = [d for d in spec.switch_dims if d in state.table.columns]
    months = state.period.months
    auto_grain = "month" if months <= 24 else "quarter"
    return {
        "dim": dim if dim in dims else (dims[0] if dims else "period"),
        "metric": metric if metric in spec.metrics else spec.default_metric,
        "grain": grain or auto_grain,
        "view": "abs",
    }


# ---------------------------------------------------------------------------
# Отрисовка целиком
# ---------------------------------------------------------------------------


def render(state: DashboardState) -> dict:
    frame = state.filtered(state.table)
    prev = state.filtered_prev()
    orders = entity_order(state)
    ctx = ChartContext(state.spec, frame, prev, state.calcs, state.switches, entity_colors(orders), orders)
    charts = []
    for i, chart in enumerate(state.charts):
        built = build_chart(ctx, chart)
        charts.append({"id": chart["id"], "index": i + 1, "type": chart["type"], "title": built.get("title"),
                       "figure": built["figure"], "table": built.get("table"), "x_shared": built.get("x_shared"),
                       "value": chart.get("value"), "error": built.get("error")})
    active_filters = {k: v for k, v in state.filters.items() if v}
    return {
        "uid": state.uid,
        "name": state.name,
        "dataset": state.dataset,
        "dataset_label": state.spec.label,
        "query": state.query,
        "period": state.period.to_dict(),
        "period_label": f"{state.period.range_label()}",
        "kpis": compute_kpis(state, frame, prev),
        "filters": filter_panel(state),
        "active_filters": active_filters,
        "switches": switch_panel(state),
        "charts": charts,
        "calcs": [{"id": c["id"], "label": c["label"], "formula": c.get("formula", ""), "kind": c["kind"]} for c in state.calcs],
        "summary": summarize(state, frame, prev),
        "history": state.history[-15:],
        "rows_total": int(len(state.table)),
        "rows_filtered": int(len(frame)),
        "saved": {"id": state.saved_id, "version": state.saved_version} if state.saved_id else None,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }
