"""Построение графиков Plotly в едином стиле.

Каждый график строится из одной и той же (отфильтрованной) единой таблицы.
Цвета закреплены за сущностями (регион «Центральный» всегда одного цвета),
план — нейтральный серый, факт — синий, статус «выполнен/не выполнен» —
зелёный/красный с подписью в легенде (не только цветом).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from . import fmt
from .calc import aggregate, bucket
from .semantic import (DERIVED, DIM_GENITIVE_PLURAL, DIM_LABELS, DatasetSpec, MetricFamily,
                       column_label, column_unit, month_label)

# --- Палитра (проверена валидатором на различимость при дальтонизме) ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PLAN_COLOR = "#b3b1a8"
FACT_COLOR = "#2a78d6"
GOOD = "#0ca30c"
BAD = "#d03b3b"
NEUTRAL = "#9c9a92"
OTHER_COLOR = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif'

MAX_LINES = 8
MAX_SLICES = 7


@dataclass
class ChartContext:
    spec: DatasetSpec
    frame: pd.DataFrame
    prev: pd.DataFrame | None
    calcs: list[dict]
    switches: dict
    colors: dict[str, dict[str, str]]
    entity_order: dict[str, list[str]] = field(default_factory=dict)

    @property
    def measures(self) -> list[str]:
        return self.spec.measures if self.spec.measures else [c for c in self.frame.columns if c != "period"]

    def agg(self, by: list[str], needed: list[str], grain: str | None = None) -> pd.DataFrame:
        return aggregate(self.frame, by, needed, self.calcs, measures=self.measures,
                         grain=grain or self.switches.get("grain", "month"), prev=self.prev)

    def family(self, chart: dict) -> MetricFamily:
        key = chart.get("metric") or self.switches.get("metric") or self.spec.default_metric
        return self.spec.metrics.get(key) or self.spec.metrics[self.spec.default_metric]

    def dim(self, chart: dict, default: str | None = None) -> str:
        dim = chart.get("dim")
        if not dim or dim == "@switch":
            dim = self.switches.get("dim") or default or self.spec.switch_dims[0]
        return dim

    def grain(self, chart: dict) -> str:
        return chart.get("grain") or self.switches.get("grain", "month")

    def view(self, chart: dict) -> str:
        return chart.get("view") or self.switches.get("view", "abs")

    def color_for(self, dim: str, entity: str, idx: int) -> str:
        mapping = self.colors.get(dim, {})
        if entity in mapping:
            return mapping[entity]
        return SERIES[idx % len(SERIES)] if len(mapping) == 0 and idx < len(SERIES) else OTHER_COLOR


# ---------------------------------------------------------------------------
# Общие элементы
# ---------------------------------------------------------------------------


def base_layout(title: str, *, legend: bool = True, height: int | None = None) -> dict:
    layout: dict[str, Any] = {
        "title": {"text": title, "x": 0.01, "xanchor": "left", "y": 0.97,
                  "font": {"size": 14, "color": INK, "family": FONT}},
        "font": {"family": FONT, "size": 12, "color": INK_2},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "margin": {"l": 60, "r": 20, "t": 64, "b": 48},
        "separators": ", ",
        "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": GRID, "font": {"color": INK, "family": FONT, "size": 12}},
        "showlegend": legend,
        "legend": {"orientation": "h", "xref": "container", "yref": "container", "x": 0.012, "xanchor": "left",
                   "y": 0.915, "yanchor": "top", "font": {"size": 11, "color": INK_2}, "bgcolor": "rgba(0,0,0,0)"},
        "xaxis": _axis(),
        "yaxis": _axis(grid=True),
        "bargap": 0.28,
        "bargroupgap": 0.06,
        "hovermode": "closest",
        "dragmode": "zoom",
    }
    if height:
        layout["height"] = height
    return layout


def _axis(grid: bool = False, title: str = "") -> dict:
    return {
        "showgrid": grid, "gridcolor": GRID, "gridwidth": 1, "zeroline": False,
        "showline": True, "linecolor": AXIS, "linewidth": 1,
        "tickfont": {"color": MUTED, "size": 11}, "ticks": "",
        "title": {"text": title, "font": {"size": 11, "color": MUTED}},
        "automargin": True,
    }


def empty_figure(title: str, message: str = "Нет данных для выбранных фильтров") -> dict:
    layout = base_layout(title, legend=False)
    layout["xaxis"] = {"visible": False}
    layout["yaxis"] = {"visible": False}
    layout["annotations"] = [{"text": message, "showarrow": False, "x": 0.5, "y": 0.5, "xref": "paper",
                              "yref": "paper", "font": {"size": 13, "color": MUTED}}]
    return {"data": [], "layout": layout}


def period_label(ts: pd.Timestamp, grain: str) -> str:
    ts = pd.Timestamp(ts)
    if grain == "quarter":
        return f"{(ts.month - 1) // 3 + 1} кв. {ts.year}"
    if grain == "year":
        return str(ts.year)
    return month_label(ts)


def grain_word(grain: str) -> str:
    return {"month": "месяцам", "quarter": "кварталам", "year": "годам"}[grain]


def _scaled(values: pd.Series, unit: str) -> tuple[pd.Series, str]:
    if unit == "%":
        return values, "%"
    finite = values.replace([np.inf, -np.inf], np.nan).abs().dropna()
    div, word = fmt.axis_scale(finite.max() if len(finite) else 0)
    suffix = " ".join(x for x in (word, unit) if x)
    return values / div, suffix


def _hover_texts(frame: pd.DataFrame, cols: list[str], calcs: list[dict]) -> list[str]:
    texts = []
    for _, row in frame.iterrows():
        parts = []
        for col in cols:
            if col not in frame.columns:
                continue
            unit = column_unit(col, calcs)
            value = row[col]
            label = column_label(col, calcs)
            if unit == "%":
                parts.append(f"{label}: <b>{fmt.pct(value)}</b>")
            else:
                parts.append(f"{label}: <b>{fmt.full(value, unit)}</b>")
        texts.append("<br>".join(parts))
    return texts


def _short_name(name: str, limit: int = 34) -> str:
    name = str(name)
    return name if len(name) <= limit else name[: limit - 1].rstrip(" ,") + "…"


def _entity_colors(ctx: ChartContext, dim: str, entities: list[str]) -> list[str]:
    mapping = ctx.colors.get(dim, {})
    result = []
    for i, entity in enumerate(entities):
        result.append(mapping.get(entity, OTHER_COLOR if entity == "Прочие" else SERIES[i % len(SERIES)]))
    return result


def _data_table(frame: pd.DataFrame, cols: list[str], calcs: list[dict], grain: str = "month",
                limit: int = 60) -> dict:
    rows = []
    for _, row in frame.head(limit).iterrows():
        cells = []
        for col in cols:
            value = row.get(col)
            if col == "period":
                cells.append(period_label(value, grain))
            elif isinstance(value, (int, float, np.floating, np.integer)):
                unit = column_unit(col, calcs)
                cells.append(fmt.pct(value) if unit == "%" else fmt.full(float(value), unit))
            else:
                cells.append(str(value))
        rows.append(cells)
    return {"columns": [column_label(c, calcs) for c in cols], "rows": rows, "truncated": len(frame) > limit}


def value_meta(col: str, calcs: list[dict]) -> tuple[str, str]:
    return column_label(col, calcs), column_unit(col, calcs)


def is_ratio(col: str, calcs: list[dict]) -> bool:
    return column_unit(col, calcs) == "%"


# ---------------------------------------------------------------------------
# 1. План-факт по периодам
# ---------------------------------------------------------------------------


def planfact_period(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    grain = ctx.grain(chart)
    view = ctx.view(chart)
    value_col = chart.get("value")
    title = chart.get("title") or (
        f"План-факт по {grain_word(grain)}" if fam.plan else f"{fam.label.split(',')[0]} по {grain_word(grain)}")
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}

    if value_col:  # график перестроен на пользовательское вычисление
        return _period_calc(ctx, chart, value_col, grain, title)

    needed = ([fam.plan] if fam.plan else []) + [fam.fact]
    data = ctx.agg(["period"], needed, grain)
    if fam.plan:
        data["__exec"] = data[fam.fact] / data[fam.plan].replace(0, np.nan) * 100
        data["exec_pct" if fam.key == "amount" else "exec_qty_pct"] = data["__exec"]
        needed = needed + ["exec_pct" if fam.key == "amount" else "exec_qty_pct"]
    x = [period_label(p, grain) for p in data["period"]]
    layout = base_layout(title)
    traces: list[dict] = []
    table_cols = ["period"] + needed

    if view == "pct" and fam.plan:
        values = data["__exec"]
        good = values >= 100
        for mask, name, color in ((good, "План выполнен", GOOD), (~good, "План не выполнен", BAD)):
            traces.append(go.Bar(
                x=[xi for xi, m in zip(x, mask) if m], y=values[mask], name=name, marker={"color": color, "cornerradius": 4},
                hovertemplate="%{x}<br>Выполнение: <b>%{y:.1f}%</b><extra></extra>").to_plotly_json())
        layout["barmode"] = "overlay"
        layout["yaxis"]["title"]["text"] = "Выполнение плана, %"
        layout["shapes"] = [_hline(100)]
        layout["annotations"] = [_hline_label(100, "План 100%")]
        layout["title"]["text"] = title.replace("План-факт", "Выполнение плана") + ", %"
        table_cols = ["period"] + needed
    elif view == "dyn":
        prev_values = _previous_bucket(ctx, fam.fact, grain, data)
        change = (data[fam.fact] / prev_values.replace(0, np.nan) - 1) * 100
        up = change >= 0
        for mask, name, color in ((up, "Рост к пред. периоду", GOOD), (~up, "Снижение", BAD)):
            traces.append(go.Bar(
                x=[xi for xi, m in zip(x, mask) if m], y=change[mask], name=name, marker={"color": color, "cornerradius": 4},
                hovertemplate="%{x}<br>Динамика: <b>%{y:+.1f}%</b><extra></extra>").to_plotly_json())
        layout["barmode"] = "overlay"
        layout["yaxis"]["title"]["text"] = "Изменение к пред. периоду, %"
        layout["shapes"] = [_hline(0)]
        layout["title"]["text"] = f"Динамика факта по {grain_word(grain)}, %"
        data["__chg"] = change
        table_cols = ["period", fam.fact]
    else:
        scaled_fact, axis_unit = _scaled(data[fam.fact], fam.unit)
        hover = _planfact_hover(data, fam, ctx.calcs)
        series = ctx.switches.get("series", "both")
        if fam.plan and series in ("both", "plan"):
            scaled_plan = _scale_like(data[fam.plan], data[fam.fact], scaled_fact)
            traces.append(go.Bar(x=x, y=scaled_plan, name="План", marker={"color": PLAN_COLOR, "cornerradius": 4},
                                 customdata=hover, hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>").to_plotly_json())
        if not fam.plan or series in ("both", "fact"):
            traces.append(go.Bar(x=x, y=scaled_fact, name="Факт", marker={"color": FACT_COLOR, "cornerradius": 4},
                                 customdata=hover, hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>").to_plotly_json())
        layout["barmode"] = "group"
        layout["yaxis"]["title"]["text"] = axis_unit
        layout["hovermode"] = "closest"
        layout["showlegend"] = True
    layout["xaxis"]["type"] = "category"
    return {"figure": {"data": traces, "layout": layout}, "title": layout["title"]["text"],
            "table": _data_table(data, [c for c in table_cols if c in data.columns], ctx.calcs, grain),
            "x_shared": "period"}


def _scale_like(values: pd.Series, reference: pd.Series, scaled_reference: pd.Series) -> pd.Series:
    ratio = (reference / scaled_reference.replace(0, np.nan)).dropna()
    div = ratio.iloc[0] if len(ratio) else 1.0
    return values / div


def _planfact_hover(data: pd.DataFrame, fam: MetricFamily, calcs: list[dict]) -> list[str]:
    texts = []
    for _, row in data.iterrows():
        fact = row[fam.fact]
        if fam.plan:
            plan = row[fam.plan]
            exec_pct = fact / plan * 100 if plan else np.nan
            dev = fact - plan
            color = GOOD if (not np.isnan(exec_pct) and exec_pct >= 100) else BAD
            texts.append(
                f"План: <b>{fmt.full(plan, fam.unit)}</b><br>Факт: <b>{fmt.full(fact, fam.unit)}</b><br>"
                f"Отклонение: <b><span style='color:{color}'>{fmt.signed_short(dev, fam.unit)}</span></b><br>"
                f"Выполнение: <b>{fmt.pct(exec_pct)}</b>")
        else:
            texts.append(f"{fam.label.split(',')[0]}: <b>{fmt.full(fact, fam.unit)}</b>")
    return texts


def _previous_bucket(ctx: ChartContext, col: str, grain: str, data: pd.DataFrame) -> pd.Series:
    frames = [ctx.frame] + ([ctx.prev] if ctx.prev is not None and len(ctx.prev) else [])
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.assign(period=bucket(pool["period"], grain)).groupby("period")[col].sum()
    offset = {"month": pd.DateOffset(months=1), "quarter": pd.DateOffset(months=3), "year": pd.DateOffset(years=1)}[grain]
    return pd.Series([pool.get(p - offset, np.nan) for p in data["period"]], index=data.index, dtype=float)


def _period_calc(ctx: ChartContext, chart: dict, value_col: str, grain: str, title: str) -> dict:
    """План-факт/тренд, перестроенный на пользовательское вычисление."""
    calc = next((c for c in ctx.calcs if c["id"] == value_col), None)
    fam = ctx.family(chart)
    label, unit = value_meta(value_col, ctx.calcs)
    layout = base_layout(chart.get("title") or calc_title(ctx, value_col, grain) or f"{label.split(',')[0]} по {grain_word(grain)}")
    x_needed = [value_col]
    extra_plan = None
    if calc and calc["kind"] == "cumsum" and fam.plan and calc["measure"] == fam.fact:
        from .calc import make_calc
        extra_plan = make_calc("cumsum", existing=ctx.calcs, measure=fam.plan)
        extra_plan["id"] = "__cum_plan"
    calcs = ctx.calcs + ([extra_plan] if extra_plan else [])
    if extra_plan and not chart.get("title"):
        layout["title"]["text"] = f"План и факт нарастающим итогом по {grain_word(grain)}"
    data = aggregate(ctx.frame, ["period"], x_needed + (["__cum_plan"] if extra_plan else []), calcs,
                     measures=ctx.measures, grain=grain, prev=ctx.prev)
    x = [period_label(p, grain) for p in data["period"]]
    traces = []
    if unit == "%":
        values = data[value_col]
        if calc and calc["kind"] == "change":
            up = values >= 0
            for mask, name, color in ((up, "Рост", GOOD), (~up, "Снижение", BAD)):
                traces.append(go.Bar(x=[xi for xi, m in zip(x, mask) if m], y=values[mask], name=name,
                                     marker={"color": color, "cornerradius": 4},
                                     hovertemplate="%{x}<br>" + label + ": <b>%{y:+.1f}%</b><extra></extra>").to_plotly_json())
            layout["barmode"] = "overlay"
            layout["shapes"] = [_hline(0)]
        else:
            traces.append(go.Scatter(x=x, y=values, mode="lines+markers", name=label, line={"color": FACT_COLOR, "width": 2},
                                     marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                                     hovertemplate="%{x}<br>" + label + ": <b>%{y:.1f}%</b><extra></extra>").to_plotly_json())
        layout["yaxis"]["title"]["text"] = "%"
    else:
        scaled, axis_unit = _scaled(data[value_col], unit)
        hover = [fmt.full(v, unit) for v in data[value_col]]
        if extra_plan:
            plan_scaled = _scale_like(data["__cum_plan"], data[value_col], scaled)
            plan_hover = [fmt.full(v, unit) for v in data["__cum_plan"]]
            traces.append(go.Scatter(x=x, y=plan_scaled, mode="lines+markers", name="План нарастающим итогом",
                                     line={"color": PLAN_COLOR, "width": 2}, marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                                     customdata=plan_hover, hovertemplate="%{x}<br>План: <b>%{customdata}</b><extra></extra>").to_plotly_json())
            name = "Факт нарастающим итогом"
        else:
            name = label
        kind = "line" if calc and calc["kind"] == "cumsum" else "bar"
        if kind == "line":
            traces.append(go.Scatter(x=x, y=scaled, mode="lines+markers", name=name, line={"color": FACT_COLOR, "width": 2},
                                     marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                                     customdata=hover, hovertemplate="%{x}<br>" + name + ": <b>%{customdata}</b><extra></extra>").to_plotly_json())
        else:
            colors = [GOOD if v >= 0 else BAD for v in data[value_col].fillna(0)] if (data[value_col] < 0).any() else FACT_COLOR
            traces.append(go.Bar(x=x, y=scaled, name=name, marker={"color": colors, "cornerradius": 4},
                                 customdata=hover, hovertemplate="%{x}<br>" + name + ": <b>%{customdata}</b><extra></extra>").to_plotly_json())
        layout["yaxis"]["title"]["text"] = axis_unit
    layout["xaxis"]["type"] = "category"
    layout["showlegend"] = len(traces) > 1
    return {"figure": {"data": traces, "layout": layout}, "title": layout["title"]["text"],
            "table": _data_table(data, ["period", value_col], ctx.calcs, grain), "x_shared": "period"}


def calc_title(ctx: ChartContext, value_col: str, grain: str) -> str | None:
    """Заголовок графика, перестроенного на вычисление, с учётом зерна периода."""
    calc = next((c for c in ctx.calcs if c["id"] == value_col), None)
    if not calc:
        return None
    if calc["kind"] == "change":
        what = "факта" if calc["measure"] == "fact" else f"«{column_label(calc['measure'], ctx.calcs).split(',')[0].lower()}»"
        if calc.get("lag") == "year":
            return f"Динамика {what} год к году по {grain_word(grain)}, %"
        prev_word = {"month": "месяцу", "quarter": "кварталу", "year": "году"}[grain]
        return f"Динамика {what} к предыдущему {prev_word}, %"
    return f"{calc['label'].split(',')[0]} по {grain_word(grain)}"


def _hline(y: float) -> dict:
    return {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": y, "y1": y,
            "line": {"color": INK_2, "width": 1}}


def _hline_label(y: float, text: str) -> dict:
    return {"xref": "paper", "x": 1, "xanchor": "right", "y": y, "yanchor": "bottom", "text": text,
            "showarrow": False, "font": {"size": 10, "color": INK_2}}


# ---------------------------------------------------------------------------
# 2. Структура (круговая диаграмма)
# ---------------------------------------------------------------------------


def structure(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = chart.get("dim") if chart.get("dim") and chart.get("dim") != "@switch" else ctx.dim(chart)
    grain = ctx.grain(chart)
    value_col = chart.get("value") if chart.get("value") and not is_ratio(chart.get("value"), ctx.calcs) else fam.fact
    if is_ratio(value_col, ctx.calcs):  # доли от процентов не складываются — берём денежную метрику
        value_col = next((m.fact for m in ctx.spec.metrics.values() if not is_ratio(m.fact, ctx.calcs)), value_col)
    scope = chart.get("scope", "last")
    frame = ctx.frame
    if frame.empty:
        title = chart.get("title") or f"Структура по {DIM_LABELS.get(dim, dim).lower()}"
        return {"figure": empty_figure(title), "title": title}
    scope_label = "весь период"
    if scope == "last" and "period" in frame.columns:
        buckets = bucket(frame["period"], grain)
        last = buckets.max()
        frame = frame[buckets == last]
        scope_label = period_label(last, grain)
    what = {"amount": "продаж", "qty": "продаж в штуках", "discount": "скидок", "deals": "сделок"}.get(fam.key, "продаж")
    if ctx.spec.key == "discounts":
        what = "скидок"
    title = chart.get("title") or f"Структура {what} по {DIM_DATIVE_PL(dim)}, {scope_label}"
    sub = ChartContext(ctx.spec, frame, None, ctx.calcs, ctx.switches, ctx.colors, ctx.entity_order)
    data = sub.agg([dim], [value_col]).sort_values(value_col, ascending=False)
    data = data[data[value_col] > 0]
    if data.empty:
        return {"figure": empty_figure(title), "title": title}
    if len(data) > MAX_SLICES + 1:
        head = data.head(MAX_SLICES)
        rest = data.iloc[MAX_SLICES:]
        other = {dim: "Прочие", value_col: rest[value_col].sum()}
        data = pd.concat([head, pd.DataFrame([other])], ignore_index=True)
    labels = [str(v) for v in data[dim]]
    total = data[value_col].sum()
    unit = column_unit(value_col, ctx.calcs)
    hover = [f"{fmt.full(v, unit)}" for v in data[value_col]]
    trace = go.Pie(
        labels=[_short_name(lbl, 28) for lbl in labels], values=data[value_col], hole=0.58, sort=False,
        direction="clockwise", marker={"colors": _entity_colors(ctx, dim, labels), "line": {"color": SURFACE, "width": 2}},
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        texttemplate=["%{percent:.0%}" if v / total >= 0.05 else "" for v in data[value_col]],
        customdata=[[h, full] for h, full in zip(hover, labels)],
        hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]}<br>Доля: <b>%{percent:.1%}</b><extra></extra>",
        textfont={"color": "#ffffff", "size": 11},
    ).to_plotly_json()
    layout = base_layout(title)
    layout["legend"] = {"orientation": "v", "x": 1.0, "xanchor": "left", "y": 0.5, "yanchor": "middle",
                        "font": {"size": 11, "color": INK_2}}
    layout["margin"] = {"l": 10, "r": 10, "t": 64, "b": 20}
    layout["annotations"] = [{"text": f"<b>{fmt.short(total, unit)}</b><br><span style='font-size:11px;color:{MUTED}'>всего</span>",
                              "showarrow": False, "x": 0.5, "y": 0.5, "xref": "paper", "yref": "paper",
                              "font": {"size": 15, "color": INK}}]
    data["__share"] = data[value_col] / total * 100
    table = _data_table(data, [dim, value_col], ctx.calcs)
    table["columns"].append("Доля, %")
    for row, share in zip(table["rows"], data["__share"]):
        row.append(fmt.pct(share))
    return {"figure": {"data": [trace], "layout": layout}, "title": title, "table": table}


# ---------------------------------------------------------------------------
# 3. ТОП (горизонтальные столбцы)
# ---------------------------------------------------------------------------

SORT_LABELS = {
    "fact": "объёму продаж", "plan": "плану", "deviation": "отклонению от плана", "exec_pct": "выполнению плана",
    "deviation_pct": "отклонению от плана, %", "qty": "количеству", "discount": "скидкам", "deals": "числу сделок",
    "discounts_cnt": "числу скидок", "avg_pct": "среднему % скидки",
}


DATIVE_FIRST_WORD = {
    "выполнение": "выполнению", "отклонение": "отклонению", "доля": "доле", "динамика": "динамике",
    "накопительный": "накопительному", "итог": "итогу", "ранг": "рангу", "медиана": "медиане",
    "среднее": "среднему", "минимум": "минимуму", "максимум": "максимуму", "сумма": "сумме",
    "средний": "среднему", "факт": "факту", "план": "плану", "скидка": "скидке", "расчёт": "расчёту",
    "процент": "проценту",
}


def dative_phrase(label: str) -> str:
    """«Выполнение плана, %» → «выполнению плана» (для заголовков «ТОП по …»)."""
    text = label.split(",")[0].strip()
    words = text.split()
    if not words:
        return text.lower()
    first = words[0].lower()
    words[0] = DATIVE_FIRST_WORD.get(first, first)
    if first == "накопительный" and len(words) > 1 and words[1].lower().startswith("итог"):
        words[1] = "итогу" + words[1][4:]
    return " ".join([words[0].lower()] + words[1:])


def top(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = ctx.dim(chart)
    view = ctx.view(chart)
    top_n = int(chart.get("top_n") or (10 if dim not in ("region", "category", "segment", "reason", "outlet_type") else 12))
    sort_by = chart.get("sort_by")
    sort_dir = chart.get("sort_dir", "desc")
    value_col = chart.get("value")
    if not value_col:
        if view == "pct" and fam.plan:
            value_col = "exec_pct" if fam.key == "amount" else "exec_qty_pct"
        elif view == "dyn":
            value_col = "__yoy"
        else:
            value_col = fam.fact
    if not sort_by:
        sort_by = value_col
    elif not chart.get("value") and view == "abs" and sort_by not in (fam.fact, fam.plan) and \
            (sort_by in DERIVED or any(c["id"] == sort_by for c in ctx.calcs)):
        # сортируем по вычислению — его и показываем длиной столбца
        value_col = sort_by
    title_what = SORT_LABELS.get(sort_by) or ("динамике к прошлому году" if sort_by == "__yoy" else dative_phrase(column_label(sort_by, ctx.calcs)))
    lowest = sort_dir == "asc"
    n_entities = ctx.frame[dim].nunique() if dim in ctx.frame.columns else 0
    if n_entities and n_entities <= top_n:
        title = chart.get("title") or f"{DIM_LABELS_PL(dim).capitalize()} по {title_what}" + (" (от худших)" if lowest else "")
    else:
        prefix = "Худшие" if lowest else "ТОП"
        title = chart.get("title") or (f"{prefix} {top_n} {DIM_GENITIVE_PLURAL.get(dim, dim)} по {title_what}" if lowest
                                       else f"ТОП-{top_n} {DIM_GENITIVE_PLURAL.get(dim, dim)} по {title_what}")
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}

    needed = [fam.fact] + ([fam.plan] if fam.plan else [])
    for col in (value_col, sort_by):
        if col and not col.startswith("__") and col not in needed:
            needed.append(col)
    data = ctx.agg([dim], needed)
    if value_col == "__yoy" or sort_by == "__yoy":
        data["__yoy"] = _yoy_by(ctx, dim, fam.fact, data)
    data = data.dropna(subset=[sort_by]) if sort_by in data.columns else data
    data = data.sort_values(sort_by, ascending=lowest).head(top_n)
    if data.empty:
        return {"figure": empty_figure(title), "title": title}
    # Сверху — первое место
    data = data.iloc[::-1].reset_index(drop=True)
    names = [str(v) for v in data[dim]]
    y = [_short_name(v) for v in names]
    layout = base_layout(title)
    traces: list[dict] = []
    unit = "%" if value_col == "__yoy" else column_unit(value_col, ctx.calcs)
    label = "Динамика к прошлому году, %" if value_col == "__yoy" else column_label(value_col, ctx.calcs)
    hover = []
    for _, row in data.iterrows():
        parts = [f"<b>{row[dim]}</b>"]
        if fam.plan:
            plan, fact = row[fam.plan], row[fam.fact]
            parts.append(f"План: {fmt.full(plan, fam.unit)}")
            parts.append(f"Факт: {fmt.full(fact, fam.unit)}")
            parts.append(f"Выполнение: <b>{fmt.pct(fact / plan * 100 if plan else np.nan)}</b>")
        else:
            parts.append(f"{column_label(fam.fact, ctx.calcs)}: <b>{fmt.full(row[fam.fact], fam.unit)}</b>")
        if value_col not in (fam.fact, fam.plan):
            parts.append(f"{label}: <b>{fmt.pct(row[value_col]) if unit == '%' else fmt.full(row[value_col], unit)}</b>")
        if sort_by not in (value_col, fam.fact, fam.plan) and sort_by in data.columns:
            s_unit = column_unit(sort_by, ctx.calcs)
            parts.append(f"{column_label(sort_by, ctx.calcs)}: <b>{fmt.pct(row[sort_by]) if s_unit == '%' else fmt.full(row[sort_by], s_unit)}</b>")
        hover.append("<br>".join(parts))

    if unit == "%":
        values = data[value_col]
        ref = 0 if value_col == "__yoy" or column_label(value_col, ctx.calcs).lower().startswith(("динамика", "отклонение", "доля")) else 100
        if value_col.startswith("share") or "доля" in label.lower() or "ранг" in label.lower():
            colors_mask = None
        else:
            colors_mask = values >= ref
        if colors_mask is None:
            traces.append(go.Bar(x=values, y=y, orientation="h", name=label, marker={"color": FACT_COLOR, "cornerradius": 4},
                                 text=[fmt.pct(v) for v in values], textposition="outside", cliponaxis=False,
                                 customdata=hover, hovertemplate="%{customdata}<extra></extra>").to_plotly_json())
        else:
            names_ok = ("Рост" if ref == 0 else "План выполнен", "Снижение" if ref == 0 else "План не выполнен")
            for mask, name, color in ((colors_mask, names_ok[0], GOOD), (~colors_mask, names_ok[1], BAD)):
                if not mask.any():
                    continue
                traces.append(go.Bar(x=values[mask], y=[yi for yi, m in zip(y, mask) if m], orientation="h", name=name,
                                     marker={"color": color, "cornerradius": 4},
                                     text=[fmt.pct(v) for v in values[mask]], textposition="outside", cliponaxis=False,
                                     customdata=[h for h, m in zip(hover, mask) if m],
                                     hovertemplate="%{customdata}<extra></extra>").to_plotly_json())
            layout["shapes"] = [{"type": "line", "yref": "paper", "y0": 0, "y1": 1, "x0": ref, "x1": ref,
                                 "line": {"color": INK_2, "width": 1}}]
        layout["xaxis"]["title"]["text"] = label
        layout["barmode"] = "overlay"
    else:
        scaled, axis_unit = _scaled(data[value_col], unit)
        if fam.plan and value_col == fam.fact:
            ok = data[fam.fact] >= data[fam.plan]
            for mask, name, color in ((ok, "План выполнен", GOOD), (~ok, "План не выполнен", BAD)):
                if not mask.any():
                    continue
                traces.append(go.Bar(x=scaled[mask], y=[yi for yi, m in zip(y, mask) if m], orientation="h", name=name,
                                     marker={"color": color, "cornerradius": 4},
                                     customdata=[h for h, m in zip(hover, mask) if m],
                                     hovertemplate="%{customdata}<extra></extra>").to_plotly_json())
            plan_scaled = _scale_like(data[fam.plan], data[fam.fact], scaled)
            if ctx.switches.get("series", "both") != "fact":
                traces.append(go.Scatter(x=plan_scaled, y=y, mode="markers", name="План",
                                         marker={"symbol": "line-ns", "size": 18, "line": {"width": 3, "color": INK_2}},
                                         hoverinfo="skip").to_plotly_json())
        else:
            signs = data[value_col] < 0
            colors = [BAD if s else (GOOD if (data[value_col] < 0).any() else FACT_COLOR) for s in signs]
            traces.append(go.Bar(x=scaled, y=y, orientation="h", name=label, marker={"color": colors, "cornerradius": 4},
                                 text=[fmt.signed_short(v, unit) if (data[value_col] < 0).any() else fmt.short(v, unit) for v in data[value_col]],
                                 textposition="auto", cliponaxis=False, textfont={"size": 10},
                                 insidetextfont={"size": 10, "color": "#ffffff"}, outsidetextfont={"size": 10, "color": INK_2},
                                 customdata=hover, hovertemplate="%{customdata}<extra></extra>").to_plotly_json())
            layout["showlegend"] = False
            if (data[value_col] < 0).any():
                layout["shapes"] = [{"type": "line", "yref": "paper", "y0": 0, "y1": 1, "x0": 0, "x1": 0,
                                     "line": {"color": INK_2, "width": 1}}]
        layout["xaxis"]["title"]["text"] = axis_unit
        layout["barmode"] = "overlay"
    layout["yaxis"] = {**_axis(), "type": "category", "categoryorder": "array", "categoryarray": y,
                       "tickfont": {"color": INK_2, "size": 11}}
    layout["xaxis"]["showgrid"] = True
    layout["margin"]["l"] = 12
    layout["margin"]["r"] = 80
    table_cols = [dim] + [c for c in dict.fromkeys(needed + [value_col, sort_by]) if c in data.columns and not c.startswith("__")]
    return {"figure": {"data": traces, "layout": layout}, "title": title,
            "table": _data_table(data.iloc[::-1], table_cols, ctx.calcs)}


def _yoy_by(ctx: ChartContext, dim: str, col: str, data: pd.DataFrame) -> pd.Series:
    if ctx.prev is None or ctx.prev.empty:
        return pd.Series(np.nan, index=data.index)
    prev = ctx.prev.groupby(dim)[col].sum()
    return pd.Series([(row[col] / prev.get(row[dim], np.nan) - 1) * 100 if prev.get(row[dim], 0) else np.nan
                      for _, row in data.iterrows()], index=data.index)


# ---------------------------------------------------------------------------
# 4. Динамика выполнения плана (линии) / тепловая карта
# ---------------------------------------------------------------------------


def exec_line(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = ctx.dim(chart)
    grain = ctx.grain(chart)
    view = ctx.view(chart)
    value_col = chart.get("value")
    if chart.get("variant") == "heatmap":
        return heatmap(ctx, chart)
    if not fam.plan and not value_col:
        return trend(ctx, {**chart, "mode": "line", "split": dim if ctx.frame[dim].nunique() <= MAX_LINES else None})
    if value_col:
        metric = value_col
    elif view == "dyn":
        metric = "__chg"
    else:
        metric = "exec_pct" if fam.key == "amount" else "exec_qty_pct"
    label = {"__dev": "Отклонение от плана, шт", "__chg": "Динамика к пред. периоду, %"}.get(metric) or column_label(metric, ctx.calcs)
    unit = {"__dev": "шт", "__chg": "%"}.get(metric) or column_unit(metric, ctx.calcs)
    title = chart.get("title") or (f"Динамика выполнения плана по {grain_word(grain)}, %" if metric in ("exec_pct", "exec_qty_pct")
                                   else calc_title(ctx, metric, grain) or f"{label.split(',')[0]} по {grain_word(grain)}")
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}

    def compute(frame_by: list[str]) -> pd.DataFrame:
        base_needed = [c for c in (fam.fact, fam.plan) if c]
        if metric in ("__dev", "__chg"):
            out = ctx.agg(frame_by, base_needed, grain)
            if metric == "__dev":
                out["__dev"] = out[fam.fact] - (out[fam.plan] if fam.plan else 0)
            return out
        return ctx.agg(frame_by, base_needed + [metric], grain)

    total = compute(["period"])
    if metric == "__chg":
        prev_vals = _previous_bucket(ctx, fam.fact, grain, total)
        total["__chg"] = (total[fam.fact] / prev_vals.replace(0, np.nan) - 1) * 100
    x_total = [period_label(p, grain) for p in total["period"]]
    layout = base_layout(title)
    traces: list[dict] = []
    n_entities = ctx.frame[dim].nunique() if dim in ctx.frame.columns else 0
    show_entities = 1 < n_entities <= MAX_LINES and metric != "__chg"
    if show_entities:
        by_dim = compute(["period", dim])
        order = [e for e in ctx.entity_order.get(dim, []) if e in set(by_dim[dim])] or sorted(by_dim[dim].unique())
        for i, entity in enumerate(order):
            part = by_dim[by_dim[dim] == entity].sort_values("period")
            vals, _ = (part[metric], unit) if unit == "%" else _scaled_pair(part[metric], total[metric], unit)
            traces.append(go.Scatter(
                x=[period_label(p, grain) for p in part["period"]], y=vals, mode="lines", name=_short_name(entity, 24),
                line={"color": ctx.colors.get(dim, {}).get(entity, SERIES[i % len(SERIES)]), "width": 2},
                customdata=[fmt.pct(v) if unit == "%" else fmt.full(v, unit) for v in part[metric]],
                hovertemplate=f"<b>{entity}</b><br>%{{x}}: <b>%{{customdata}}</b><extra></extra>").to_plotly_json())
    tvals, axis_unit = (total[metric], "%") if unit == "%" else _scaled(total[metric], unit)
    traces.append(go.Scatter(
        x=x_total, y=tvals, mode="lines+markers", name="Итого",
        line={"color": INK, "width": 3}, marker={"size": 8, "color": INK, "line": {"color": SURFACE, "width": 2}},
        customdata=[fmt.pct(v) if unit == "%" else fmt.full(v, unit) for v in total[metric]],
        hovertemplate="<b>Итого</b><br>%{x}: <b>%{customdata}</b><extra></extra>").to_plotly_json())
    ref = 100 if metric in ("exec_pct", "exec_qty_pct") else 0
    if unit == "%" and ref == 100:
        layout["shapes"] = [
            {"type": "rect", "xref": "paper", "x0": 0, "x1": 1, "yref": "y", "y0": 100, "y1": 1000,
             "fillcolor": GOOD, "opacity": 0.06, "line": {"width": 0}, "layer": "below"},
            {"type": "rect", "xref": "paper", "x0": 0, "x1": 1, "yref": "y", "y0": -1000, "y1": 100,
             "fillcolor": BAD, "opacity": 0.05, "line": {"width": 0}, "layer": "below"},
            _hline(100),
        ]
        all_vals = [v for t in traces for v in t["y"] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        lo = min(all_vals + [100]) if all_vals else 80
        hi = max(all_vals + [100]) if all_vals else 110
        pad = max(2.0, (hi - lo) * 0.12)
        layout["yaxis"]["range"] = [lo - pad, hi + pad]
        layout["annotations"] = [
            {"xref": "paper", "x": 0.005, "xanchor": "left", "y": hi + pad * 0.5, "yref": "y", "text": "перевыполнение",
             "showarrow": False, "font": {"size": 10, "color": "#006300"}},
            {"xref": "paper", "x": 0.005, "xanchor": "left", "y": lo - pad * 0.5, "yref": "y", "text": "недовыполнение",
             "showarrow": False, "font": {"size": 10, "color": BAD}},
        ]
    else:
        layout["shapes"] = [_hline(0)]
    layout["yaxis"]["title"]["text"] = axis_unit
    layout["xaxis"]["type"] = "category"
    layout["hovermode"] = "closest"
    layout["showlegend"] = show_entities
    if show_entities:
        layout["legend"] = {"orientation": "v", "x": 1.01, "xanchor": "left", "y": 1, "yanchor": "top",
                            "font": {"size": 11, "color": INK_2}, "bgcolor": "rgba(0,0,0,0)"}
        layout["margin"]["r"] = 8
    table_cols = ["period", metric] if not metric.startswith("__") else ["period", fam.plan, fam.fact]
    return {"figure": {"data": traces, "layout": layout}, "title": title,
            "table": _data_table(total, [c for c in table_cols if c in total.columns], ctx.calcs, grain),
            "x_shared": "period"}


def _scaled_pair(values: pd.Series, reference: pd.Series, unit: str) -> tuple[pd.Series, str]:
    finite = reference.replace([np.inf, -np.inf], np.nan).abs().dropna()
    div, word = fmt.axis_scale(finite.max() if len(finite) else 0)
    return values / div, " ".join(x for x in (word, unit) if x)


def heatmap(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = ctx.dim(chart)
    grain = ctx.grain(chart)
    metric = chart.get("value") or ("exec_pct" if fam.plan and fam.key == "amount" else ("exec_qty_pct" if fam.plan else fam.fact))
    label = column_label(metric, ctx.calcs)
    unit = column_unit(metric, ctx.calcs)
    title = chart.get("title") or f"{label.split(',')[0]}: {DIM_LABELS_PL(dim)} × {grain_word(grain).replace('ам', 'ы')}"
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}
    needed = [fam.fact] + ([fam.plan] if fam.plan else []) + ([metric] if metric not in (fam.fact, fam.plan) else [])
    data = ctx.agg(["period", dim], needed, grain)
    totals = data.groupby(dim)[fam.fact].sum().sort_values(ascending=False)
    entities = list(totals.index[:25])
    data = data[data[dim].isin(entities)]
    pivot = data.pivot_table(index=dim, columns="period", values=metric, aggfunc="first").reindex(entities)
    x = [period_label(p, grain) for p in pivot.columns]
    z = pivot.values.astype(float)
    text = [[(fmt.pct(v) if unit == "%" else fmt.short(v, unit)) for v in row] for row in z]
    if unit == "%" and metric in ("exec_pct", "exec_qty_pct"):
        spread = max(5.0, float(np.nanmax(np.abs(z - 100))) if np.isfinite(z).any() else 10)
        colorscale = [[0, BAD], [0.5, "#f0efec"], [1, FACT_COLOR]]
        extra = {"zmid": 100, "zmin": 100 - spread, "zmax": 100 + spread}
    else:
        colorscale = [[0, "#cde2fb"], [1, "#104281"]]
        extra = {}
    trace = go.Heatmap(z=z, x=x, y=[_short_name(e, 26) for e in entities], colorscale=colorscale, xgap=2, ygap=2,
                       customdata=text, hovertemplate="<b>%{y}</b><br>%{x}: <b>%{customdata}</b><extra></extra>",
                       colorbar={"thickness": 10, "outlinewidth": 0, "tickfont": {"size": 10, "color": MUTED},
                                 "title": {"text": "%" if unit == "%" else unit, "font": {"size": 10}}}, **extra).to_plotly_json()
    layout = base_layout(title, legend=False)
    layout["yaxis"] = {**_axis(), "autorange": "reversed", "tickfont": {"size": 11, "color": INK_2}}
    layout["xaxis"]["type"] = "category"
    layout["margin"]["l"] = 12
    return {"figure": {"data": [trace], "layout": layout}, "title": title,
            "table": _data_table(data.sort_values(["period"]), ["period", dim, metric], ctx.calcs, grain), "x_shared": "period"}


def DIM_LABELS_PL(dim: str) -> str:  # noqa: N802 — короткий помощник для подписи
    from .semantic import DIM_LABELS_PLURAL

    return DIM_LABELS_PLURAL.get(dim, dim).lower()


# ---------------------------------------------------------------------------
# Дополнительные типы: план-факт по разрезу, сравнение точек, тренд, столбцы
# ---------------------------------------------------------------------------


def planfact_dim(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = ctx.dim(chart)
    top_n = int(chart.get("top_n") or 12)
    sort_by = chart.get("sort_by") or fam.fact
    asc = chart.get("sort_dir") == "asc"
    what = "План-факт" if fam.plan else fam.label.split(",")[0]
    title = chart.get("title") or f"{what} по {DIM_DATIVE_PL(dim)}"
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}
    needed = [fam.fact] + ([fam.plan] if fam.plan else [])
    if sort_by not in needed:
        needed.append(sort_by)
    data = ctx.agg([dim], needed).sort_values(sort_by, ascending=asc)
    truncated = len(data) > top_n
    data = data.head(top_n)
    if truncated:
        title += f" (ТОП-{top_n})"
    names = [str(v) for v in data[dim]]
    x = [_short_name(n, 22) for n in names]
    scaled_fact, axis_unit = _scaled(data[fam.fact], fam.unit)
    hover = [f"<b>{n}</b><br>{h}" for n, h in zip(names, _planfact_hover(data, fam, ctx.calcs))]
    traces = []
    series = ctx.switches.get("series", "both")
    if fam.plan and series in ("both", "plan"):
        traces.append(go.Bar(x=x, y=_scale_like(data[fam.plan], data[fam.fact], scaled_fact), name="План",
                             marker={"color": PLAN_COLOR, "cornerradius": 4}, customdata=hover,
                             hovertemplate="%{customdata}<extra></extra>").to_plotly_json())
    if not fam.plan or series in ("both", "fact"):
        traces.append(go.Bar(x=x, y=scaled_fact, name="Факт", marker={"color": FACT_COLOR, "cornerradius": 4},
                             customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                             text=[fmt.pct(f / p * 100) if fam.plan and p else "" for f, p in
                                   zip(data[fam.fact], data[fam.plan] if fam.plan else data[fam.fact])],
                             textposition="outside", cliponaxis=False, textfont={"size": 10, "color": INK_2}).to_plotly_json())
    layout = base_layout(title)
    layout["barmode"] = "group"
    layout["yaxis"]["title"]["text"] = axis_unit
    layout["xaxis"]["type"] = "category"
    layout["xaxis"]["tickangle"] = -30 if len(x) > 6 else 0
    return {"figure": {"data": traces, "layout": layout}, "title": title,
            "table": _data_table(data, [dim] + needed + (["exec_pct"] if "exec_pct" in data.columns else []), ctx.calcs)}


def DIM_DATIVE_PL(dim: str) -> str:  # noqa: N802
    return {"region": "регионам", "outlet": "торговым точкам", "category": "категориям", "product": "продуктам",
            "client": "клиентам", "segment": "сегментам", "reason": "причинам", "city": "городам",
            "outlet_type": "типам точек"}.get(dim, dim)


def scatter_planfact(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = ctx.dim(chart, "outlet")
    threshold = float(chart.get("threshold", 90))
    title = chart.get("title") or f"Все {DIM_LABELS_PL(dim)}: план против факта"
    if ctx.frame.empty or not fam.plan:
        return {"figure": empty_figure(title, "Для сравнения нужен план" if not fam.plan else "Нет данных"), "title": title}
    data = ctx.agg([dim], [fam.fact, fam.plan])
    data = data[data[fam.plan] > 0]
    data["__exec"] = data[fam.fact] / data[fam.plan] * 100
    max_v = max(data[fam.plan].max(), data[fam.fact].max()) if len(data) else 1
    div, word = fmt.axis_scale(max_v)
    axis_unit = " ".join(x for x in (word, fam.unit) if x)
    groups = [
        (data["__exec"] < threshold, f"Отстают (<{threshold:.0f}%)", BAD),
        ((data["__exec"] >= threshold) & (data["__exec"] < 100), f"Близко к плану ({threshold:.0f}–100%)", NEUTRAL),
        (data["__exec"] >= 100, "Выполнили план", GOOD),
    ]
    traces = []
    for mask, name, color in groups:
        part = data[mask]
        if part.empty:
            continue
        traces.append(go.Scatter(
            x=part[fam.plan] / div, y=part[fam.fact] / div, mode="markers", name=f"{name}: {len(part)}",
            marker={"size": 9, "color": color, "opacity": 0.85, "line": {"color": SURFACE, "width": 1.5}},
            customdata=[[n, fmt.full(p, fam.unit), fmt.full(f, fam.unit), fmt.pct(e)] for n, p, f, e in
                        zip(part[dim], part[fam.plan], part[fam.fact], part["__exec"])],
            hovertemplate="<b>%{customdata[0]}</b><br>План: %{customdata[1]}<br>Факт: %{customdata[2]}<br>Выполнение: <b>%{customdata[3]}</b><extra></extra>",
        ).to_plotly_json())
    lim = max_v / div * 1.05
    layout = base_layout(title)
    layout["shapes"] = [{"type": "line", "x0": 0, "y0": 0, "x1": lim, "y1": lim, "line": {"color": INK_2, "width": 1}}]
    layout["annotations"] = [{"x": lim, "y": lim, "text": "факт = план", "showarrow": False, "xanchor": "right",
                              "yanchor": "bottom", "font": {"size": 10, "color": INK_2}}]
    layout["xaxis"] = {**_axis(grid=True, title=f"План, {axis_unit}"), "range": [0, lim]}
    layout["yaxis"] = {**_axis(grid=True, title=f"Факт, {axis_unit}"), "range": [0, lim]}
    data = data.sort_values("__exec")
    data["exec_pct"] = data["__exec"]
    return {"figure": {"data": traces, "layout": layout}, "title": title,
            "table": _data_table(data, [dim, fam.plan, fam.fact, "exec_pct"], ctx.calcs)}


def trend(ctx: ChartContext, chart: dict) -> dict:
    """Тренд показателя по периодам, можно с разбивкой (линии или столбцы с накоплением)."""
    fam = ctx.family(chart)
    grain = ctx.grain(chart)
    metric = chart.get("value") or fam.fact
    split = chart.get("split")
    mode = chart.get("mode", "bar")
    label, unit = value_meta(metric, ctx.calcs)
    title = chart.get("title") or f"{label.split(',')[0]} по {grain_word(grain)}"
    if split:
        title += f" в разрезе: {DIM_LABELS.get(split, split).lower()}"
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}
    total = ctx.agg(["period"], [metric], grain)
    x_all = [period_label(p, grain) for p in total["period"]]
    layout = base_layout(title)
    traces = []
    if split and split in ctx.frame.columns:
        data = ctx.agg(["period", split], [metric], grain)
        order = [e for e in ctx.entity_order.get(split, []) if e in set(data[split])]
        order += [e for e in data.groupby(split)[metric].sum().sort_values(ascending=False).index if e not in order]
        if len(order) > MAX_LINES:
            keep = order[: MAX_LINES - 1]
            data[split] = data[split].where(data[split].isin(keep), "Прочие")
            data = data.groupby(["period", split], as_index=False)[metric].sum()
            order = keep + ["Прочие"]
        if unit != "%":
            div, word = fmt.axis_scale(total[metric].abs().max())
        else:
            div, word = 1, ""
        for i, entity in enumerate(order):
            part = data[data[split] == entity].sort_values("period")
            color = ctx.colors.get(split, {}).get(entity, OTHER_COLOR if entity == "Прочие" else SERIES[i % len(SERIES)])
            xs = [period_label(p, grain) for p in part["period"]]
            custom = [fmt.pct(v) if unit == "%" else fmt.full(v, unit) for v in part[metric]]
            if mode == "line" or unit == "%":
                traces.append(go.Scatter(x=xs, y=part[metric] / div, mode="lines", name=_short_name(entity, 26),
                                         line={"color": color, "width": 2}, customdata=custom,
                                         hovertemplate=f"<b>{entity}</b><br>%{{x}}: <b>%{{customdata}}</b><extra></extra>").to_plotly_json())
            else:
                traces.append(go.Bar(x=xs, y=part[metric] / div, name=_short_name(entity, 26),
                                     marker={"color": color, "line": {"color": SURFACE, "width": 1}}, customdata=custom,
                                     hovertemplate=f"<b>{entity}</b><br>%{{x}}: <b>%{{customdata}}</b><extra></extra>").to_plotly_json())
        layout["barmode"] = "stack"
        axis_unit = "%" if unit == "%" else " ".join(x for x in (word, unit) if x)
    else:
        vals, axis_unit = (total[metric], "%") if unit == "%" else _scaled(total[metric], unit)
        custom = [fmt.pct(v) if unit == "%" else fmt.full(v, unit) for v in total[metric]]
        if mode == "line" or unit == "%":
            traces.append(go.Scatter(x=x_all, y=vals, mode="lines+markers", name=label, line={"color": FACT_COLOR, "width": 2},
                                     marker={"size": 8, "line": {"color": SURFACE, "width": 2}}, customdata=custom,
                                     hovertemplate="%{x}: <b>%{customdata}</b><extra></extra>").to_plotly_json())
        else:
            traces.append(go.Bar(x=x_all, y=vals, name=label, marker={"color": FACT_COLOR, "cornerradius": 4},
                                 customdata=custom, hovertemplate="%{x}: <b>%{customdata}</b><extra></extra>").to_plotly_json())
        layout["showlegend"] = False
    layout["yaxis"]["title"]["text"] = axis_unit
    if split and len(traces) > 3:
        layout["legend"] = {"orientation": "v", "x": 1.01, "xanchor": "left", "y": 1, "yanchor": "top",
                            "font": {"size": 11, "color": INK_2}, "bgcolor": "rgba(0,0,0,0)", "traceorder": "normal"}
        layout["margin"]["r"] = 8
    layout["xaxis"]["type"] = "category"
    layout["xaxis"]["categoryorder"] = "array"
    layout["xaxis"]["categoryarray"] = x_all
    return {"figure": {"data": traces, "layout": layout}, "title": title,
            "table": _data_table(total, ["period", metric], ctx.calcs, grain), "x_shared": "period"}


def bar_dim(ctx: ChartContext, chart: dict) -> dict:
    fam = ctx.family(chart)
    dim = ctx.dim(chart)
    metric = chart.get("value") or fam.fact
    top_n = int(chart.get("top_n") or 10)
    asc = chart.get("sort_dir") == "asc"
    sort_by = chart.get("sort_by") or metric
    label, unit = value_meta(metric, ctx.calcs)
    title = chart.get("title") or f"{label.split(',')[0]}: ТОП-{top_n} {DIM_GENITIVE_PLURAL.get(dim, dim)}"
    if ctx.frame.empty:
        return {"figure": empty_figure(title), "title": title}
    needed = list(dict.fromkeys([metric, sort_by]))
    data = ctx.agg([dim], needed).dropna(subset=[sort_by]).sort_values(sort_by, ascending=asc).head(top_n).iloc[::-1]
    names = [str(v) for v in data[dim]]
    y = [_short_name(n) for n in names]
    vals, axis_unit = (data[metric], "%") if unit == "%" else _scaled(data[metric], unit)
    custom = [f"<b>{n}</b><br>{label}: <b>{fmt.pct(v) if unit == '%' else fmt.full(v, unit)}</b>" for n, v in zip(names, data[metric])]
    trace = go.Bar(x=vals, y=y, orientation="h", marker={"color": FACT_COLOR, "cornerradius": 4},
                   text=[fmt.pct(v) if unit == "%" else fmt.short(v, unit) for v in data[metric]], textposition="outside",
                   cliponaxis=False, textfont={"size": 10, "color": INK_2},
                   customdata=custom, hovertemplate="%{customdata}<extra></extra>").to_plotly_json()
    layout = base_layout(title, legend=False)
    layout["yaxis"] = {**_axis(), "type": "category", "categoryorder": "array", "categoryarray": y,
                       "tickfont": {"color": INK_2, "size": 11}}
    layout["xaxis"]["title"]["text"] = axis_unit
    layout["xaxis"]["showgrid"] = True
    layout["margin"]["l"] = 12
    layout["margin"]["r"] = 56
    return {"figure": {"data": [trace], "layout": layout}, "title": title,
            "table": _data_table(data.iloc[::-1], [dim] + needed, ctx.calcs)}


BUILDERS = {
    "planfact_period": planfact_period,
    "structure": structure,
    "top": top,
    "exec_line": exec_line,
    "heatmap": heatmap,
    "planfact_dim": planfact_dim,
    "scatter_planfact": scatter_planfact,
    "trend": trend,
    "bar_dim": bar_dim,
}

CHART_TYPE_LABELS = {
    "planfact_period": "План-факт по периодам (столбцы)",
    "structure": "Структура (круговая)",
    "top": "ТОП (горизонтальные столбцы)",
    "exec_line": "Динамика выполнения (линии)",
    "heatmap": "Тепловая карта",
    "planfact_dim": "План-факт по разрезу (столбцы)",
    "scatter_planfact": "План против факта (точечная)",
    "trend": "Тренд по периодам",
    "bar_dim": "Столбцы по разрезу",
}


def build_chart(ctx: ChartContext, chart: dict) -> dict:
    builder = BUILDERS.get(chart["type"])
    if builder is None:
        title = chart.get("title") or "График"
        return {"figure": empty_figure(title, "Неизвестный тип графика"), "title": title}
    try:
        result = builder(ctx, chart)
    except Exception as err:  # график никогда не должен «ронять» дашборд
        import logging

        logging.getLogger(__name__).exception("Ошибка построения графика %s", chart)
        title = chart.get("title") or "График"
        return {"figure": empty_figure(title, f"Не удалось построить: {err}"), "title": title, "error": str(err)}
    result["figure"] = json.loads(json.dumps(result["figure"], cls=PlotlyJSONEncoder))
    return result
