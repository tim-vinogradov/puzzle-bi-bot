"""Любые сочетания переключателей и фильтров не должны ломать графики."""
import itertools

import pytest

from app import semantic
from app.bot import resolve_period
from app.calc import make_calc
from app.dashboard import DashboardState, default_charts, default_switches, render


@pytest.mark.parametrize("dataset,intent", [("sales", "dashboard"), ("sales", "compare"), ("discounts", "dashboard"),
                                            ("clients", "dashboard")])
def test_switch_combinations(dataset, intent):
    period = resolve_period({"kind": "last_n_months", "n": 6})
    fetched = semantic.fetch_dataset(dataset, period)
    state = DashboardState(dataset, "тест", period, fetched.table, fetched.prev, fetched.links)
    state.switches = default_switches(state)
    state.charts = default_charts(dataset, intent)
    fact = state.spec.metrics[state.spec.default_metric].fact
    state.calcs = [make_calc("change", existing=[], measure=fact, lag="month")]
    state.calcs.append(make_calc("share", existing=state.calcs, measure=fact, by=state.spec.switch_dims[0]))
    dims = [d for d in state.spec.switch_dims if d in state.table.columns][:3]
    for dim, metric, grain, view in itertools.product(dims, state.spec.metrics, ["month", "quarter"], ["abs", "pct", "dyn"]):
        state.switches.update(dim=dim, metric=metric, grain=grain, view=view)
        payload = render(state)
        errors = [c["error"] for c in payload["charts"] if c.get("error")]
        assert not errors, (dim, metric, grain, view, errors)
    # графики, перестроенные на вычисления
    for chart in state.charts:
        for calc in state.calcs:
            chart["value"] = calc["id"]
            payload = render(state)
            assert not [c["error"] for c in payload["charts"] if c.get("error")], (chart["type"], calc["kind"])
        chart.pop("value")
