"""Корректность вычислений и сохранения."""
import pandas as pd
import pytest

from app.calc import aggregate, make_calc, materialize

MEASURES = ["plan", "fact"]


@pytest.fixture()
def frame():
    rows = []
    for month in ("2026-01-01", "2026-02-01", "2026-03-01"):
        for region, plan, fact in (("A", 100.0, 90.0), ("B", 300.0, 330.0)):
            rows.append({"period": pd.Timestamp(month), "region": region, "plan": plan, "fact": fact * (1 + 0.1 * int(month[6]))})
    return pd.DataFrame(rows)


def test_ratio_recomputed_after_aggregation(frame):
    calcs = [make_calc("expr", existing=[], expr="fact / plan * 100", label="Выполнение, %", unit="%")]
    agg = aggregate(frame, [], [calcs[0]["id"]], calcs, measures=MEASURES)
    expected = frame["fact"].sum() / frame["plan"].sum() * 100
    assert agg[calcs[0]["id"]].iloc[0] == pytest.approx(expected)  # отношение сумм, а не среднее отношений


def test_share_sums_to_100(frame):
    calcs = [make_calc("share", existing=[], measure="fact", by="region")]
    agg = aggregate(frame, ["region"], [calcs[0]["id"]], calcs, measures=MEASURES)
    assert agg[calcs[0]["id"]].sum() == pytest.approx(100)


def test_mom_change_and_cumsum(frame):
    calcs = [make_calc("change", existing=[], measure="fact", lag="month"),
             make_calc("cumsum", existing=[], measure="fact")]
    agg = aggregate(frame, ["period"], [c["id"] for c in calcs], calcs, measures=MEASURES).sort_values("period")
    totals = agg["fact"].tolist()
    assert agg[calcs[0]["id"]].iloc[1] == pytest.approx((totals[1] / totals[0] - 1) * 100)
    assert agg[calcs[1]["id"]].iloc[-1] == pytest.approx(sum(totals))


def test_materialize_rows(frame):
    calcs = [make_calc("expr", existing=[], expr="fact - plan", label="Откл", unit="₽"),
             make_calc("rank", existing=[], measure="fact", by="region")]
    out = materialize(frame, calcs, dims=["period", "region"], measures=MEASURES)
    assert (out[calcs[0]["id"]] == out["fact"] - out["plan"]).all()
    assert set(out[calcs[1]["id"]]) == {1.0, 2.0}


def test_storage_roundtrip(frame):
    from app import storage

    meta = {"dataset": "sales", "query": "тест", "name": "Тестовый"}
    info = storage.save_new("Тестовый дашборд", meta, (frame, None, None))
    loaded_info, loaded_meta, frames = storage.load(info["id"])
    assert loaded_meta["query"] == "тест"
    pd.testing.assert_frame_equal(frames[0].reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False)
    storage.save_version(info["id"], {**meta, "query": "v2"}, (frame, None, None))
    assert [v["version"] for v in storage.versions(info["id"])] == [1, 2]
    assert storage.load(info["id"], 1)[1]["query"] == "тест"
