import numpy as np
import pandas as pd
import pytest

from src.aqi.cpcb import compute_aqi, sub_index


def test_sub_index_breakpoint_anchors():
    # CPCB PM2.5: 30 ug/m3 -> 50, 60 -> 100, 90 -> 200.
    s = sub_index(pd.Series([30.0, 60.0, 90.0]), "PM2.5")
    assert s.tolist() == pytest.approx([50.0, 100.0, 200.0])


def test_sub_index_interpolates_linearly():
    # Midpoint of the 30-60 band maps to the midpoint of 51-100.
    assert sub_index(pd.Series([45.0]), "PM2.5")[0] == pytest.approx(75.5, abs=0.5)


def test_sub_index_caps_and_rejects_negatives():
    assert sub_index(pd.Series([5000.0]), "PM2.5")[0] == 500.0
    assert np.isnan(sub_index(pd.Series([-1.0]), "PM2.5")[0])


def test_pm25_concentration_is_not_the_aqi():
    """The bug this rewrite fixes: 55 ug/m3 of PM2.5 is AQI ~92, not 55."""
    assert sub_index(pd.Series([55.0]), "PM2.5")[0] == pytest.approx(91.7, abs=1.0)


def _frame(hours, **cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame({
        "City": ["Delhi"] * n,
        "datetime": pd.date_range("2026-01-01", periods=n, freq="h"),
        **cols,
    })


def test_compute_aqi_takes_worst_pollutant():
    n = 30
    df = _frame(n, **{"PM2.5": [20.0] * n, "PM10": [300.0] * n, "NO2": [10.0] * n})
    out = compute_aqi(df)
    last = out.iloc[-1]
    assert last["Dominant_Pollutant"] == "PM10"
    assert 200 < last["AQI"] <= 300


def test_compute_aqi_requires_three_pollutants():
    n = 30
    df = _frame(n, **{"PM2.5": [50.0] * n, "PM10": [50.0] * n})
    assert compute_aqi(df)["AQI"].isna().all()


def test_compute_aqi_needs_a_particulate():
    n = 30
    df = _frame(n, **{"NO2": [50.0] * n, "SO2": [50.0] * n, "O3": [50.0] * n})
    assert compute_aqi(df)["AQI"].isna().all()


def test_co_converted_from_ug_to_mg():
    n = 30
    # 1500 ug/m3 CO = 1.5 mg/m3 -> sub-index ~75, i.e. inside the 51-100 band.
    df = _frame(n, **{"PM2.5": [1.0] * n, "PM10": [1.0] * n, "CO": [1500.0] * n})
    out = compute_aqi(df)
    assert out["SubIndex_CO"].iloc[-1] == pytest.approx(75.5, abs=1.0)


def test_rolling_average_is_per_city():
    n = 30
    df = pd.DataFrame({
        "City": ["Delhi"] * n + ["Mumbai"] * n,
        "datetime": list(pd.date_range("2026-01-01", periods=n, freq="h")) * 2,
        "PM2.5": [200.0] * n + [10.0] * n,
        "PM10": [200.0] * n + [10.0] * n,
        "NO2": [50.0] * n + [5.0] * n,
    })
    out = compute_aqi(df)
    delhi = out[out["City"] == "Delhi"]["AQI"].iloc[-1]
    mumbai = out[out["City"] == "Mumbai"]["AQI"].iloc[-1]
    assert delhi > mumbai * 3


def test_ozone_uses_max_8h_average_not_latest():
    """CPCB scores O3 on the day's peak 8h mean, so AQI must not fall away
    with the evening ozone decline the way a plain trailing mean does."""
    n = 60
    # Ozone peaks mid-series then collapses to near zero.
    o3 = [20.0] * 20 + [200.0] * 12 + [5.0] * 28
    df = _frame(n, **{"PM2.5": [1.0] * n, "PM10": [1.0] * n, "NO2": [1.0] * n, "O3": o3})
    out = compute_aqi(df)

    peak = out["SubIndex_O3"].iloc[31]
    # 19h after the peak -- still inside the 24h window, so it must be retained
    # even though current ozone has collapsed to 5 ug/m3.
    later = out["SubIndex_O3"].iloc[50]
    assert peak > 150
    assert later == pytest.approx(peak, rel=0.01)


def test_ozone_peak_expires_after_24h():
    n = 90
    o3 = [20.0] * 10 + [200.0] * 12 + [5.0] * 68
    df = _frame(n, **{"PM2.5": [1.0] * n, "PM10": [1.0] * n, "NO2": [1.0] * n, "O3": o3})
    out = compute_aqi(df)
    # Beyond the 24h window the stale peak must no longer drive the sub-index.
    assert out["SubIndex_O3"].iloc[-1] < out["SubIndex_O3"].iloc[21] / 2
