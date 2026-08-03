"""CPCB (India) National Air Quality Index computation from pollutant concentrations."""
import numpy as np
import pandas as pd

# (c_low, c_high, i_low, i_high) per CPCB National AQI, CPCB 2014.
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
              (90, 120, 201, 300), (120, 250, 301, 400), (250, 1000, 401, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
             (250, 350, 201, 300), (350, 430, 301, 400), (430, 1000, 401, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
            (180, 280, 201, 300), (280, 400, 301, 400), (400, 1000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
            (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2400, 401, 500)],
    "O3": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
           (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
    "NH3": [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
            (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)],
    # CO breakpoints are in mg/m3, not ug/m3.
    "CO": [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
           (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
}

# CPCB averaging windows: 24h for particulates/NO2/SO2/NH3, 8h for CO/O3.
AVERAGING_HOURS = {"PM2.5": 24, "PM10": 24, "NO2": 24, "SO2": 24, "NH3": 24, "CO": 8, "O3": 8}

AQI_CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                  (300, "Poor"), (400, "Very Poor"), (np.inf, "Severe")]


def sub_index(concentration: pd.Series, pollutant: str) -> pd.Series:
    """Piecewise-linear CPCB sub-index for one pollutant."""
    bp = BREAKPOINTS[pollutant]
    c = pd.to_numeric(concentration, errors="coerce")
    out = pd.Series(np.nan, index=c.index, dtype=float)

    for c_lo, c_hi, i_lo, i_hi in bp:
        m = (c > c_lo) & (c <= c_hi)
        out[m] = i_lo + (i_hi - i_lo) * (c[m] - c_lo) / (c_hi - c_lo)

    out[c == 0] = 0.0
    # Above the top breakpoint CPCB caps the index at 500.
    out[c > bp[-1][1]] = 500.0
    out[c < 0] = np.nan
    return out


def rolling_average(df: pd.DataFrame, pollutant: str, group_col: str = "City") -> pd.Series:
    """CPCB-mandated trailing average of a pollutant, computed per city."""
    hours = AVERAGING_HOURS[pollutant]
    return (df.groupby(group_col, sort=False)[pollutant]
              .transform(lambda s: s.rolling(hours, min_periods=max(2, hours // 2)).mean()))


def compute_aqi(df: pd.DataFrame, group_col: str = "City") -> pd.DataFrame:
    """Compute CPCB AQI. Assumes df is sorted by (group_col, datetime).

    Returns df with per-pollutant sub-index columns, AQI, AQI_Category and
    Dominant_Pollutant. CPCB requires >=3 pollutants with PM2.5 or PM10 present.
    """
    df = df.copy()
    sub_cols = []

    for pollutant in BREAKPOINTS:
        if pollutant not in df.columns:
            continue
        conc = rolling_average(df, pollutant, group_col)
        if pollutant == "CO":
            conc = conc / 1000.0  # Open-Meteo reports CO in ug/m3; CPCB uses mg/m3.
        col = f"SubIndex_{pollutant}"
        df[col] = sub_index(conc, pollutant)
        sub_cols.append(col)

    subs = df[sub_cols]
    has_particulate = df[[c for c in ("SubIndex_PM2.5", "SubIndex_PM10") if c in sub_cols]].notna().any(axis=1)
    valid = (subs.notna().sum(axis=1) >= 3) & has_particulate

    df["AQI"] = subs.max(axis=1).where(valid)
    df["Dominant_Pollutant"] = pd.Series(pd.NA, index=df.index, dtype="object")
    if valid.any():
        # idxmax raises on all-NaN rows, so restrict it to valid rows.
        dominant = subs.loc[valid].idxmax(axis=1).str.replace("SubIndex_", "", regex=False)
        df.loc[valid, "Dominant_Pollutant"] = dominant

    bounds = [b for b, _ in AQI_CATEGORIES]
    labels = [lab for _, lab in AQI_CATEGORIES]
    df["AQI_Category"] = pd.cut(df["AQI"], bins=[-np.inf] + bounds, labels=labels)
    return df
