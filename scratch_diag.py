import sys
import pandas as pd
sys.path.insert(0, ".")
import scratch_exp as E

raw = pd.read_csv("data/aqi/raw.csv", parse_dates=["datetime"])
raw = raw[raw["City"].isin(E.SUBSET)].sort_values(["City", "datetime"]).reset_index(drop=True)
print("raw cols:", list(raw.columns))
print("\nraw all-NaN:", [c for c in raw.columns if raw[c].isna().all()])
print("\nnon-numeric:", [c for c in raw.columns if raw[c].dtype == object])

df = E.build(raw)
print(f"\nbuilt {df.shape}")
bad = [c for c in df.columns if df[c].isna().all()]
print("all-NaN cols:", bad)
nonnum = [c for c in df.columns if df[c].dtype == object and c != "City"]
print("object cols:", nonnum)
nuniq = {c: df[c].nunique(dropna=True) for c in df.columns if c not in ("City", "datetime")}
print("constant cols:", [c for c, n in nuniq.items() if n <= 1])
