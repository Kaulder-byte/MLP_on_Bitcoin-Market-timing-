"""Daily BTC signal. Run once a day, shortly after 00:00 UTC (when the daily candle closes).

Needs btc_5d_mlp.keras and btc_5d_config.json, both saved by the notebook.
Appends one row per day to signal_log.csv. It places no orders.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf

cfg = json.load(open("btc_5d_config.json"))
model = tf.keras.models.load_model("btc_5d_mlp.keras")

raw = yf.download(cfg["ticker"], period="2y", interval="1d", auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
if raw.index.tz is not None:
    raw.index = raw.index.tz_localize(None)
raw = raw.sort_index()

# Drop today's still-forming candle: only completed daily closes, exactly as in training
today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
raw = raw[raw.index < today_utc]


def compound(x):
    return 100 * (np.prod(1 + x / 100) - 1)


ret = raw["Close"].pct_change() * 100
feats = pd.DataFrame({f"Ret{n}_i": ret.rolling(n).apply(compound, raw=True) for n in cfg["lookbacks"]})
latest = feats[cfg["features"]].dropna().iloc[[-1]].astype("float32")

prob_up = float(model.predict(latest, verbose=0)[0, 0])
signal = "LONG" if prob_up > 0.5 else "SHORT"
bar_date = latest.index[0].date()
close = float(raw["Close"].loc[latest.index[0]])

log = Path("signal_log.csv")
if log.exists() and str(bar_date) in pd.read_csv(log)["bar_date"].astype(str).values:
    print(f"{bar_date} already logged, nothing to do.")
else:
    row = pd.DataFrame([{"bar_date": bar_date, "close": round(close, 2),
                         "prob_up": round(prob_up, 4), "signal": signal}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
    print(row.to_string(index=False))
