# BTC-USD market timing with an MLP: a null result

Can a small neural network predict whether Bitcoin's next N-day return is positive, using only past returns?

**Short answer:** at a 5-day horizon, no. Test accuracy was about 51%, which is indistinguishable from a coin flip, so nothing here is deployed. The more useful part was working out why an earlier 120-day backtest looked so much better than it deserved to.

The full write-up is in [`BTC_MLP_report.pdf`](BTC_MLP_report.pdf).

## Results

| Horizon | Test accuracy | Backtest (gross of costs) | Verdict |
|---|---|---|---|
| 120 days | 63.0% | Long/short +98.7%, long-only +56.2%, buy-and-hold +1.6% | Promising on paper, thin evidence |
| 5 days | ~51% | Not relied on | Coin flip, no edge |

The 120-day figures come from an earlier run (test window 2024-05-14 to 2026-07-17). Re-running with `HORIZON = 120` and `LOOKBACKS = [25, 60, 90, 120, 240]` will not reproduce them exactly: the data keeps updating and the training-boundary embargo was added later.

## Setup

- **Data:** daily BTC-USD closes from 2015 via `yfinance`, completed candles only
- **Inputs:** trailing compounded returns over several look-back windows
- **Target:** is the next `HORIZON`-day return positive? (1 = up, 0 = down)
- **Model:** MLP (25-15-10 units, ReLU, dropout 0.2, sigmoid output), binary cross-entropy, class weights, early stopping
- **Evaluation:** chronological 80/20 split with no shuffling; a signal at the close of day *t* is traded on day *t+1*
- **Strategies:** long/short, long-only, buy-and-hold, with a configurable transaction cost

## Pitfalls found along the way

1. **Silent labelling bug.** Shifting a series to build a forward-looking target leaves `NaN` in the last rows, and `NaN > 0` is `False`, so those rows quietly become "down" labels. Drop them before labelling.
2. **Partial candle.** Crypto trades 24/7, so the latest daily bar is incomplete until 00:00 UTC. The code drops it.
3. **Overlapping labels.** A 120-day label overlaps its neighbours almost entirely, so 795 test days is only about 6 or 7 independent outcomes.
4. **No baseline.** About 59% of days in the 120-day data were "up" labels, so 63% accuracy beats "always up" by only a few points.
5. **Flattering backtest.** Buy-and-hold was nearly flat over the test window, which rewards any model that happened to be short during drawdowns.

## Files

| File | Purpose |
|---|---|
| `btc_5day_mlp.ipynb` | Data download, features, model, evaluation, backtest. Saves the model and its settings |
| `live_signal.py` | Daily signal generator. Logs LONG/SHORT and the model's probability to `signal_log.csv`. **Places no orders** |
| `BTC_MLP_report.pdf` | Two-page write-up |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook btc_5day_mlp.ipynb   # run all cells; saves btc_5d_mlp.keras and btc_5d_config.json
python live_signal.py                 # run daily, shortly after 00:00 UTC
```

## Next steps

Walk-forward testing, several random seeds, simple benchmarks (logistic regression, a plain momentum rule), richer inputs (volatility, volume, funding rates), and a cost-aware backtest before anything goes near real money.

## Disclaimer

Educational research only. This is not investment advice, and past or backtested performance says nothing reliable about future results.
