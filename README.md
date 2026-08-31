# Kalshi DOGE 15-minute maker research

This repository covers the valuation and historical execution testing of a KXDOGE15M maker strategy:

1. Apply the contract-defined $[T-60,T)$ settlement window and use exchange-recorded targets for primary labels.
2. Estimate settlement-aware probabilities using a zero-drift, locally estimated-volatility structural model.
3. Calibrate the forecast distribution out of sample using prior-days-only empirical residuals.
4. Replay inventory-adjusted maker quotes using trade-through and queue-aware fill assumptions.

The primary probability model uses a discrete arithmetic-average Asian-digital approximation. It conditions on settlement samples already observed in the final minute and moment-matches the sum of unknown samples to a lognormal distribution. The repository also supports an older geometric-average approximation, deterministic Sobol Monte Carlo evaluation, and a pure empirical alternative. Published results compare 1x and 10x last-minute replays, full-market quoting, and a temporal out-of-sample period.

## Documentation

The research documentation is split into short, topic-focused pages:

- [Documentation overview](docs/README.md)
- [Probability model](docs/model.md)
- [Backtesting and execution](docs/backtesting.md)
- [Current results and interpretations](docs/results.md)

## Run

The `.env` file must contain `DATABASE_URL`. Create the project virtual environment, install dependencies, and run directly from `src`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m kxdoge_backtest
```

Useful options:

```bash
PYTHONPATH=src .venv/bin/python -m kxdoge_backtest \
  --model arithmetic \
  --horizons 600,300,120,60,30 \
  --vol-lookback 300 \
  --min-training-days 7
```

Each run creates a timestamped directory under `artifacts/` containing:

- `settlement_replay.csv`: reconstructed targets, settlement averages, and outcomes
- `diffusion_features.csv`: point-in-time baseline features and probabilities
- `walk_forward_predictions.csv`: out-of-sample calibrated probabilities
- `metrics.csv`: log loss, Brier score, calibration error, AUC, and accuracy
- `paired_comparisons.csv`: same-market score deltas with day-block bootstrap intervals
- `calibration_bins.csv`: reliability-curve data
- `report.md` and `summary.json`: concise run summaries

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Event-time maker backtest

After generating the probability artifacts, run:

```bash
PYTHONPATH=src .venv/bin/python -m kxdoge_backtest.maker_cli \
  --fair-model arithmetic \
  --probability-artifacts artifacts/<probability-run-directory>
```

This replay uses full book snapshots and deltas, public liquidity-taking trades,
visible queue ahead, cancel latency, inventory limits, settlement P&L, and
one- and five-second post-fill markouts. It runs conservative trade-through,
queue-aware, latency, spread, optimistic-touch, and hypothetical-fee scenarios.
The report also includes UTC-day block-bootstrap confidence intervals and
time-to-expiry P&L decomposition.
