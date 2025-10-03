# ML Backtest Orchestrator (Next-Gen)

This document describes the new machine-learning driven backtesting stack that powers
parameter discovery for the live Ultima bot.

## Overview

The goal of the new framework is to close the loop between synthetic experiments and
live telemetry:

1. **Market data acquisition** – pull or reuse cached OHLCV data via
   `MarketDataCache` with transparent parquet caching.
2. **Experiment execution** – replay price history through the
   `ExperimentRunner`, which mirrors the live telemetry schema and emits synthetic
   `trade_closed` events for downstream analytics.
3. **Optimization loop** – explore configuration values with the
   `ExperimentOptimizer` (currently a lightweight random search) and score trials
   using metrics that align with live deployment goals.
4. **Model feedback** – generated telemetry can be fed into the existing
   analytics pipeline (`analytics/auto_train.py`) for classifier retraining.

The simulator now encodes an ATR-aware moving-average crossover model with
configurable stop-loss/take-profit, slippage, and spread assumptions. It remains
modular so that shared live-agent components can be swapped in as they evolve.

## Package layout

```
ml_backtest/
  __init__.py            # public exports for configs, parameter space, results
  config.py              # dataclasses for experiment/data/telemetry settings
  data_loader.py         # cache-aware MT5/parquet data ingestion helpers
  parameter_space.py     # reusable building blocks for parameter sampling
  experiment_runner.py   # ATR + MA based replay engine emitting live-style telemetry
  optimizer.py           # random-search orchestrator with plateau detection
  cli.py                 # entry point for running optimization sessions
```

## Quick start

Run an optimization session from the project root:

```powershell
python -m ml_backtest.cli --symbols BTCUSD --timeframe H1 --trials 5 --lookback-days 30
```

Key flags:

- `--symbols`: comma-separated list of instruments to simulate.
- `--timeframe`: MT5 timeframe code (e.g., `M15`, `H1`).
- `--trials`: number of randomized parameter samples to evaluate.
- `--start` / `--end`: ISO timestamps for explicit windows (defaults to a rolling
  lookback).
- `--cache-dir`: directory for parquet caches (default `data_cache`).
- `--output-dir`: telemetry output location (`analytics_output/ml_backtest`).

The CLI prints a JSON summary describing the best trial and the sampled
parameters. Telemetry for each run is stored under
`analytics_output/ml_backtest/<run_id>/telemetry.jsonl` and can be ingested with
`analytics.telemetry_ingest`.

## Next steps

- Share sizing, guard, and persona presets with the live stack so experiments
  replay identical risk logic.
- Expand the optimizer beyond random search (Bayesian optimisation, Hyperband)
  and expose multi-objective scoring.
- Emit equity curve telemetry and richer analytics artifacts for direct model
  ingestion.
- Automate promotion of winning configurations to a staging manifest that can be
  reviewed before deployment to live trading.
- Integrate analytics auto-training so fresh telemetry retrains the trade
  classifier as part of the optimization pipeline.
