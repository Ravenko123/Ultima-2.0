# ML Backtesting Harness

Ultima's ML backtesting package orchestrates parameter sweeps, telemetry capture, and experiment scoring for the strategy lab. The stack couples a configurable dataset cache, a parameter-space builder, and the `ExperimentRunner` simulator that emits JSONL telemetry compatible with live analytics tooling.

## Control panel defaults

The backtester now ships with a "controls panel" you can edit in one place: `trading_defaults.py`. It exposes a `BacktestControls` dataclass that defines the canonical symbol universe, primary timeframe, lookback window, number of trials, and telemetry/cache directories. By default it mirrors the live supervisor configuration—M15 candles over the full weekday FX + weekend BTC roster:

- Symbols: `EURUSD+`, `USDJPY+`, `GBPUSD+`, `GBPJPY+`, `XAUUSD+`, `BTCUSD`
- Timeframe: `M15`
- Lookback days: `120`
- Trials per optimizer run: `80`
- Telemetry directory: `analytics_output/ml_backtest`

Editing the dataclass automatically updates the CLI defaults; you can still override anything ad-hoc with command-line flags when running `python -m ml_backtest.cli`.

### Multi-timeframe confirmation

To stay in lockstep with the live supervisor, the simulator now supports an H1 confirmation layer. The parameter space exposes three new knobs:

- `mtf_confirmation`: toggle (`on`/`off`) for the higher-timeframe guard.
- `mtf_short_window`: short moving-average length (default `5`) applied to resampled H1 closes.
- `mtf_long_window`: long moving-average length (default `10`).

When enabled, the backtester resamples the primary timeframe data into H1 closes, computes the two MAs, and only admits M15 signals that agree with the higher-timeframe bias. Rejected signals are logged to telemetry (`mtf_filter_reject`) and surfaced in the scorecard metric `mtf_filter_rejections` so optimizations can trade off strictness versus opportunity.

## Risk presets & personas

Backtests can now blend configurable risk presets with behavioural personas to explore more realistic guardrails:

- **Risk presets** adjust volume, ATR targets, slippage, and the maximum acceptable drawdown when scoring a trial. Built-in profiles include `conservative`, `balanced`, `aggressive`, and `ultra`, plus a neutral fallback. Aliases such as `moderate` → `balanced` or `max` → `ultra` are supported.
- **Personas** nudge indicator windows and holding periods while scaling volume and ATR brackets. Defaults cover `scalper`, `swing`, `position`, and `mean_reverter` with a neutral baseline. Synonyms like `intraday` → `scalper` resolve automatically.

When the CLI builds a parameter space, it automatically adds categorical parameters named `risk_preset` and `persona` when the experiment configuration exposes them. The simulator reads the sampled values on every symbol replay and:

- Applies multiplier adjustments before sizing trades.
- Embeds the chosen persona and preset into each `trade_closed` telemetry event.
- Emits `equity_curve_point` snapshots annotated with the current equity, drawdown, persona, and risk preset.
- Scores trials with the drawdown ceiling supplied by the active risk profile and surfaces the profile multipliers inside the metrics map.

### Quick example

```python
from ml_backtest import ExperimentConfig, ExperimentRunner, MarketDataCache
from ml_backtest.cli import build_parameter_space

config = ExperimentConfig(
    name="lab",
    data=DataCacheConfig(symbols=("EURUSD",), timeframe="H1"),
    personas=("scalper", "swing"),
    risk_presets=("conservative", "aggressive"),
)
space = build_parameter_space(config)
sample = next(space.iter_samples(1))
cache = MarketDataCache(config.data)
result = ExperimentRunner(config, cache).run(sample)
print(result.scorecard.metrics["risk_drawdown_limit_pct"])  # -> preset-aware threshold
```

## Telemetry enrichment

Runs now start with a `run_context` event capturing the sampled parameters and resolved profile details. After each symbol replay the runner publishes incremental `equity_curve_point` events so downstream analytics can chart per-trial equity without recomputing it from the raw trades.

Telemetry is persisted under the experiment's `TelemetryOutputConfig` directory, mirroring the live JSONL format so dashboards and ingestion scripts work unchanged.
