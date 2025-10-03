"""Command-line helpers for running ML-driven backtest experiments."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .config import (
    DataCacheConfig,
    ExperimentConfig,
    ExperimentTimeRange,
    TelemetryOutputConfig,
)
from .data_loader import MarketDataCache
from .experiment_runner import ExperimentRunner
from .optimizer import ExperimentOptimizer, OptimizationSettings
from .parameter_space import CategoricalParameter, ContinuousParameter, DiscreteParameter, ParameterSpace
from trading_defaults import DEFAULT_BACKTEST_CONTROLS

logger = logging.getLogger(__name__)
CONTROLS = DEFAULT_BACKTEST_CONTROLS


def _parse_symbols(raw: str | None) -> Sequence[str]:
    if not raw:
        return CONTROLS.symbols
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_parameter_space(config: ExperimentConfig) -> ParameterSpace:
    risk_choices: Sequence[str]
    if config.risk_presets:
        risk_choices = tuple(dict.fromkeys(config.risk_presets))
    else:
        risk_choices = ("conservative", "balanced", "aggressive")

    parameters: list = [
        ContinuousParameter(name="base_volume", lower=0.015, upper=0.12),
        ContinuousParameter(name="pip_value", lower=40.0, upper=450.0),
        ContinuousParameter(name="commission_per_lot", lower=-4.0, upper=-0.05),
        DiscreteParameter(name="fast_window", values=tuple(range(5, 31))),
        DiscreteParameter(name="slow_window", values=tuple(range(20, 121))),
        DiscreteParameter(name="atr_window", values=tuple(range(10, 61))),
        DiscreteParameter(name="max_holding_bars", values=tuple(range(2, 49))),
        ContinuousParameter(name="take_profit_atr", lower=0.5, upper=4.5),
        ContinuousParameter(name="stop_loss_atr", lower=0.5, upper=3.0),
        ContinuousParameter(name="spread_bps", lower=0.0, upper=40.0),
        ContinuousParameter(name="slippage_bps", lower=0.0, upper=20.0),
        ContinuousParameter(name="breakeven_atr_trigger", lower=0.0, upper=3.0),
        ContinuousParameter(name="trailing_start_atr", lower=0.0, upper=5.0),
        ContinuousParameter(name="trailing_distance_atr", lower=0.25, upper=4.0),
        ContinuousParameter(name="drawdown_limit_pct", lower=15.0, upper=55.0),
        CategoricalParameter(name="mtf_confirmation", choices=("on", "off")),
        DiscreteParameter(name="mtf_short_window", values=tuple(range(3, 11))),
        DiscreteParameter(name="mtf_long_window", values=tuple(range(8, 25))),
        CategoricalParameter(name="risk_preset", choices=risk_choices),
    ]

    persona_choices: Sequence[str] = tuple(dict.fromkeys(config.personas)) if config.personas else ()
    if persona_choices:
        parameters.append(CategoricalParameter(name="persona", choices=persona_choices))

    return ParameterSpace(parameters)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML backtest optimization trials")
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols",
        default=CONTROLS.serialise_symbols(),
    )
    parser.add_argument(
        "--timeframe",
        help="Timeframe code (e.g. H1, M30)",
        default=CONTROLS.timeframe,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=CONTROLS.trials,
        help="Number of optimization trials",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=CONTROLS.lookback_days,
        help="Historical lookback window in days",
    )
    parser.add_argument("--start", help="ISO timestamp for start of backtest window", default=None)
    parser.add_argument("--end", help="ISO timestamp for end of backtest window", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(CONTROLS.output_directory),
        help="Telemetry output directory",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(CONTROLS.cache_directory or "data_cache"),
        help="Directory for cached market data",
    )
    parser.add_argument("--experiment-name", default="mlbt_experiment", help="Label for this optimization run")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args(argv)


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    symbols = _parse_symbols(args.symbols)
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    time_range = ExperimentTimeRange(start=start, end=end, lookback_days=args.lookback_days)
    data_cfg = DataCacheConfig(
        symbols=tuple(symbols),
        timeframe=args.timeframe,
        cache_dir=Path(args.cache_dir),
    )
    telemetry_cfg = TelemetryOutputConfig(directory=Path(args.output_dir))
    return ExperimentConfig(
        name=args.experiment_name,
        data=data_cfg,
        time_range=time_range,
        telemetry=telemetry_cfg,
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = build_experiment_config(args)
    cache = MarketDataCache(config.data)
    runner = ExperimentRunner(config, cache)
    parameter_space = build_parameter_space(config)
    settings = OptimizationSettings(
        trials=args.trials,
        random_seed=args.seed,
    )
    optimizer = ExperimentOptimizer(runner, parameter_space, settings)
    summary = optimizer.optimize()

    best = summary.best_result
    if best is None:
        logger.warning("No valid trials completed. Check data availability or constraints.")
        return {"best": None, "history": len(summary.history)}

    best_metric = best.metric(settings.primary_metric)
    report = {
        "best_metric": best_metric,
        "primary_metric": settings.primary_metric,
        "run_id": best.artifacts[0].run_id if best.artifacts else None,
        "metrics": best.scorecard.metrics,
        "constraints": best.scorecard.constraints,
        "sample": best.sample.as_dict(),
    }

    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:  # pragma: no cover - CLI wiring
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
