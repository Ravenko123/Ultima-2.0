"""Canonical trading defaults shared across live and backtest tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

__all__ = [
    "PRIMARY_TIMEFRAME_CODE",
    "PRIMARY_TIMEFRAME_LABEL",
    "WEEKDAY_SYMBOLS",
    "WEEKEND_SYMBOLS",
    "ALL_SYMBOLS",
    "DEFAULT_TELEMETRY_DIR",
    "BacktestControls",
    "DEFAULT_BACKTEST_CONTROLS",
]

# -- Timeframe defaults ----------------------------------------------------
# The live bot and the optimized backtester both assume M15 candles as their
# primary decision window. We expose both a MetaTrader-style code and a human
# label so that downstream modules do not need to duplicate tiny constants.
PRIMARY_TIMEFRAME_LABEL: str = "M15"
PRIMARY_TIMEFRAME_CODE: str = "TIMEFRAME_M15"

# -- Symbol universe -------------------------------------------------------
# Weekday FX/metal symbols plus the weekend crypto pair. Ordering matters when
# we serialise defaults to comma-separated CLI arguments, so stick to tuples.
WEEKDAY_SYMBOLS: Tuple[str, ...] = (
    "EURUSD+",
    "USDJPY+",
    "GBPUSD+",
    "GBPJPY+",
    "XAUUSD+",
)
WEEKEND_SYMBOLS: Tuple[str, ...] = ("BTCUSD",)
ALL_SYMBOLS: Tuple[str, ...] = tuple(dict.fromkeys(WEEKDAY_SYMBOLS + WEEKEND_SYMBOLS))

# -- Telemetry defaults ----------------------------------------------------
DEFAULT_TELEMETRY_DIR: Path = Path("analytics_output/ml_backtest")


@dataclass(slots=True)
class BacktestControls:
    """Friendly knobs that act as the backtester\'s control panel."""

    symbols: Tuple[str, ...]
    timeframe: str
    lookback_days: int
    trials: int
    output_directory: Path
    cache_directory: Path | None = None

    def serialise_symbols(self) -> str:
        return ",".join(self.symbols)

    def with_symbols(self, symbols: Iterable[str]) -> "BacktestControls":
        deduped = tuple(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
        return BacktestControls(
            symbols=deduped or self.symbols,
            timeframe=self.timeframe,
            lookback_days=self.lookback_days,
            trials=self.trials,
            output_directory=self.output_directory,
            cache_directory=self.cache_directory,
        )


DEFAULT_BACKTEST_CONTROLS = BacktestControls(
    symbols=ALL_SYMBOLS,
    timeframe=PRIMARY_TIMEFRAME_LABEL,
    lookback_days=120,
    trials=80,
    output_directory=DEFAULT_TELEMETRY_DIR,
    cache_directory=Path("data_cache"),
)
