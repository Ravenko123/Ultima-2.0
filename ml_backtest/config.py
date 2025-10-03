"""Configuration primitives for the ML-driven backtest orchestration layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "DataCacheConfig",
    "ExperimentTimeRange",
    "TelemetryOutputConfig",
    "ExperimentConfig",
    "resolve_time_range",
]


@dataclass(frozen=True, slots=True)
class DataCacheConfig:
    """Information about where historical market data should be sourced from."""

    symbols: tuple[str, ...]
    timeframe: str
    cache_dir: Path = Path("data_cache")
    source: str = "mt5"  # either "mt5" (live pull) or "parquet" (pre-cached files)
    use_cache: bool = True
    max_age: timedelta = timedelta(days=1)

    def cache_path_for(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.replace('/', '_').replace('+', '')}_{self.timeframe}.parquet"


@dataclass(frozen=True, slots=True)
class ExperimentTimeRange:
    """Defines the segment of history to replay for an experiment."""

    start: datetime | None = None
    end: datetime | None = None
    lookback_days: int | None = 90

    def resolve(self, *, now: datetime | None = None) -> tuple[datetime, datetime]:
        now = now or datetime.now(timezone.utc)
        end = self.end or now
        if self.start is not None:
            start = self.start
        elif self.lookback_days is not None:
            start = end - timedelta(days=self.lookback_days)
        else:
            raise ValueError("Either start or lookback_days must be provided")
        if start >= end:
            raise ValueError("Resolved start time must be before end time")
        return start, end


@dataclass(frozen=True, slots=True)
class TelemetryOutputConfig:
    """Where to persist telemetry artifacts generated during experiments."""

    directory: Path = Path("analytics_output") / "ml_backtest"
    format: str = "jsonl"
    retain_runs: int = 10

    def run_directory(self, run_id: str) -> Path:
        return self.directory / run_id


@dataclass(slots=True)
class ExperimentConfig:
    """Composite configuration for an optimization experiment."""

    name: str
    data: DataCacheConfig
    time_range: ExperimentTimeRange = field(default_factory=ExperimentTimeRange)
    telemetry: TelemetryOutputConfig = field(default_factory=TelemetryOutputConfig)
    initial_balance: float = 10_000.0
    personas: Sequence[str] = field(default_factory=tuple)
    risk_presets: Sequence[str] = field(default_factory=tuple)
    max_concurrent_positions: int | None = None
    notes: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        symbols: Iterable[str],
        timeframe: str = "H1",
        name: str | None = None,
    ) -> "ExperimentConfig":
        """Build an experiment config using environment hints with sane defaults."""

        cache_dir = Path(os.getenv("ULTIMA_MLBT_CACHE_DIR", "data_cache"))
        telemetry_dir = Path(os.getenv("ULTIMA_MLBT_OUTPUT_DIR", "analytics_output/ml_backtest"))
        lookback = int(os.getenv("ULTIMA_MLBT_LOOKBACK_DAYS", "120"))
        balance = float(os.getenv("ULTIMA_MLBT_INITIAL_BALANCE", "10000"))
        persona_raw = os.getenv("ULTIMA_MLBT_PERSONAS", "")
        personas: tuple[str, ...] = tuple(filter(None, (item.strip() for item in persona_raw.split(","))))
        risk_raw = os.getenv("ULTIMA_MLBT_RISK_PRESETS", "")
        risk_presets: tuple[str, ...] = tuple(filter(None, (item.strip() for item in risk_raw.split(","))))

        data_cfg = DataCacheConfig(
            symbols=tuple(symbols),
            timeframe=timeframe,
            cache_dir=cache_dir,
            use_cache=os.getenv("ULTIMA_MLBT_USE_CACHE", "1") != "0",
        )
        range_cfg = ExperimentTimeRange(lookback_days=lookback)
        telemetry_cfg = TelemetryOutputConfig(directory=telemetry_dir)
        return cls(
            name=name or os.getenv("ULTIMA_MLBT_EXPERIMENT_NAME", "mlbt_experiment"),
            data=data_cfg,
            time_range=range_cfg,
            telemetry=telemetry_cfg,
            initial_balance=balance,
            personas=personas,
            risk_presets=risk_presets,
        )


def resolve_time_range(range_cfg: ExperimentTimeRange, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Helper to resolve a time range independent of ExperimentConfig."""

    return range_cfg.resolve(now=now)
