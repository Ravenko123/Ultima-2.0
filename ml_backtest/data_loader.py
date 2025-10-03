"""Historical market data loading with local caching support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

try:  # Lazy import to keep module importable in environments without MetaTrader5
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover - optional dependency guard
    mt5 = None  # type: ignore
    _mt5_import_error = exc
else:
    _mt5_import_error = None

from .config import DataCacheConfig

__all__ = ["MarketDataCache", "load_time_series"]

logger = logging.getLogger(__name__)

_MT5_INITIALIZED = False


def _ensure_mt5_initialized() -> None:
    global _MT5_INITIALIZED
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not available") from _mt5_import_error
    if not _MT5_INITIALIZED:
        if not mt5.initialize():
            code, message = mt5.last_error()
            raise RuntimeError(f"MetaTrader5 initialize failed: {code} | {message}")
        _MT5_INITIALIZED = True


def _resolve_timeframe_code(timeframe: str) -> int:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not available") from _mt5_import_error
    attribute = f"TIMEFRAME_{timeframe.upper()}"
    code = getattr(mt5, attribute, None)
    if code is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return code


def _read_cached_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - corrupted cache fallback
        logger.warning("Failed to read cache %s: %s", path, exc)
        return None


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@dataclass(slots=True)
class MarketDataCache:
    config: DataCacheConfig

    def load_symbol(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        """Load historical bars for a symbol, refreshing cache if needed."""

        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware UTC datetimes")

        cache_path = self.config.cache_path_for(symbol)
        frame = _read_cached_frame(cache_path) if self.config.use_cache else None
        if frame is not None and "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")

        refresh_needed = True
        if frame is not None:
            latest = frame["time"].max() if not frame.empty else None
            if isinstance(latest, pd.Timestamp):
                latest = latest.to_pydatetime()
            if latest and latest >= end and (datetime.now(timezone.utc) - latest.replace(tzinfo=timezone.utc)) <= self.config.max_age:
                refresh_needed = False

        if refresh_needed:
            frame = self._pull_from_source(symbol, start=start, end=end)
            if frame.empty:
                raise RuntimeError(f"No market data returned for {symbol}")
            if self.config.use_cache:
                _write_cache(cache_path, frame)
        else:
            frame = frame.copy()

        mask = (frame["time"] >= pd.Timestamp(start)) & (frame["time"] <= pd.Timestamp(end))
        return frame.loc[mask].reset_index(drop=True)

    def load_all(self, *, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        datasets: Dict[str, pd.DataFrame] = {}
        for symbol in self.config.symbols:
            datasets[symbol] = self.load_symbol(symbol, start=start, end=end)
        return datasets

    def _pull_from_source(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        source = self.config.source.lower()
        if source == "mt5":
            return self._pull_mt5(symbol, start=start, end=end)
        if source == "parquet":
            cache_path = self.config.cache_path_for(symbol)
            frame = _read_cached_frame(cache_path)
            if frame is None:
                raise RuntimeError(f"Parquet cache missing for {symbol}: {cache_path}")
            return frame
        raise ValueError(f"Unsupported data source: {self.config.source}")

    def _pull_mt5(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        _ensure_mt5_initialized()
        timeframe_code = _resolve_timeframe_code(self.config.timeframe)
        rates = mt5.copy_rates_range(symbol, timeframe_code, start, end)
        if rates is None:
            code, message = mt5.last_error()
            raise RuntimeError(f"MT5 copy_rates_range failed for {symbol}: {code} | {message}")
        frame = pd.DataFrame(rates)
        if frame.empty:
            return frame
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        metadata = {
            "symbol": symbol,
            "timeframe": self.config.timeframe,
            "source": "mt5",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        frame.attrs["metadata"] = metadata
        return frame


def load_time_series(
    cache: MarketDataCache,
    *,
    start: datetime,
    end: datetime,
    symbols: Iterable[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper to load a subset of symbols."""

    if symbols is None:
        symbols = cache.config.symbols
    datasets: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        datasets[symbol] = cache.load_symbol(symbol, start=start, end=end)
    return datasets
