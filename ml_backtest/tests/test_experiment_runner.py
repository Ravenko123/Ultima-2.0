from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_backtest.cli import build_parameter_space
from ml_backtest.config import (
    DataCacheConfig,
    ExperimentConfig,
    ExperimentTimeRange,
    TelemetryOutputConfig,
)
from ml_backtest.experiment_runner import ExperimentRunner
from ml_backtest.parameter_space import ParameterSample
from ml_backtest.risk import resolve_persona_profile, resolve_risk_profile


class StubCache:
    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        self._datasets = datasets

    def load_all(self, *, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        return self._datasets


def _make_dataset() -> pd.DataFrame:
    periods = 240
    times = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="1h", tz="UTC")
    trend = np.linspace(-0.02, 0.02, periods)
    wave = np.sin(np.linspace(0.0, 32.0, periods)) * 0.05
    close = 50.0 + trend + wave
    high = close + 0.07
    low = close - 0.07
    return pd.DataFrame({"time": times, "close": close, "high": high, "low": low})


def _make_crypto_dataset() -> pd.DataFrame:
    base = _make_dataset()
    close = 60_000.0 + (base["close"] - base["close"].mean()) * 9_000.0
    high = close + 180.0
    low = close - 180.0
    return pd.DataFrame({"time": base["time"], "close": close, "high": high, "low": low})


def test_runner_generates_metrics(tmp_path: Path) -> None:
    frame = _make_dataset()
    start = frame["time"].iloc[0].to_pydatetime()
    end = frame["time"].iloc[-1].to_pydatetime()
    config = ExperimentConfig(
        name="unit_test",
        data=DataCacheConfig(symbols=("TEST",), timeframe="H1", use_cache=False),
        time_range=ExperimentTimeRange(start=start, end=end),
        telemetry=TelemetryOutputConfig(directory=tmp_path),
        initial_balance=5_000.0,
    )
    cache = StubCache({"TEST": frame})
    runner = ExperimentRunner(config, cache)  # type: ignore[arg-type]
    sample = ParameterSample(
        values={
            "base_volume": 0.08,
            "pip_value": 280.0,
            "commission_per_lot": -0.4,
            "fast_window": 4,
            "slow_window": 16,
            "atr_window": 14,
            "max_holding_bars": 24,
            "take_profit_atr": 1.6,
            "stop_loss_atr": 1.0,
            "spread_bps": 2.0,
            "slippage_bps": 1.0,
            "risk_preset": "balanced",
        }
    )

    result = runner.run(sample)

    metrics = result.scorecard.metrics
    assert metrics.get("trade_count", 0.0) > 0
    assert metrics.get("final_equity", config.initial_balance) != config.initial_balance
    assert 0.0 <= metrics.get("max_drawdown_pct", 0.0) <= 100.0
    risk_profile = resolve_risk_profile("balanced")
    assert metrics.get("risk_volume_multiplier") == pytest.approx(risk_profile.volume_multiplier)
    assert metrics.get("risk_drawdown_limit_pct") == pytest.approx(risk_profile.max_drawdown_pct * 100.0)
    assert metrics.get("mtf_filter_rejections") is not None
    artifact = result.artifacts[0]
    assert artifact.telemetry_file.exists()
    lines = [json.loads(line) for line in artifact.telemetry_file.read_text().splitlines() if line.strip()]
    assert any(event.get("event") == "run_context" for event in lines)
    equity_events = [event for event in lines if event.get("event") == "equity_curve_point"]
    assert equity_events, "Expected equity_curve_point events in telemetry"
    assert all(event.get("risk_preset") == "balanced" for event in equity_events)
    assert all(event.get("persona") == "neutral" for event in equity_events)
    trade_events = [event for event in lines if event.get("event") == "trade_closed"]
    assert trade_events, "Expected trade_closed events in telemetry"
    assert all(event.get("risk_preset") == "balanced" for event in trade_events)
    assert all(event.get("persona") == "neutral" for event in trade_events)
    assert all("mtf_bias" in event for event in trade_events)
    mtf_summary = [event for event in lines if event.get("event") == "mtf_filter_summary"]
    assert mtf_summary, "Expected mtf_filter_summary telemetry event"


def test_crypto_symbol_scaling_limits_extreme_pnl(tmp_path: Path) -> None:
    frame = _make_crypto_dataset()
    start = frame["time"].iloc[0].to_pydatetime()
    end = frame["time"].iloc[-1].to_pydatetime()
    config = ExperimentConfig(
        name="crypto_scaling",
        data=DataCacheConfig(symbols=("BTCUSD",), timeframe="H1", use_cache=False),
        time_range=ExperimentTimeRange(start=start, end=end),
        telemetry=TelemetryOutputConfig(directory=tmp_path),
        initial_balance=25_000.0,
    )
    cache = StubCache({"BTCUSD": frame})
    runner = ExperimentRunner(config, cache)  # type: ignore[arg-type]
    sample = ParameterSample(
        values={
            "base_volume": 0.12,
            "pip_value": 450.0,
            "commission_per_lot": -0.7,
            "fast_window": 6,
            "slow_window": 18,
            "atr_window": 14,
            "max_holding_bars": 18,
            "take_profit_atr": 1.4,
            "stop_loss_atr": 1.2,
            "spread_bps": 5.0,
            "slippage_bps": 2.0,
            "risk_preset": "balanced",
        }
    )

    result = runner.run(sample)

    metrics = result.scorecard.metrics
    assert metrics.get("trade_count", 0.0) > 0
    artifact = result.artifacts[0]
    trades = []
    for line in artifact.telemetry_file.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "trade_closed":
            trades.append(record)
    assert trades, "Expected crypto trades to be produced"
    max_abs_net = max(abs(trade.get("net_result", 0.0)) for trade in trades)
    assert max_abs_net < 2_500.0, f"Unexpectedly high crypto net result: {max_abs_net}"
    max_price_points = max(abs(trade.get("price_diff_points", 0.0)) for trade in trades)
    assert max_price_points < 25_000.0, f"Crypto price diff points not scaled: {max_price_points}"


def test_parameter_space_names_include_simulator_params() -> None:
    config = ExperimentConfig(
        name="with_personas",
        data=DataCacheConfig(symbols=("EURUSD+",), timeframe="H1"),
        personas=("alpha", "beta"),
        risk_presets=("balanced", "aggressive"),
    )
    space = build_parameter_space(config)
    expected = {
        "fast_window",
        "slow_window",
        "atr_window",
        "take_profit_atr",
        "stop_loss_atr",
        "max_holding_bars",
        "spread_bps",
        "slippage_bps",
        "breakeven_atr_trigger",
        "trailing_start_atr",
        "trailing_distance_atr",
        "drawdown_limit_pct",
        "mtf_confirmation",
        "mtf_short_window",
        "mtf_long_window",
    }
    names = set(space.names)
    assert expected.issubset(names)
    assert {"risk_preset", "persona"}.issubset(names)


def test_risk_and_persona_resolvers_handle_aliases() -> None:
    assert resolve_risk_profile("Moderate").name == "balanced"
    assert resolve_risk_profile("MAX").name == "ultra"
    assert resolve_persona_profile("intraday").name == "scalper"
    assert resolve_persona_profile("unknown").name == "unknown"


def test_drawdown_override_in_metrics(tmp_path: Path) -> None:
    frame = _make_dataset()
    start = frame["time"].iloc[0].to_pydatetime()
    end = frame["time"].iloc[-1].to_pydatetime()
    config = ExperimentConfig(
        name="override_test",
        data=DataCacheConfig(symbols=("TEST",), timeframe="H1", use_cache=False),
        time_range=ExperimentTimeRange(start=start, end=end),
        telemetry=TelemetryOutputConfig(directory=tmp_path),
        initial_balance=10_000.0,
    )
    cache = StubCache({"TEST": frame})
    runner = ExperimentRunner(config, cache)  # type: ignore[arg-type]
    sample = ParameterSample(
        values={
            "base_volume": 0.075,
            "pip_value": 260.0,
            "commission_per_lot": -0.5,
            "fast_window": 6,
            "slow_window": 20,
            "atr_window": 14,
            "max_holding_bars": 16,
            "take_profit_atr": 1.4,
            "stop_loss_atr": 1.1,
            "spread_bps": 2.0,
            "slippage_bps": 1.0,
            "risk_preset": "balanced",
            "drawdown_limit_pct": 12.0,
        }
    )

    result = runner.run(sample)

    metrics = result.scorecard.metrics
    assert metrics.get("risk_drawdown_limit_pct") == pytest.approx(12.0)
    assert metrics.get("drawdown_override_pct") == pytest.approx(12.0)


def test_mtf_confirmation_blocks_conflicts(monkeypatch, tmp_path: Path) -> None:
    periods = 160
    times = pd.date_range("2024-02-01T00:00:00Z", periods=periods, freq="15min", tz="UTC")
    first_half = np.linspace(60.0, 50.0, periods // 2)
    second_half = np.linspace(50.0, 62.0, periods - periods // 2)
    base_close = np.concatenate([first_half, second_half])
    frame = pd.DataFrame({
        "time": times,
        "close": base_close,
        "high": base_close + 0.05,
        "low": base_close - 0.05,
    })
    start = times[0].to_pydatetime()
    end = times[-1].to_pydatetime()
    config = ExperimentConfig(
        name="mtf_test",
        data=DataCacheConfig(symbols=("TEST",), timeframe="M15", use_cache=False),
        time_range=ExperimentTimeRange(start=start, end=end),
        telemetry=TelemetryOutputConfig(directory=tmp_path),
        initial_balance=5_000.0,
    )
    cache = StubCache({"TEST": frame})

    def fake_lookup(self, frame, short_window, long_window, freq="1H"):
        return lambda ts: "bearish"

    monkeypatch.setattr(ExperimentRunner, "_build_mtf_bias_lookup", fake_lookup)

    baseline_sample = ParameterSample(
        values={
            "base_volume": 0.07,
            "pip_value": 220.0,
            "commission_per_lot": -0.4,
            "fast_window": 4,
            "slow_window": 12,
            "atr_window": 10,
            "max_holding_bars": 24,
            "take_profit_atr": 1.5,
            "stop_loss_atr": 1.0,
            "spread_bps": 2.0,
            "slippage_bps": 1.0,
            "risk_preset": "balanced",
            "mtf_confirmation": "off",
        }
    )
    runner_off = ExperimentRunner(config, cache)  # type: ignore[arg-type]
    baseline_result = runner_off.run(baseline_sample)
    assert baseline_result.scorecard.metrics.get("trade_count", 0.0) > 0

    mtf_sample = ParameterSample(
        values={
            **baseline_sample.values,
            "mtf_confirmation": "on",
            "mtf_short_window": 3,
            "mtf_long_window": 8,
        }
    )
    runner_on = ExperimentRunner(config, cache)  # type: ignore[arg-type]
    mtf_result = runner_on.run(mtf_sample)
    assert mtf_result.scorecard.metrics.get("trade_count", 0.0) == 0.0
    assert mtf_result.scorecard.metrics.get("mtf_filter_rejections", 0.0) > 0.0
