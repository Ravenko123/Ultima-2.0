"""Tests for feature engineering utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analytics.feature_engineering import TradeFeatures, build_trade_features, load_trade_features
from analytics.config import TelemetryConfig


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_build_trade_features_basic() -> None:
    trade_df = pd.DataFrame(
        [
            {
                "ticket": 1,
                "position_id": 1,
                "symbol": "EURUSD",
                "strategy_key": "momentum_trend",
                "direction": "buy",
                "volume": 0.5,
                "price_open": 1.1000,
                "stop_loss": 1.0980,
                "take_profit": 1.1040,
                "entry_time": "2025-10-03T08:30:00Z",
                "exit_time": "2025-10-03T09:30:00Z",
                "price_diff_points": 12.0,
                "holding_minutes": 20.0,
                "rr_ratio": 1.5,
                "net_result": 45.0,
                "profit": 48.0,
                "swap": -1.2,
                "commission": -1.8,
                "outcome": "win",
            }
        ]
    )

    guard_df = pd.DataFrame(
        [
            {
                "timestamp": "2025-10-03T08:00:00Z",
                "scan": 11,
                "combined": 0.78,
                "guard_factor": 0.82,
                "soft_factor": 0.80,
                "margin_factor": 0.84,
                "equity_factor": 0.76,
                "var_factor": 0.90,
                "risk_allowed": True,
                "risk_status": "clear",
                "combined_bucket": "soft",
            }
        ]
    )
    risk_df = pd.DataFrame(
        [
            {
                "timestamp": "2025-10-03T08:05:00Z",
                "preset": "high",
                "persona": "fire",
                "persona_label": "Firestorm Aggressor",
                "account_risk": 0.2,
                "risk_multiplier_max": 7.8,
                "risk_multiplier_relief": 6.2,
                "scan_interval": 18,
                "min_session_priority": 1,
                "alpha_session_priority": 2,
                "soft_guard_limit": 0.74,
                "margin_usage_block": 0.999,
                "low_vol_skip": 0.22,
                "low_vol_scale": 1.2,
                "micro_guard_min": 0.72,
                "high_micro_guard_min": 0.74,
                "dynamic_var_enabled": True,
                "equity_governor_enabled": False,
            }
        ]
    )

    bundle = build_trade_features(
        trade_df,
        guard_snapshots=guard_df,
        risk_presets=risk_df,
        position_snapshots=pd.DataFrame(
            [
                {
                    "timestamp": "2025-10-03T08:20:00Z",
                    "scan": 9,
                    "total_positions": 2,
                    "unique_symbols": 1,
                    "total_volume": 1.0,
                    "buy_volume": 0.6,
                    "sell_volume": 0.4,
                    "net_volume": 0.2,
                    "floating_profit": 12.0,
                    "floating_swap": -0.3,
                    "floating_commission": -0.4,
                    "balance": 10050.0,
                    "equity": 10062.0,
                    "margin_used": 1200.0,
                    "margin_free": 8850.0,
                    "margin_level": 845.0,
                    "margin_usage_ratio": 0.12,
                    "symbol_concentration": 0.9,
                    "top_symbol": "EURUSD",
                    "top_symbol_volume": 0.6,
                    "top_symbol_net_volume": 0.2,
                    "top_symbol_profit": 11.0,
                    "currency_exposure_total": 250000.0,
                    "currency_exposure_ratio": 2.5,
                    "top_currency": "USD",
                    "top_currency_notional": 150000.0,
                    "symbol_1": "EURUSD",
                    "symbol_1_volume": 0.6,
                    "symbol_1_net_volume": 0.2,
                    "symbol_1_profit": 11.0,
                }
            ]
        ),
    )

    assert isinstance(bundle, TradeFeatures)
    assert bundle.features.shape[0] == 1
    # net_per_lot = 45 / 0.5 = 90
    # risk_price = 1.1000 - 1.0980 = 0.002
    assert bundle.features.iloc[0]["risk_price"] == pytest.approx(0.002)
    assert bundle.features.iloc[0]["commission_per_lot"] == pytest.approx(-3.6)
    assert bundle.features.iloc[0]["expected_cost_per_lot"] == pytest.approx(-6.0)
    assert bundle.features.iloc[0]["direction_encoded"] == 1
    assert bundle.features.iloc[0]["net_result_per_lot"] == pytest.approx(90.0)
    assert bundle.features.iloc[0]["profit_per_minute"] == pytest.approx(2.25)
    assert bundle.features.iloc[0]["guard_combined_at_entry"] == pytest.approx(0.78)
    assert bundle.features.iloc[0]["risk_account_risk"] == pytest.approx(0.2)
    assert bundle.features.iloc[0]["positions_total_at_entry"] == pytest.approx(2.0)
    assert bundle.features.iloc[0]["positions_top_symbol_volume_at_entry"] == pytest.approx(0.6)
    top_symbol_columns = [col for col in bundle.features.columns if col.startswith("positions_top_symbol_")]
    assert any(bundle.features.iloc[0][col] == 1 for col in top_symbol_columns)
    assert bundle.metadata.loc[0, "context_risk_preset"] == "high"
    assert bundle.metadata.loc[0, "context_persona"] == "fire"
    assert bundle.metadata.loc[0, "positions_top_symbol_at_entry"] == "EURUSD"
    assert bundle.labels.iloc[0] == 1
    assert bundle.metadata.loc[0, "symbol"] == "EURUSD"


def test_load_trade_features_from_telemetry(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            {
                "event": "position_snapshot",
                "timestamp": "2025-10-03T08:55:00Z",
                "scan": 7,
                "total_positions": 1,
                "unique_symbols": 1,
                "total_volume": 0.2,
                "buy_volume": 0.0,
                "sell_volume": 0.2,
                "net_volume": -0.2,
                "floating_profit": 18.0,
                "floating_swap": -0.2,
                "floating_commission": -0.3,
                "balance": 10020.0,
                "equity": 10035.0,
                "margin_used": 900.0,
                "margin_free": 9100.0,
                "margin_level": 1115.0,
                "margin_usage_ratio": 0.09,
                "symbol_concentration": 1.0,
                "top_symbol": "USDJPY",
                "top_symbol_volume": 0.2,
                "top_symbol_net_volume": -0.2,
                "top_symbol_profit": 18.0,
                "currency_exposure_total": 150000.0,
                "currency_exposure_ratio": 1.5,
                "top_currency": "JPY",
                "top_currency_notional": 150000.0,
            },
            {
                "event": "trade_closed",
                "payload": {
                    "ticket": 11,
                    "position_id": 11,
                    "magic": 100,
                    "symbol": "USDJPY",
                    "direction": "sell",
                    "volume": 0.2,
                    "entry_time": "2025-10-03T09:00:00Z",
                    "exit_time": "2025-10-03T09:45:00Z",
                    "duration_seconds": 2700,
                    "price_open": 110.100,
                    "price_close": 109.950,
                    "price_diff_points": -15.0,
                    "stop_loss": 110.200,
                    "take_profit": 109.800,
                    "profit": 30.0,
                    "swap": -0.5,
                    "commission": -0.9,
                    "net_result": 28.6,
                    "comment": "MTB01",
                    "strategy_key": "momentum_trend",
                    "strategy_code": "MT",
                },
            }
        ],
    )
    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    bundle = load_trade_features(config)
    assert bundle.features.shape[0] == 1
    assert bundle.labels.iloc[0] == 1
    # Sell direction should encode as -1
    assert bundle.features.iloc[0]["direction_encoded"] == -1
    # Commission per lot = -0.9 / 0.2 = -4.5
    assert bundle.features.iloc[0]["commission_per_lot"] == pytest.approx(-4.5)
    # Context defaults to unknown if risk data absent
    assert bundle.metadata.loc[0, "context_risk_preset"] == "unknown"
    assert bundle.features.iloc[0]["positions_total_at_entry"] == pytest.approx(1.0)
