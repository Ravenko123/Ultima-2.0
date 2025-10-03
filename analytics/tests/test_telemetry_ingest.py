"""Unit tests for telemetry ingestion."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from analytics.telemetry_ingest import ingest_telemetry
from analytics.config import TelemetryConfig


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_ingest_guard_snapshot(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            {
                "event": "guard_snapshot",
                "timestamp": "2025-10-03T12:34:56Z",
                "payload": {
                    "scan": 101,
                    "guard_factor": 0.82,
                    "soft_factor": 0.91,
                    "margin_factor": 0.88,
                    "equity_factor": 0.97,
                    "var_factor": 0.75,
                    "combined": 0.55,
                    "risk_allowed": True,
                    "risk_status": "throttle",
                    "soft_status": "caution",
                    "margin_status": "clear",
                    "var_status": "clamp",
                    "combined_bucket": "compressed",
                    "daily_drawdown": 0.04,
                    "weekly_drawdown": 0.06,
                    "guard_pressure": 0.45,
                    "guard_relief": 0.12,
                },
            }
        ],
    )
    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    datasets = ingest_telemetry(config)
    assert "guard_snapshots" in datasets
    df = datasets["guard_snapshots"]
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "scan"] == 101
    assert df.loc[0, "guard_factor"] == 0.82
    assert df.loc[0, "risk_status"] == "throttle"


def test_ingest_micro_override(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            {
                "event": "micro_override",
                "payload": {
                    "scan": 42,
                    "symbol": "EURUSD+",
                    "signal": "buy",
                    "regime": "TRENDING",
                    "kind": "guard_near_miss",
                    "guard_factor": 1.05,
                    "drawdown_atr": 0.4,
                    "momentum": 0.015,
                    "threshold": 0.018,
                    "near_miss_ratio": 0.88,
                    "guard_scale": 0.66,
                },
            }
        ],
    )
    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    datasets = ingest_telemetry(config)
    assert "micro_overrides" in datasets
    df = datasets["micro_overrides"]
    assert df.loc[0, "symbol"] == "EURUSD+"
    assert df.loc[0, "extra_guard_scale"] == 0.66


def test_ingest_trade_closed(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            {
                "event": "trade_closed",
                "payload": {
                    "ticket": 321,
                    "position_id": 321,
                    "magic": 9001,
                    "symbol": "EURUSD",
                    "direction": "buy",
                    "volume": 0.25,
                    "entry_time": "2025-10-03T10:00:00Z",
                    "exit_time": "2025-10-03T10:30:00Z",
                    "duration_seconds": 1800,
                    "price_open": 1.1000,
                    "price_close": 1.1020,
                    "price_diff_points": 20.0,
                    "stop_loss": 1.0980,
                    "take_profit": 1.1050,
                    "profit": 25.0,
                    "net_result": 22.5,
                    "comment": "MTB01",
                    "strategy_key": "momentum_trend",
                    "strategy_code": "MT",
                },
            }
        ],
    )
    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    datasets = ingest_telemetry(config)
    assert "trade_closed" in datasets
    df = datasets["trade_closed"]
    assert df.loc[0, "ticket"] == 321
    assert df.loc[0, "outcome"] == "win"
    assert df.loc[0, "holding_minutes"] == 30.0
    assert df.loc[0, "rr_ratio"] == 1.0
    assert df.loc[0, "swap"] is None or df.loc[0, "swap"] == 0.0
    assert df.loc[0, "commission"] is None or df.loc[0, "commission"] == 0.0


def test_ingest_risk_preset_applied(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            {
                "ts": "2025-10-03T18:15:01Z",
                "event": "risk_preset_applied",
                "preset": "high",
                "changed": True,
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
        ],
    )

    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    datasets = ingest_telemetry(config)

    assert "risk_preset_applied" in datasets
    df = datasets["risk_preset_applied"]
    assert df.loc[0, "preset"] == "high"
    assert df.loc[0, "persona"] == "fire"
    assert df.loc[0, "risk_multiplier_max"] == 7.8
    assert bool(df.loc[0, "dynamic_var_enabled"]) is True
    assert bool(df.loc[0, "equity_governor_enabled"]) is False
