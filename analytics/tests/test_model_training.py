"""Tests for trade classifier training."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from analytics.model_training import train_trade_classifier
from analytics.config import TelemetryConfig


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_train_trade_classifier_creates_artifact(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    records: list[dict[str, object]] = []
    base_trade = {
        "magic": 100,
        "volume": 0.2,
        "entry_time": "2025-10-03T08:00:00Z",
        "exit_time": "2025-10-03T09:15:00Z",
        "duration_seconds": 4500,
        "comment": "MTB01",
        "strategy_key": "momentum_trend",
        "strategy_code": "MT",
    }
    # Create both winning and losing trades
    for idx, (direction, price_open, price_close, net_result) in enumerate(
        [
            ("buy", 1.1000, 1.1040, 40.0),
            ("buy", 1.1010, 1.0985, -25.0),
            ("sell", 1.1050, 1.1010, 35.0),
            ("sell", 1.1045, 1.1065, -20.0),
            ("buy", 1.1020, 1.1060, 30.0),
            ("sell", 1.0990, 1.0950, 28.0),
        ]
    ):
        stop_loss = price_open - 0.0025 if direction == "buy" else price_open + 0.0025
        take_profit = price_open + 0.0040 if direction == "buy" else price_open - 0.0040
        price_diff_points = (price_close - price_open) * 10000

        records.append(
            {
                "event": "trade_closed",
                "payload": {
                    "ticket": idx + 1,
                    "position_id": idx + 1,
                    "symbol": "EURUSD" if idx % 2 == 0 else "USDJPY",
                    "direction": direction,
                    "price_open": price_open,
                    "price_close": price_close,
                    "net_result": net_result,
                    "profit": net_result + 1.5,
                    "swap": -0.5,
                    "commission": -0.9,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "volume": base_trade["volume"],
                    "entry_time": base_trade["entry_time"],
                    "exit_time": base_trade["exit_time"],
                    "duration_seconds": base_trade["duration_seconds"],
                    "price_diff_points": price_diff_points,
                    "comment": base_trade["comment"],
                    "strategy_key": base_trade["strategy_key"],
                    "strategy_code": base_trade["strategy_code"],
                },
            }
        )
    _write_jsonl(telemetry_path, records)

    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    model_path = tmp_path / "model.joblib"
    result = train_trade_classifier(config, output_path=model_path, test_size=0.3, random_state=7)

    assert result.model_path.exists()
    artifact = joblib.load(result.model_path)
    assert "model" in artifact
    assert artifact["feature_columns"]
    assert "accuracy" in result.metrics
    assert artifact["trained_at"]
    assert artifact["training_rows"] == result.training_rows
    assert artifact["sklearn_version"].startswith("1.")
    assert result.trained_at is not None
    assert result.sklearn_version == artifact["sklearn_version"]
    assert "risk_presets" in artifact
    assert "context_weight_map" in artifact
    assert isinstance(artifact["context_metrics"], dict)
