"""Tests for automated classifier retraining helpers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib

from analytics.auto_train import (
    AutoTrainDecision,
    AutoTrainResult,
    AutoTrainThresholds,
    evaluate_training_readiness,
    run_auto_train,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _trade_payload(
    *,
    idx: int,
    direction: str,
    price_open: float,
    price_close: float,
    net_result: float,
    exit_time: datetime,
) -> dict[str, object]:
    stop_loss = price_open - 0.0025 if direction == "buy" else price_open + 0.0025
    take_profit = price_open + 0.0040 if direction == "buy" else price_open - 0.0040
    entry_time = (exit_time - timedelta(hours=1)).isoformat()
    return {
        "event": "trade_closed",
        "payload": {
            "ticket": idx + 1,
            "position_id": idx + 1,
            "symbol": "EURUSD" if idx % 2 == 0 else "USDJPY",
            "direction": direction,
            "price_open": price_open,
            "price_close": price_close,
            "net_result": net_result,
            "profit": net_result + 1.0,
            "swap": -0.4,
            "commission": -0.9,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "volume": 0.2,
            "entry_time": entry_time,
            "exit_time": exit_time.isoformat(),
            "duration_seconds": 3600,
            "price_diff_points": (price_close - price_open) * 10000,
            "comment": "AUTO",
            "strategy_key": "momentum_trend",
            "strategy_code": "MT",
        },
    }


def test_evaluate_training_requires_min_rows(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    telemetry_path = tmp_path / "telemetry.jsonl"
    records: list[dict[str, object]] = []
    for i in range(4):
        direction = "buy" if i % 2 == 0 else "sell"
        price_open = 1.1000 + 0.0002 * i
        price_close = price_open + (0.0015 if direction == "buy" else -0.0012)
        net_result = 22.0 if direction == "buy" else -18.0
        exit_time = now - timedelta(minutes=5 * i)
        records.append(
            _trade_payload(
                idx=i,
                direction=direction,
                price_open=price_open,
                price_close=price_close,
                net_result=net_result,
                exit_time=exit_time,
            )
        )
    _write_jsonl(telemetry_path, records)

    thresholds = AutoTrainThresholds(min_rows=5, require_recent_rows=1)
    decision = evaluate_training_readiness(telemetry_path, thresholds=thresholds)

    assert isinstance(decision, AutoTrainDecision)
    assert not decision.should_train
    assert "Only" in decision.reason
    assert decision.stats["rows"] == 4


def test_run_auto_train_dry_run(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    telemetry_path = tmp_path / "telemetry.jsonl"
    records: list[dict[str, object]] = []
    for i in range(8):
        direction = "buy" if i % 2 == 0 else "sell"
        price_open = 1.1000 + 0.0005 * i
        price_close = price_open + (0.0030 if i % 3 != 0 else -0.0025)
        net_result = 30.0 if price_close > price_open else -25.0
        exit_time = now - timedelta(minutes=3 * i)
        records.append(
            _trade_payload(
                idx=i,
                direction=direction,
                price_open=price_open,
                price_close=price_close,
                net_result=net_result,
                exit_time=exit_time,
            )
        )
    _write_jsonl(telemetry_path, records)

    thresholds = AutoTrainThresholds(min_rows=6, max_age_hours=None, require_recent_rows=0)
    result = run_auto_train(
        telemetry_path=telemetry_path,
        model_output=tmp_path / "model.joblib",
        thresholds=thresholds,
        dry_run=True,
    )

    assert isinstance(result, AutoTrainResult)
    assert not result.decision.should_train
    assert "Dry run" in result.decision.reason
    assert result.training is None
    assert result.decision.stats["dry_run"]


def test_run_auto_train_with_archiving(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    telemetry_path = tmp_path / "telemetry.jsonl"
    records: list[dict[str, object]] = []
    for i in range(12):
        direction = "buy" if i % 2 == 0 else "sell"
        price_open = 1.0990 + 0.0004 * i
        price_close = price_open + (0.0035 if i % 4 != 0 else -0.0030)
        net_result = 35.0 if price_close > price_open else -30.0
        exit_time = now - timedelta(minutes=2 * i)
        records.append(
            _trade_payload(
                idx=i,
                direction=direction,
                price_open=price_open,
                price_close=price_close,
                net_result=net_result,
                exit_time=exit_time,
            )
        )
    _write_jsonl(telemetry_path, records)

    model_output = tmp_path / "model.joblib"
    archive_dir = tmp_path / "archive"
    model_output.write_text("placeholder", encoding="utf-8")

    thresholds = AutoTrainThresholds(min_rows=10, max_age_hours=None, require_recent_rows=0)
    result = run_auto_train(
        telemetry_path=telemetry_path,
        model_output=model_output,
        thresholds=thresholds,
        random_state=13,
        test_size=0.25,
        archive_dir=archive_dir,
    )

    assert result.decision.should_train
    assert result.training is not None
    assert result.training.model_path == model_output
    assert result.training.training_rows == len(records)
    assert not result.training.synthetic
    assert archive_dir.exists()
    assert result.archived_model is not None
    assert result.archived_model.exists()
    artifact = joblib.load(result.training.model_path)
    assert artifact["training_rows"] == len(records)