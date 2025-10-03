"""Regression tests for inference helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np

from analytics.config import TelemetryConfig
from analytics.feature_engineering import NUMERIC_FEATURES
from analytics.inference import load_trade_outcome_model
from analytics.model_training import sklearn_version, train_trade_classifier
from sklearn.linear_model import LogisticRegression


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _sample_trade(direction: str, price_open: float, price_close: float, net_result: float, *, idx: int) -> dict[str, object]:
    stop_loss = price_open - 0.0025 if direction == "buy" else price_open + 0.0025
    take_profit = price_open + 0.0040 if direction == "buy" else price_open - 0.0040
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
            "entry_time": "2025-10-02T08:00:00Z",
            "exit_time": "2025-10-02T09:00:00Z",
            "duration_seconds": 3600,
            "price_diff_points": (price_close - price_open) * 10000,
            "comment": "MTB01",
            "strategy_key": "momentum_trend",
            "strategy_code": "MT",
        },
    }


def test_predict_probability_preserves_feature_names(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    records = [
        _sample_trade("buy", 1.1000, 1.1040, 40.0, idx=0),
        _sample_trade("buy", 1.1010, 1.0990, -18.0, idx=1),
        _sample_trade("sell", 1.1050, 1.1015, 32.0, idx=2),
        _sample_trade("sell", 1.1045, 1.1060, -25.0, idx=3),
        _sample_trade("buy", 1.0995, 1.1035, 30.0, idx=4),
        _sample_trade("sell", 1.0970, 1.0935, 31.0, idx=5),
    ]
    _write_jsonl(telemetry_path, records)

    config = TelemetryConfig(telemetry_path=telemetry_path, output_path=tmp_path)
    model_path = tmp_path / "model.joblib"
    result = train_trade_classifier(config, output_path=model_path, test_size=0.3, random_state=11)

    artifact = joblib.load(result.model_path)
    assert artifact["trained_at"]
    assert artifact["sklearn_version"]
    assert artifact["training_rows"] == len(records)

    model = load_trade_outcome_model(model_path)
    feature_inputs = {name: 0.0 for name in NUMERIC_FEATURES}
    feature_inputs["volume"] = 0.2
    feature_inputs["direction_encoded"] = 1.0
    feature_inputs["rr_ratio"] = 1.6

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        probability = model.predict_probability(
            feature_inputs,
            symbol="EURUSD",
            strategy_key="momentum_trend",
            risk_preset="medium",
            persona="earth",
        )

    assert 0.0 <= probability <= 1.0
    assert not model.synthetic
    assert model.training_rows == len(records)
    assert model.sklearn_version == artifact["sklearn_version"]


def test_predict_probability_handles_models_without_feature_names(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy_model.joblib"

    X = np.array(
        [
            [0.2, 1.0, 1.0, 1.0],
            [0.1, -1.0, 0.0, 1.0],
            [0.3, 1.0, 1.0, 0.0],
            [0.4, -1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 1, 0], dtype=int)

    legacy_model = LogisticRegression(max_iter=200)
    legacy_model.fit(X, y)

    artifact = {
        "model": legacy_model,
        "feature_columns": [
            "volume",
            "direction_encoded",
            "symbol_EURUSD",
            "strategy_momentum_trend",
        ],
        "metrics": {},
        "cost_profiles": {},
        "global_cost_profile": {},
        "synthetic": False,
        "trained_at": "2025-10-03T00:00:00",
        "training_rows": int(len(X)),
    "sklearn_version": sklearn_version,
    }
    joblib.dump(artifact, model_path)

    model = load_trade_outcome_model(model_path)
    feature_inputs = {name: 0.0 for name in NUMERIC_FEATURES}
    feature_inputs["volume"] = 0.25
    feature_inputs["direction_encoded"] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        probability = model.predict_probability(
            feature_inputs,
            symbol="EURUSD",
            strategy_key="momentum_trend",
            risk_preset="medium",
            persona="earth",
        )

    assert 0.0 <= probability <= 1.0
    assert model.training_rows == len(X)