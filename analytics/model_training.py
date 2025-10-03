"""Model training utilities for trade outcome classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import TelemetryConfig
from .feature_engineering import TradeFeatures, build_trade_features, load_trade_features


@dataclass(slots=True)
class TrainingResult:
    metrics: dict[str, float]
    model_path: Path
    feature_columns: list[str]
    synthetic: bool = False
    trained_at: datetime | None = None
    training_rows: int = 0
    sklearn_version: str | None = None


def _split_data(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    sample_weights: np.ndarray | pd.Series | None,
    *,
    test_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.Series | np.ndarray,
    pd.Series | np.ndarray,
    np.ndarray | pd.Series | None,
    np.ndarray | pd.Series | None,
]:
    if len(labels) < 5 or len(np.unique(labels)) < 2:
        # Not enough samples for a stratified split; fall back to training on full dataset.
        return features, features, labels, labels, sample_weights, sample_weights

    if sample_weights is None:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        return X_train, X_test, y_train, y_test, None, None

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features,
        labels,
        sample_weights,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test, w_train, w_test


def _derive_context_weights(metadata: pd.DataFrame) -> tuple[np.ndarray | None, dict[str, float]]:
    if "context_risk_preset" not in metadata:
        return None, {}
    context = metadata["context_risk_preset"].fillna("unknown")
    counts = context.value_counts(dropna=False)
    if counts.empty:
        return None, {}

    total = float(len(context))
    groups = float(len(counts))
    if total <= 0 or groups <= 0:
        return None, {}

    max_weight = 3.0
    weight_map: dict[str, float] = {}
    for key, count in counts.items():
        adjusted = max(float(count), 1.0)
        weight = min(max_weight, total / (groups * adjusted))
        weight_map[str(key)] = float(weight)

    weights = context.map(lambda value: weight_map.get(str(value), 1.0)).to_numpy(dtype=float, copy=False)
    return weights, weight_map


def _compute_context_metrics(metadata: pd.DataFrame, features: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    if metadata.empty:
        return {}

    metrics: dict[str, dict[str, dict[str, float]]] = {}

    extended = metadata.copy()
    extended["net_result_per_lot"] = features.get("net_result_per_lot", pd.Series(np.nan, index=metadata.index))

    for column, bucket in (("context_risk_preset", "risk_presets"), ("context_persona", "personas")):
        if column not in extended:
            continue
        grouped: dict[str, dict[str, float]] = {}
        for key, group in extended.groupby(column):
            rows = int(len(group))
            if rows == 0:
                continue
            wins = int((group["outcome"] == "win").sum())
            avg_net = float(pd.to_numeric(group.get("net_result"), errors="coerce").mean() or 0.0)
            avg_per_lot = float(pd.to_numeric(group.get("net_result_per_lot"), errors="coerce").mean() or 0.0)
            grouped[str(key)] = {
                "rows": rows,
                "win_rate": float(wins / rows) if rows else 0.0,
                "avg_net_result": avg_net,
                "avg_net_per_lot": avg_per_lot,
            }
        metrics[bucket] = grouped

    return metrics


def train_trade_classifier(
    config: Optional[TelemetryConfig] = None,
    *,
    output_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    synthetic = False
    try:
        bundle: TradeFeatures = load_trade_features(config)
    except (FileNotFoundError, ValueError):
        bundle = _build_synthetic_trade_features()
        synthetic = True
    feature_frame = bundle.features.astype(float, copy=False)
    labels = bundle.labels.to_numpy(copy=False)

    if len(np.unique(labels)) < 2:
        raise ValueError("Classifier training requires both winning and losing samples")

    context_weights, weight_map = _derive_context_weights(bundle.metadata)
    X_train, X_test, y_train, y_test, w_train, w_test = _split_data(
        feature_frame,
        labels,
        context_weights,
        test_size=test_size,
        random_state=random_state,
    )

    if w_train is not None:
        w_train = np.asarray(w_train, dtype=float)
    if w_test is not None:
        w_test = np.asarray(w_test, dtype=float)

    model = LogisticRegression(max_iter=500, class_weight="balanced")
    if w_train is not None:
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        y_proba = None

    metric_kwargs = {"sample_weight": w_test} if w_test is not None else {}
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred, **metric_kwargs)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0, **metric_kwargs)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0, **metric_kwargs)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0, **metric_kwargs)),
    }
    if y_proba is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba, **metric_kwargs))
    else:
        metrics["roc_auc"] = float("nan")

    feature_columns = bundle.features.columns.tolist()
    enriched_feature_frame = bundle.features.copy()
    enriched_feature_frame["symbol"] = bundle.metadata["symbol"].fillna("UNKNOWN")
    enriched_feature_frame["direction"] = bundle.metadata["direction"].fillna("unknown")
    cost_profiles: dict[str, dict[str, float]] = {}
    if not enriched_feature_frame.empty:
        for (symbol, direction), group in enriched_feature_frame.groupby(["symbol", "direction"]):
            key = f"{symbol}::{direction}"
            cost_profiles[key] = {
                "swap_per_lot": float(group["swap_per_lot"].mean()),
                "commission_per_lot": float(group["commission_per_lot"].mean()),
            }

    global_cost_profile = {
        "swap_per_lot": float(enriched_feature_frame["swap_per_lot"].mean()) if not enriched_feature_frame.empty else 0.0,
        "commission_per_lot": float(enriched_feature_frame["commission_per_lot"].mean()) if not enriched_feature_frame.empty else 0.0,
    }

    trained_at = datetime.now(timezone.utc)

    sklearn_version_str = str(sklearn_version)

    context_metrics = _compute_context_metrics(bundle.metadata, bundle.features)
    risk_presets = (
        sorted(bundle.metadata.get("context_risk_preset", pd.Series(dtype="object")).dropna().unique().tolist())
        if "context_risk_preset" in bundle.metadata
        else []
    )
    personas = (
        sorted(bundle.metadata.get("context_persona", pd.Series(dtype="object")).dropna().unique().tolist())
        if "context_persona" in bundle.metadata
        else []
    )

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "symbols": sorted(bundle.metadata["symbol"].dropna().unique().tolist()),
        "strategies": sorted(bundle.metadata["strategy_key"].dropna().unique().tolist()),
        "metrics": metrics,
        "cost_profiles": cost_profiles,
        "global_cost_profile": global_cost_profile,
        "synthetic": synthetic,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "training_rows": int(len(bundle.features)),
        "sklearn_version": sklearn_version_str,
        "risk_presets": risk_presets,
        "personas": personas,
        "context_weight_map": weight_map,
        "context_metrics": context_metrics,
    }

    model_output_path = output_path or Path("analytics_data/models/trade_classifier.joblib")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output_path)

    return TrainingResult(
        metrics=metrics,
        model_path=model_output_path,
        feature_columns=feature_columns,
        synthetic=synthetic,
        trained_at=trained_at,
        training_rows=int(len(bundle.features)),
        sklearn_version=sklearn_version_str,
    )

def _build_synthetic_trade_features(now: datetime | None = None) -> TradeFeatures:
    now = now or datetime.now(timezone.utc)
    rows: list[dict[str, object]] = []
    base = now - timedelta(days=30)
    scenarios = [
        {"outcome": "win", "direction": "buy", "symbol": "EURUSD+", "strategy_key": "momentum_trend"},
        {"outcome": "loss", "direction": "sell", "symbol": "EURUSD+", "strategy_key": "momentum_trend"},
        {"outcome": "win", "direction": "buy", "symbol": "GBPUSD+", "strategy_key": "mean_reversion"},
        {"outcome": "loss", "direction": "sell", "symbol": "GBPUSD+", "strategy_key": "mean_reversion"},
    ]

    for idx, scenario in enumerate(scenarios, start=1):
        outcome = scenario["outcome"]
        direction = scenario["direction"]
        symbol = scenario["symbol"]
        strategy_key = scenario["strategy_key"]
        entry_time = base + timedelta(hours=idx * 6)
        exit_time = entry_time + timedelta(hours=4)
        price_open = 1.1000 + 0.0007 * idx
        reward = 0.0012
        risk = 0.0008

        if direction == "buy":
            take_profit = price_open + reward
            stop_loss = price_open - risk
            price_close = take_profit if outcome == "win" else price_open - risk
        else:
            take_profit = price_open - reward
            stop_loss = price_open + risk
            price_close = take_profit if outcome == "win" else price_open + risk

        net_result = 45.0 if outcome == "win" else -35.0

        rows.append(
            {
                "ticket": 1000 + idx,
                "position_id": 1000 + idx,
                "symbol": symbol,
                "strategy_key": strategy_key,
                "strategy_code": strategy_key,
                "direction": direction,
                "volume": 0.10 + idx * 0.02,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "duration_seconds": int((exit_time - entry_time).total_seconds()),
                "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0,
                "price_open": price_open,
                "price_close": price_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "profit": net_result,
                "swap": 0.0,
                "commission": -1.5,
                "net_result": net_result,
                "outcome": outcome,
                "rr_ratio": round(reward / risk, 4),
                "price_diff_points": (price_close - price_open) / 0.0001,
                "comment": f"synthetic-{strategy_key}",
            }
        )

    df = pd.DataFrame(rows)
    return build_trade_features(df, include_symbols=True, include_strategies=True)
