"""Automated trade classifier retraining helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
import shutil

import pandas as pd

from .config import TelemetryConfig
from .feature_engineering import build_trade_features
from .model_training import TrainingResult, train_trade_classifier
from .telemetry_ingest import ingest_telemetry


@dataclass(slots=True)
class AutoTrainThresholds:
    """Thresholds that must be satisfied before retraining proceeds."""

    min_rows: int = 50
    min_class_ratio: float = 0.1
    max_age_hours: int | None = 24
    require_recent_rows: int = 10


@dataclass(slots=True)
class AutoTrainDecision:
    should_train: bool
    reason: str
    stats: dict[str, object]


@dataclass(slots=True)
class AutoTrainResult:
    decision: AutoTrainDecision
    training: TrainingResult | None = None
    archived_model: Path | None = None


def _normalize_config(config: TelemetryConfig | None, telemetry_path: Path) -> TelemetryConfig:
    base = config or TelemetryConfig.from_env()
    if base.telemetry_path == telemetry_path:
        return base
    return TelemetryConfig(
        telemetry_path=telemetry_path,
        output_path=base.output_path,
        output_format=base.output_format,
    )


def _coerce_datetime(series: Iterable[object]) -> pd.Series:
    return pd.to_datetime(pd.Series(list(series)), utc=True, errors="coerce")


def evaluate_training_readiness(
    telemetry_path: Path,
    *,
    thresholds: AutoTrainThresholds,
    now: datetime | None = None,
    config: TelemetryConfig | None = None,
) -> AutoTrainDecision:
    stats: dict[str, object] = {
        "rows": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "recent_rows": 0,
    }
    if not telemetry_path.exists():
        return AutoTrainDecision(False, f"Telemetry file missing: {telemetry_path}", stats)

    telemetry_config = _normalize_config(config, telemetry_path)
    try:
        datasets = ingest_telemetry(telemetry_config)
    except FileNotFoundError:
        return AutoTrainDecision(False, f"Telemetry file missing: {telemetry_path}", stats)

    trades = datasets.get("trade_closed")
    if trades is None or trades.empty:
        return AutoTrainDecision(False, "No trade_closed events available", stats)

    guard_snapshots = datasets.get("guard_snapshots")
    risk_presets = datasets.get("risk_preset_applied")
    try:
        feature_bundle = build_trade_features(
            trades,
            include_symbols=False,
            include_strategies=False,
            guard_snapshots=guard_snapshots,
            risk_presets=risk_presets,
        )
    except ValueError as exc:
        return AutoTrainDecision(False, f"Feature preparation failed: {exc}", stats)

    stats["rows"] = int(len(feature_bundle.features))
    if stats["rows"] < thresholds.min_rows:
        return AutoTrainDecision(False, f"Only {stats['rows']} rows (< {thresholds.min_rows})", stats)

    metadata = feature_bundle.metadata

    risk_counts = metadata.get("context_risk_preset").value_counts(dropna=False) if "context_risk_preset" in metadata else pd.Series(dtype=int)
    persona_counts = metadata.get("context_persona").value_counts(dropna=False) if "context_persona" in metadata else pd.Series(dtype=int)
    stats["risk_presets"] = {str(k): int(v) for k, v in risk_counts.items()}
    stats["personas"] = {str(k): int(v) for k, v in persona_counts.items()}

    outcome_counts = metadata["outcome"].value_counts(dropna=True)
    wins = int(outcome_counts.get("win", 0))
    losses = int(outcome_counts.get("loss", 0))
    flats = int(outcome_counts.get("flat", 0))
    stats.update({"wins": wins, "losses": losses, "flats": flats})

    if wins == 0 or losses == 0:
        return AutoTrainDecision(False, "Require both win and loss samples", stats)

    min_ratio = thresholds.min_class_ratio
    ratio_wins = wins / stats["rows"]
    ratio_losses = losses / stats["rows"]
    stats.update({"win_ratio": ratio_wins, "loss_ratio": ratio_losses})
    if ratio_wins < min_ratio or ratio_losses < min_ratio:
        return AutoTrainDecision(False, "Class distribution below threshold", stats)

    if thresholds.max_age_hours is not None:
        now = now or datetime.now(timezone.utc)
        exit_times = _coerce_datetime(trades.get("exit_time", []))
        entry_times = _coerce_datetime(trades.get("entry_time", []))
        timestamps = exit_times.fillna(entry_times)
        if timestamps.dropna().empty:
            return AutoTrainDecision(False, "Missing timestamps to evaluate freshness", stats)
        horizon = now - timedelta(hours=thresholds.max_age_hours)
        recent_mask = timestamps >= horizon
        stats["recent_rows"] = int(recent_mask.sum())
        if stats["recent_rows"] < thresholds.require_recent_rows:
            return AutoTrainDecision(False, "Not enough recent trades", stats)

    return AutoTrainDecision(True, "Thresholds satisfied", stats)


def run_auto_train(
    *,
    telemetry_path: Path,
    model_output: Path,
    thresholds: AutoTrainThresholds | None = None,
    config: TelemetryConfig | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    dry_run: bool = False,
    archive_dir: Path | None = None,
) -> AutoTrainResult:
    thresholds = thresholds or AutoTrainThresholds()
    decision = evaluate_training_readiness(
        telemetry_path,
        thresholds=thresholds,
        config=config,
    )

    if dry_run and decision.should_train:
        dry_stats = dict(decision.stats)
        dry_stats["dry_run"] = True
        decision = AutoTrainDecision(False, "Dry run: thresholds satisfied", dry_stats)

    if not decision.should_train:
        return AutoTrainResult(decision=decision)

    archived_model: Path | None = None
    if archive_dir is not None and model_output.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archived_model = archive_dir / f"{model_output.stem}_{timestamp}{model_output.suffix}"
        shutil.copy2(model_output, archived_model)

    telemetry_config = _normalize_config(config, telemetry_path)
    training_result = train_trade_classifier(
        telemetry_config,
        output_path=model_output,
        test_size=test_size,
        random_state=random_state,
    )

    return AutoTrainResult(
        decision=decision,
        training=training_result,
        archived_model=archived_model,
    )