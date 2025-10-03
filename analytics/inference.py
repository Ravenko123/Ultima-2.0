"""Inference helpers for trade outcome classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(slots=True)
class TradeOutcomeModel:
    path: Path
    artifact: dict[str, Any] | None = None
    feature_columns: list[str] | None = None
    synthetic: bool = False
    trained_at: str | None = None
    training_rows: int = 0
    sklearn_version: str | None = None

    def ensure_loaded(self) -> None:
        if self.artifact is None:
            artifact = joblib.load(self.path)
            if not isinstance(artifact, dict):
                raise ValueError(f"Unexpected artifact format at {self.path}")
            self.artifact = artifact
            self.feature_columns = list(artifact.get("feature_columns", []))
            if "model" not in artifact:
                raise ValueError("Artifact missing trained model")
            self.synthetic = bool(artifact.get("synthetic", False))
            self.trained_at = artifact.get("trained_at")
            self.training_rows = int(artifact.get("training_rows", 0) or 0)
            version = artifact.get("sklearn_version")
            self.sklearn_version = str(version) if version is not None else None

    @property
    def model(self):
        self.ensure_loaded()
        return self.artifact["model"]

    @property
    def cost_profiles(self) -> dict[str, dict[str, float]]:
        self.ensure_loaded()
        return dict(self.artifact.get("cost_profiles", {}))

    @property
    def global_cost_profile(self) -> dict[str, float]:
        self.ensure_loaded()
        return dict(self.artifact.get("global_cost_profile", {}))

    def _prepare_row(
        self,
        inputs: dict[str, float],
        symbol: str | None,
        strategy_key: str | None,
        risk_preset: str | None,
        persona: str | None,
        guard_risk_status: str | None,
        guard_combined_bucket: str | None,
    ) -> pd.DataFrame:
        self.ensure_loaded()
        columns = self.feature_columns or []
        row = {col: 0.0 for col in columns}

        for key, value in inputs.items():
            if key in row and np.isfinite(value):
                row[key] = float(value)

        def _assign_category(prefix: str, value: str | None) -> None:
            normalized = str(value).strip().lower() if value is not None else "unknown"
            column_key = f"{prefix}_{normalized}"
            if column_key in row:
                row[column_key] = 1.0
            else:
                fallback = f"{prefix}_unknown"
                if fallback in row:
                    row[fallback] = 1.0

        symbol_key = (symbol or "UNKNOWN")
        symbol_col = f"symbol_{symbol_key}"
        if symbol_col in row:
            row[symbol_col] = 1.0
        elif "symbol_UNKNOWN" in row:
            row["symbol_UNKNOWN"] = 1.0

        strategy_value = (strategy_key or "unknown")
        strategy_col = f"strategy_{strategy_value}"
        if strategy_col in row:
            row[strategy_col] = 1.0
        elif "strategy_unknown" in row:
            row["strategy_unknown"] = 1.0

        _assign_category("risk", risk_preset)
        _assign_category("persona", persona)
        _assign_category("guard_risk_status", guard_risk_status)
        _assign_category("guard_combined_bucket", guard_combined_bucket)

        df = pd.DataFrame([row], columns=columns)
        return df

    def predict_probability(
        self,
        features: dict[str, float],
        *,
        symbol: str | None = None,
        strategy_key: str | None = None,
        risk_preset: str | None = None,
        persona: str | None = None,
        guard_risk_status: str | None = None,
        guard_combined_bucket: str | None = None,
    ) -> float:
        dataframe = self._prepare_row(
            features,
            symbol,
            strategy_key,
            risk_preset,
            persona,
            guard_risk_status,
            guard_combined_bucket,
        )
        model = self.model

        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            input_data = dataframe
        else:
            input_data = dataframe.to_numpy(dtype=float, copy=False)

        probabilities = model.predict_proba(input_data)[:, 1]
        return float(probabilities[0])

    def get_cost_estimate(self, symbol: str | None, direction: str | None) -> dict[str, float]:
        key = f"{symbol or 'UNKNOWN'}::{direction or 'unknown'}"
        profile = self.cost_profiles.get(key)
        if profile:
            return profile
        return self.global_cost_profile


_model_cache: dict[Path, TradeOutcomeModel] = {}


def load_trade_outcome_model(path: Path) -> TradeOutcomeModel:
    resolved = path.resolve()
    cached = _model_cache.get(resolved)
    if cached is None:
        cached = TradeOutcomeModel(resolved)
        _model_cache[resolved] = cached
    cached.ensure_loaded()
    return cached
