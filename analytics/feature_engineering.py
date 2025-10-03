"""Feature engineering utilities for supervised models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import TelemetryConfig
from .telemetry_ingest import ingest_telemetry


NUMERIC_FEATURES = [
    "volume",
    "direction_encoded",
    "rr_ratio",
    "risk_price",
    "reward_price",
    "commission_per_lot",
    "swap_per_lot",
    "expected_cost_per_lot",
    "entry_hour",
    "entry_dayofweek",
    "net_result_per_lot",
    "profit_per_minute",
    "guard_combined_at_entry",
    "guard_equity_factor_at_entry",
    "guard_soft_factor_at_entry",
    "guard_margin_factor_at_entry",
    "guard_var_factor_at_entry",
    "guard_factor_at_entry",
    "guard_risk_allowed_at_entry",
    "guard_daily_drawdown_at_entry",
    "guard_weekly_drawdown_at_entry",
    "guard_pressure_at_entry",
    "guard_relief_at_entry",
    "risk_account_risk",
    "risk_equity_baseline",
    "risk_multiplier_max",
    "risk_multiplier_relief",
    "risk_scan_interval",
    "risk_min_session_priority",
    "risk_alpha_session_priority",
    "risk_soft_guard_limit",
    "risk_margin_usage_block",
    "risk_low_vol_skip",
    "risk_low_vol_scale",
    "risk_micro_guard_min",
    "risk_high_micro_guard_min",
    "risk_dynamic_var_enabled",
    "risk_equity_governor_enabled",
    "positions_scan_at_entry",
    "positions_total_at_entry",
    "positions_unique_symbols_at_entry",
    "positions_total_volume_at_entry",
    "positions_buy_volume_at_entry",
    "positions_sell_volume_at_entry",
    "positions_net_volume_at_entry",
    "positions_floating_profit_at_entry",
    "positions_floating_swap_at_entry",
    "positions_floating_commission_at_entry",
    "positions_balance_at_entry",
    "positions_equity_at_entry",
    "positions_margin_used_at_entry",
    "positions_margin_free_at_entry",
    "positions_margin_level_at_entry",
    "positions_margin_usage_ratio_at_entry",
    "positions_symbol_concentration_at_entry",
    "positions_top_symbol_volume_at_entry",
    "positions_top_symbol_net_volume_at_entry",
    "positions_top_symbol_profit_at_entry",
    "positions_currency_exposure_total_at_entry",
    "positions_currency_exposure_ratio_at_entry",
    "positions_top_currency_notional_at_entry",
]


@dataclass(slots=True)
class TradeFeatures:
    """Container for engineered trade features."""

    features: pd.DataFrame
    labels: pd.Series
    metadata: pd.DataFrame

    def validate(self) -> None:
        if self.features.empty:
            raise ValueError("Feature frame is empty")
        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels length mismatch")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    return numerator.div(denom)


def _compute_risk_price(df: pd.DataFrame) -> pd.Series:
    risk = pd.Series(np.nan, index=df.index, dtype=float)
    buy_mask = df["direction"] == "buy"
    sell_mask = df["direction"] == "sell"

    if "price_open" in df and "stop_loss" in df:
        risk.loc[buy_mask] = (df.loc[buy_mask, "price_open"] - df.loc[buy_mask, "stop_loss"]).clip(lower=0.0)
        risk.loc[sell_mask] = (df.loc[sell_mask, "stop_loss"] - df.loc[sell_mask, "price_open"]).clip(lower=0.0)
    return risk.fillna(0.0)


def _compute_reward_price(df: pd.DataFrame) -> pd.Series:
    reward = pd.Series(np.nan, index=df.index, dtype=float)
    buy_mask = df["direction"] == "buy"
    sell_mask = df["direction"] == "sell"

    if "price_open" in df and "take_profit" in df:
        reward.loc[buy_mask] = (df.loc[buy_mask, "take_profit"] - df.loc[buy_mask, "price_open"]).clip(lower=0.0)
        reward.loc[sell_mask] = (df.loc[sell_mask, "price_open"] - df.loc[sell_mask, "take_profit"]).clip(lower=0.0)
    return reward.fillna(0.0)


def _bool_to_float(series: pd.Series) -> pd.Series:
    return series.map({True: 1.0, False: 0.0}).astype(float)


def _augment_trade_context(
    trades_df: pd.DataFrame,
    guard_snapshots: pd.DataFrame | None,
    risk_presets: pd.DataFrame | None,
    position_snapshots: pd.DataFrame | None,
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    df["_order"] = np.arange(len(df))
    df["_entry_ts"] = pd.to_datetime(df.get("entry_time"), errors="coerce", utc=True)

    def _can_merge(time_series: pd.Series) -> bool:
        return time_series.notna().any()

    if guard_snapshots is not None and not guard_snapshots.empty and _can_merge(df["_entry_ts"]):
        guard = guard_snapshots.copy()
        guard["timestamp"] = pd.to_datetime(guard.get("timestamp"), errors="coerce", utc=True)
        guard = guard.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not guard.empty:
            guard_subset_cols = [
                "timestamp",
                "combined",
                "guard_factor",
                "soft_factor",
                "margin_factor",
                "equity_factor",
                "var_factor",
                "risk_allowed",
                "risk_status",
                "soft_status",
                "margin_status",
                "var_status",
                "combined_bucket",
                "daily_drawdown",
                "weekly_drawdown",
                "guard_pressure",
                "guard_relief",
            ]
            guard_subset = guard[[col for col in guard_subset_cols if col in guard.columns]]
            merged = pd.merge_asof(
                df.sort_values("_entry_ts"),
                guard_subset,
                left_on="_entry_ts",
                right_on="timestamp",
                direction="backward",
            )
            guard_rename = {
                "timestamp": "guard_timestamp",
                "combined": "guard_combined_at_entry",
                "guard_factor": "guard_factor_at_entry",
                "soft_factor": "guard_soft_factor_at_entry",
                "margin_factor": "guard_margin_factor_at_entry",
                "equity_factor": "guard_equity_factor_at_entry",
                "var_factor": "guard_var_factor_at_entry",
                "risk_allowed": "guard_risk_allowed_at_entry",
                "risk_status": "guard_risk_status_at_entry",
                "soft_status": "guard_soft_status_at_entry",
                "margin_status": "guard_margin_status_at_entry",
                "var_status": "guard_var_status_at_entry",
                "combined_bucket": "guard_combined_bucket_at_entry",
                "daily_drawdown": "guard_daily_drawdown_at_entry",
                "weekly_drawdown": "guard_weekly_drawdown_at_entry",
                "guard_pressure": "guard_pressure_at_entry",
                "guard_relief": "guard_relief_at_entry",
            }
            df = merged.rename(columns=guard_rename)

    if risk_presets is not None and not risk_presets.empty and _can_merge(df["_entry_ts"]):
        risk = risk_presets.copy()
        risk["timestamp"] = pd.to_datetime(risk.get("timestamp"), errors="coerce", utc=True)
        risk = risk.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not risk.empty:
            risk_subset_cols = [
                "timestamp",
                "preset",
                "persona",
                "persona_label",
                "account_risk",
                "risk_multiplier_max",
                "risk_multiplier_relief",
                "scan_interval",
                "min_session_priority",
                "alpha_session_priority",
                "soft_guard_limit",
                "margin_usage_block",
                "low_vol_skip",
                "low_vol_scale",
                "micro_guard_min",
                "high_micro_guard_min",
                "equity_baseline",
                "dynamic_var_enabled",
                "equity_governor_enabled",
            ]
            risk_subset = risk[[col for col in risk_subset_cols if col in risk.columns]]
            merged = pd.merge_asof(
                df.sort_values("_entry_ts"),
                risk_subset,
                left_on="_entry_ts",
                right_on="timestamp",
                direction="backward",
            )
            risk_rename = {
                "timestamp": "risk_timestamp",
                "preset": "context_risk_preset",
                "persona": "context_persona",
                "persona_label": "context_persona_label",
                "account_risk": "risk_account_risk",
                "risk_multiplier_max": "risk_multiplier_max",
                "risk_multiplier_relief": "risk_multiplier_relief",
                "scan_interval": "risk_scan_interval",
                "min_session_priority": "risk_min_session_priority",
                "alpha_session_priority": "risk_alpha_session_priority",
                "soft_guard_limit": "risk_soft_guard_limit",
                "margin_usage_block": "risk_margin_usage_block",
                "low_vol_skip": "risk_low_vol_skip",
                "low_vol_scale": "risk_low_vol_scale",
                "micro_guard_min": "risk_micro_guard_min",
                "high_micro_guard_min": "risk_high_micro_guard_min",
                "equity_baseline": "risk_equity_baseline",
                "dynamic_var_enabled": "risk_dynamic_var_enabled",
                "equity_governor_enabled": "risk_equity_governor_enabled",
            }
            df = merged.rename(columns=risk_rename)

    if position_snapshots is not None and not position_snapshots.empty and _can_merge(df["_entry_ts"]):
        positions = position_snapshots.copy()
        positions["timestamp"] = pd.to_datetime(positions.get("timestamp"), errors="coerce", utc=True)
        positions = positions.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not positions.empty:
            position_subset_cols = [
                "timestamp",
                "scan",
                "total_positions",
                "unique_symbols",
                "total_volume",
                "buy_volume",
                "sell_volume",
                "net_volume",
                "floating_profit",
                "floating_swap",
                "floating_commission",
                "balance",
                "equity",
                "margin_used",
                "margin_free",
                "margin_level",
                "margin_usage_ratio",
                "symbol_concentration",
                "top_symbol",
                "top_symbol_volume",
                "top_symbol_net_volume",
                "top_symbol_profit",
                "currency_exposure_total",
                "currency_exposure_ratio",
                "top_currency",
                "top_currency_notional",
                "symbol_1",
                "symbol_1_volume",
                "symbol_1_net_volume",
                "symbol_1_profit",
                "symbol_2",
                "symbol_2_volume",
                "symbol_2_net_volume",
                "symbol_2_profit",
                "symbol_3",
                "symbol_3_volume",
                "symbol_3_net_volume",
                "symbol_3_profit",
            ]
            position_subset = positions[[col for col in position_subset_cols if col in positions.columns]]
            merged = pd.merge_asof(
                df.sort_values("_entry_ts"),
                position_subset,
                left_on="_entry_ts",
                right_on="timestamp",
                direction="backward",
            )
            position_rename = {
                "timestamp": "positions_timestamp_at_entry",
                "scan": "positions_scan_at_entry",
                "total_positions": "positions_total_at_entry",
                "unique_symbols": "positions_unique_symbols_at_entry",
                "total_volume": "positions_total_volume_at_entry",
                "buy_volume": "positions_buy_volume_at_entry",
                "sell_volume": "positions_sell_volume_at_entry",
                "net_volume": "positions_net_volume_at_entry",
                "floating_profit": "positions_floating_profit_at_entry",
                "floating_swap": "positions_floating_swap_at_entry",
                "floating_commission": "positions_floating_commission_at_entry",
                "balance": "positions_balance_at_entry",
                "equity": "positions_equity_at_entry",
                "margin_used": "positions_margin_used_at_entry",
                "margin_free": "positions_margin_free_at_entry",
                "margin_level": "positions_margin_level_at_entry",
                "margin_usage_ratio": "positions_margin_usage_ratio_at_entry",
                "symbol_concentration": "positions_symbol_concentration_at_entry",
                "top_symbol": "positions_top_symbol_at_entry",
                "top_symbol_volume": "positions_top_symbol_volume_at_entry",
                "top_symbol_net_volume": "positions_top_symbol_net_volume_at_entry",
                "top_symbol_profit": "positions_top_symbol_profit_at_entry",
                "currency_exposure_total": "positions_currency_exposure_total_at_entry",
                "currency_exposure_ratio": "positions_currency_exposure_ratio_at_entry",
                "top_currency": "positions_top_currency_at_entry",
                "top_currency_notional": "positions_top_currency_notional_at_entry",
                "symbol_1": "positions_symbol_1_at_entry",
                "symbol_1_volume": "positions_symbol_1_volume_at_entry",
                "symbol_1_net_volume": "positions_symbol_1_net_volume_at_entry",
                "symbol_1_profit": "positions_symbol_1_profit_at_entry",
                "symbol_2": "positions_symbol_2_at_entry",
                "symbol_2_volume": "positions_symbol_2_volume_at_entry",
                "symbol_2_net_volume": "positions_symbol_2_net_volume_at_entry",
                "symbol_2_profit": "positions_symbol_2_profit_at_entry",
                "symbol_3": "positions_symbol_3_at_entry",
                "symbol_3_volume": "positions_symbol_3_volume_at_entry",
                "symbol_3_net_volume": "positions_symbol_3_net_volume_at_entry",
                "symbol_3_profit": "positions_symbol_3_profit_at_entry",
            }
            df = merged.rename(columns=position_rename)

    df = df.sort_values("_order").drop(columns=["_order", "_entry_ts"], errors="ignore")
    return df


def _ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = np.nan
    return df


def build_trade_features(
    trades_df: pd.DataFrame,
    *,
    include_symbols: bool = True,
    include_strategies: bool = True,
    guard_snapshots: pd.DataFrame | None = None,
    risk_presets: pd.DataFrame | None = None,
    position_snapshots: pd.DataFrame | None = None,
) -> TradeFeatures:
    if trades_df.empty:
        raise ValueError("Trade dataframe is empty")

    df = trades_df.copy()
    df = df.dropna(subset=["outcome", "rr_ratio", "direction", "volume"])
    if df.empty:
        raise ValueError("No trades with outcome/net_result available")

    df = _augment_trade_context(df, guard_snapshots, risk_presets, position_snapshots)

    if "context_risk_preset" not in df.columns:
        df["context_risk_preset"] = "unknown"
    else:
        df["context_risk_preset"] = df["context_risk_preset"].fillna("unknown")
    if "context_persona" not in df.columns:
        df["context_persona"] = "unknown"
    else:
        df["context_persona"] = df["context_persona"].fillna("unknown")
    if "context_persona_label" not in df.columns:
        df["context_persona_label"] = "unknown"
    else:
        df["context_persona_label"] = df["context_persona_label"].fillna("unknown")
    if "guard_timestamp" not in df.columns:
        df["guard_timestamp"] = pd.NaT
    if "risk_timestamp" not in df.columns:
        df["risk_timestamp"] = pd.NaT

    df["direction_encoded"] = df["direction"].map({"buy": 1, "sell": -1}).fillna(0)
    df["risk_price"] = _compute_risk_price(df)
    df["reward_price"] = _compute_reward_price(df)
    df["swap_per_lot"] = _safe_divide(df["swap"], df["volume"])
    df["commission_per_lot"] = _safe_divide(df["commission"], df["volume"])
    df["expected_cost_per_lot"] = (df["swap_per_lot"].fillna(0.0) + df["commission_per_lot"].fillna(0.0))
    net_result_series = df.get("net_result")
    if net_result_series is None:
        net_result_series = pd.Series(np.nan, index=df.index, dtype=float)
    df["net_result_per_lot"] = _safe_divide(net_result_series, df["volume"])
    holding_minutes = df.get("holding_minutes")
    if holding_minutes is not None:
        holding_series = holding_minutes.replace({0: np.nan})
        df["profit_per_minute"] = _safe_divide(net_result_series, holding_series)
    else:
        df["profit_per_minute"] = np.nan

    entry_ts = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df["entry_hour"] = entry_ts.dt.hour.fillna(-1).astype(float)
    df["entry_dayofweek"] = entry_ts.dt.dayofweek.fillna(-1).astype(float)

    for column in [
        "guard_risk_allowed_at_entry",
        "risk_dynamic_var_enabled",
        "risk_equity_governor_enabled",
    ]:
        if column in df.columns:
            df[column] = _bool_to_float(df[column])

    df = _ensure_numeric_columns(df)

    numeric_df = df[NUMERIC_FEATURES].fillna(0.0)

    categoricals: list[pd.DataFrame] = []
    if include_symbols:
        categoricals.append(pd.get_dummies(df["symbol"].fillna("UNKNOWN"), prefix="symbol"))
    if include_strategies:
        categoricals.append(pd.get_dummies(df["strategy_key"].fillna("unknown"), prefix="strategy"))
    categoricals.append(pd.get_dummies(df["context_risk_preset"].fillna("unknown"), prefix="risk"))
    categoricals.append(pd.get_dummies(df["context_persona"].fillna("unknown"), prefix="persona"))
    if "guard_risk_status_at_entry" in df.columns:
        categoricals.append(
            pd.get_dummies(df["guard_risk_status_at_entry"].fillna("unknown"), prefix="guard_risk_status")
        )
    if "guard_combined_bucket_at_entry" in df.columns:
        categoricals.append(
            pd.get_dummies(df["guard_combined_bucket_at_entry"].fillna("unknown"), prefix="guard_combined_bucket")
        )
    if "positions_top_symbol_at_entry" in df.columns:
        categoricals.append(
            pd.get_dummies(df["positions_top_symbol_at_entry"].fillna("none"), prefix="positions_top_symbol")
        )
    if "positions_top_currency_at_entry" in df.columns:
        categoricals.append(
            pd.get_dummies(df["positions_top_currency_at_entry"].fillna("none"), prefix="positions_top_currency")
        )
    for idx in range(1, 4):
        column = f"positions_symbol_{idx}_at_entry"
        if column in df.columns:
            categoricals.append(pd.get_dummies(df[column].fillna("none"), prefix=f"positions_symbol_{idx}"))

    feature_parts: Iterable[pd.DataFrame] = [numeric_df, *categoricals] if categoricals else [numeric_df]
    feature_frame = pd.concat(feature_parts, axis=1)

    labels = (df["outcome"] == "win").astype(int)
    metadata_cols = [
        "ticket",
        "position_id",
        "symbol",
        "strategy_key",
        "direction",
        "outcome",
        "net_result",
        "swap",
        "commission",
        "holding_minutes",
        "entry_time",
        "exit_time",
        "context_risk_preset",
        "context_persona",
        "context_persona_label",
        "guard_timestamp",
        "risk_timestamp",
        "guard_combined_at_entry",
        "risk_account_risk",
        "positions_top_symbol_at_entry",
        "positions_top_currency_at_entry",
    ]
    metadata = df.reindex(columns=metadata_cols).reset_index(drop=True)

    bundle = TradeFeatures(features=feature_frame, labels=labels.reset_index(drop=True), metadata=metadata)
    bundle.validate()
    return bundle


def load_trade_features(config: Optional[TelemetryConfig] = None) -> TradeFeatures:
    datasets = ingest_telemetry(config)
    trade_df = datasets.get("trade_closed")
    if trade_df is None or trade_df.empty:
        raise ValueError("No trade_closed dataset available in telemetry")
    guard_df = datasets.get("guard_snapshots")
    risk_df = datasets.get("risk_preset_applied")
    position_df = datasets.get("position_snapshots")
    return build_trade_features(
        trade_df,
        guard_snapshots=guard_df,
        risk_presets=risk_df,
        position_snapshots=position_df,
    )
