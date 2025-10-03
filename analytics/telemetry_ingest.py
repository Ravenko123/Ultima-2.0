"""Telemetry ingestion utilities for building AI-ready datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import TelemetryConfig


@dataclass(frozen=True)
class GuardSnapshot:
    scan: int
    symbol: str | None
    timestamp: str | None
    guard_factor: float | None
    soft_factor: float | None
    margin_factor: float | None
    equity_factor: float | None
    var_factor: float | None
    combined: float | None
    risk_allowed: bool | None
    risk_status: str | None
    soft_status: str | None
    margin_status: str | None
    var_status: str | None
    combined_bucket: str | None
    daily_drawdown: float | None
    weekly_drawdown: float | None
    guard_pressure: float | None
    guard_relief: float | None


@dataclass(frozen=True)
class RiskPresetApplied:
    timestamp: str | None
    preset: str | None
    persona: str | None
    persona_label: str | None
    changed: bool | None
    account_risk: float | None
    risk_multiplier_max: float | None
    risk_multiplier_relief: float | None
    scan_interval: float | None
    min_session_priority: int | None
    alpha_session_priority: int | None
    soft_guard_limit: float | None
    margin_usage_block: float | None
    low_vol_skip: float | None
    low_vol_scale: float | None
    micro_guard_min: float | None
    high_micro_guard_min: float | None
    equity_baseline: float | None
    dynamic_var_enabled: bool | None
    equity_governor_enabled: bool | None


@dataclass(frozen=True)
class MicroOverride:
    scan: int | None
    symbol: str | None
    signal: str | None
    regime: str | None
    kind: str | None
    guard_factor: float | None
    drawdown_atr: float | None
    momentum: float | None
    threshold: float | None
    extra: dict[str, float] | None


@dataclass(frozen=True)
class PositionSnapshot:
    timestamp: str | None
    scan: int | None
    total_positions: int | None
    unique_symbols: int | None
    total_volume: float | None
    buy_volume: float | None
    sell_volume: float | None
    net_volume: float | None
    floating_profit: float | None
    floating_swap: float | None
    floating_commission: float | None
    balance: float | None
    equity: float | None
    margin_used: float | None
    margin_free: float | None
    margin_level: float | None
    margin_usage_ratio: float | None
    symbol_concentration: float | None
    top_symbol: str | None
    top_symbol_volume: float | None
    top_symbol_net_volume: float | None
    top_symbol_profit: float | None
    currency_exposure_total: float | None
    currency_exposure_ratio: float | None
    top_currency: str | None
    top_currency_notional: float | None
    symbol_1: str | None
    symbol_1_volume: float | None
    symbol_1_net_volume: float | None
    symbol_1_profit: float | None
    symbol_2: str | None
    symbol_2_volume: float | None
    symbol_2_net_volume: float | None
    symbol_2_profit: float | None
    symbol_3: str | None
    symbol_3_volume: float | None
    symbol_3_net_volume: float | None
    symbol_3_profit: float | None


@dataclass(frozen=True)
class TradeClosed:
    ticket: int | None
    position_id: int | None
    magic: int | None
    symbol: str | None
    direction: str | None
    volume: float | None
    entry_time: str | None
    exit_time: str | None
    duration_seconds: int | None
    holding_minutes: float | None
    price_open: float | None
    price_close: float | None
    price_diff_points: float | None
    stop_loss: float | None
    take_profit: float | None
    profit: float | None
    swap: float | None
    commission: float | None
    net_result: float | None
    outcome: str | None
    rr_ratio: float | None
    comment: str | None
    strategy_key: str | None
    strategy_code: str | None


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_timestamp(record: dict[str, object]) -> str | None:
    ts = record.get("timestamp") or record.get("ts")
    if ts is None:
        return None
    text = str(ts).strip()
    return text or None


def _extract_guard_snapshots(records: Iterable[dict[str, object]]) -> list[GuardSnapshot]:
    snapshots: list[GuardSnapshot] = []
    for record in records:
        if record.get("event") != "guard_snapshot":
            continue
        payload = record.get("payload", {})
        snapshots.append(
            GuardSnapshot(
                scan=int(payload.get("scan") or 0),
                symbol=payload.get("symbol"),
                timestamp=_get_timestamp(record),
                guard_factor=_safe_float(payload.get("guard_factor")),
                soft_factor=_safe_float(payload.get("soft_factor")),
                margin_factor=_safe_float(payload.get("margin_factor")),
                equity_factor=_safe_float(payload.get("equity_factor")),
                var_factor=_safe_float(payload.get("var_factor")),
                combined=_safe_float(payload.get("combined")),
                risk_allowed=_safe_bool(payload.get("risk_allowed")),
                risk_status=_safe_str(payload.get("risk_status")),
                soft_status=_safe_str(payload.get("soft_status")),
                margin_status=_safe_str(payload.get("margin_status")),
                var_status=_safe_str(payload.get("var_status")),
                combined_bucket=_safe_str(payload.get("combined_bucket")),
                daily_drawdown=_safe_float(payload.get("daily_drawdown")),
                weekly_drawdown=_safe_float(payload.get("weekly_drawdown")),
                guard_pressure=_safe_float(payload.get("guard_pressure")),
                guard_relief=_safe_float(payload.get("guard_relief")),
            )
        )
    return snapshots


def _extract_micro_overrides(records: Iterable[dict[str, object]]) -> list[MicroOverride]:
    overrides: list[MicroOverride] = []
    for record in records:
        if record.get("event") != "micro_override":
            continue
        payload = record.get("payload", {})
        extra: dict[str, float] = {}
        for key in ("required_alignment", "tolerance", "near_miss_ratio", "neutral_band", "guard_scale"):
            value = payload.get(key)
            if value is not None:
                try:
                    extra[key] = float(value)
                except (TypeError, ValueError):
                    continue
        overrides.append(
            MicroOverride(
                scan=_safe_int(payload.get("scan")),
                symbol=_safe_str(payload.get("symbol")),
                signal=_safe_str(payload.get("signal")),
                regime=_safe_str(payload.get("regime")),
                kind=_safe_str(payload.get("kind")),
                guard_factor=_safe_float(payload.get("guard_factor")),
                drawdown_atr=_safe_float(payload.get("drawdown_atr")),
                momentum=_safe_float(payload.get("momentum")),
                threshold=_safe_float(payload.get("threshold")),
                extra=extra or None,
            )
        )
    return overrides


def _extract_position_snapshots(records: Iterable[dict[str, object]]) -> list[PositionSnapshot]:
    snapshots: list[PositionSnapshot] = []
    for record in records:
        if record.get("event") != "position_snapshot":
            continue
        snapshots.append(
            PositionSnapshot(
                timestamp=_get_timestamp(record),
                scan=_safe_int(record.get("scan")),
                total_positions=_safe_int(record.get("total_positions")),
                unique_symbols=_safe_int(record.get("unique_symbols")),
                total_volume=_safe_float(record.get("total_volume")),
                buy_volume=_safe_float(record.get("buy_volume")),
                sell_volume=_safe_float(record.get("sell_volume")),
                net_volume=_safe_float(record.get("net_volume")),
                floating_profit=_safe_float(record.get("floating_profit")),
                floating_swap=_safe_float(record.get("floating_swap")),
                floating_commission=_safe_float(record.get("floating_commission")),
                balance=_safe_float(record.get("balance")),
                equity=_safe_float(record.get("equity")),
                margin_used=_safe_float(record.get("margin_used")),
                margin_free=_safe_float(record.get("margin_free")),
                margin_level=_safe_float(record.get("margin_level")),
                margin_usage_ratio=_safe_float(record.get("margin_usage_ratio")),
                symbol_concentration=_safe_float(record.get("symbol_concentration")),
                top_symbol=_safe_str(record.get("top_symbol")),
                top_symbol_volume=_safe_float(record.get("top_symbol_volume")),
                top_symbol_net_volume=_safe_float(record.get("top_symbol_net_volume")),
                top_symbol_profit=_safe_float(record.get("top_symbol_profit")),
                currency_exposure_total=_safe_float(record.get("currency_exposure_total")),
                currency_exposure_ratio=_safe_float(record.get("currency_exposure_ratio")),
                top_currency=_safe_str(record.get("top_currency")),
                top_currency_notional=_safe_float(record.get("top_currency_notional")),
                symbol_1=_safe_str(record.get("symbol_1")),
                symbol_1_volume=_safe_float(record.get("symbol_1_volume")),
                symbol_1_net_volume=_safe_float(record.get("symbol_1_net_volume")),
                symbol_1_profit=_safe_float(record.get("symbol_1_profit")),
                symbol_2=_safe_str(record.get("symbol_2")),
                symbol_2_volume=_safe_float(record.get("symbol_2_volume")),
                symbol_2_net_volume=_safe_float(record.get("symbol_2_net_volume")),
                symbol_2_profit=_safe_float(record.get("symbol_2_profit")),
                symbol_3=_safe_str(record.get("symbol_3")),
                symbol_3_volume=_safe_float(record.get("symbol_3_volume")),
                symbol_3_net_volume=_safe_float(record.get("symbol_3_net_volume")),
                symbol_3_profit=_safe_float(record.get("symbol_3_profit")),
            )
        )
    return snapshots


def _extract_risk_preset_applied(records: Iterable[dict[str, object]]) -> list[RiskPresetApplied]:
    presets: list[RiskPresetApplied] = []
    for record in records:
        if record.get("event") != "risk_preset_applied":
            continue
        payload = record.get("payload") or {}
        if not payload:
            payload = {key: value for key, value in record.items() if key not in {"event", "payload"}}
        presets.append(
            RiskPresetApplied(
                timestamp=_get_timestamp(record),
                preset=_safe_str(payload.get("preset") or record.get("preset")),
                persona=_safe_str(payload.get("persona")),
                persona_label=_safe_str(payload.get("persona_label")),
                changed=_safe_bool(payload.get("changed")),
                account_risk=_safe_float(payload.get("account_risk")),
                risk_multiplier_max=_safe_float(payload.get("risk_multiplier_max")),
                risk_multiplier_relief=_safe_float(payload.get("risk_multiplier_relief")),
                scan_interval=_safe_float(payload.get("scan_interval")),
                min_session_priority=_safe_int(payload.get("min_session_priority")),
                alpha_session_priority=_safe_int(payload.get("alpha_session_priority")),
                soft_guard_limit=_safe_float(payload.get("soft_guard_limit")),
                margin_usage_block=_safe_float(payload.get("margin_usage_block")),
                low_vol_skip=_safe_float(payload.get("low_vol_skip")),
                low_vol_scale=_safe_float(payload.get("low_vol_scale")),
                micro_guard_min=_safe_float(payload.get("micro_guard_min")),
                high_micro_guard_min=_safe_float(payload.get("high_micro_guard_min")),
                equity_baseline=_safe_float(payload.get("equity_baseline")),
                dynamic_var_enabled=_safe_bool(payload.get("dynamic_var_enabled")),
                equity_governor_enabled=_safe_bool(payload.get("equity_governor_enabled")),
            )
        )
    return presets


def _extract_trade_closed(records: Iterable[dict[str, object]]) -> list[TradeClosed]:
    trades: list[TradeClosed] = []
    for record in records:
        if record.get("event") != "trade_closed":
            continue
        payload = record.get("payload", {})
        direction = _safe_direction(payload.get("direction"))
        price_open = _safe_float(payload.get("price_open"))
        price_close = _safe_float(payload.get("price_close"))
        stop_loss = _safe_float(payload.get("stop_loss"))
        duration_seconds = _safe_int(payload.get("duration_seconds"))
        net_result = _safe_float(payload.get("net_result"))

        trades.append(
            TradeClosed(
                ticket=_safe_int(payload.get("ticket")),
                position_id=_safe_int(payload.get("position_id")),
                magic=_safe_int(payload.get("magic")),
                symbol=_safe_str(payload.get("symbol")),
                direction=direction,
                volume=_safe_float(payload.get("volume")),
                entry_time=_safe_str(payload.get("entry_time")),
                exit_time=_safe_str(payload.get("exit_time")),
                duration_seconds=duration_seconds,
                holding_minutes=_safe_holding_minutes(duration_seconds),
                price_open=price_open,
                price_close=price_close,
                price_diff_points=_safe_float(payload.get("price_diff_points")),
                stop_loss=stop_loss,
                take_profit=_safe_float(payload.get("take_profit")),
                profit=_safe_float(payload.get("profit")),
                swap=_safe_float(payload.get("swap")),
                commission=_safe_float(payload.get("commission")),
                net_result=net_result,
                outcome=_derive_outcome(net_result),
                rr_ratio=_compute_rr_ratio(direction, price_open, price_close, stop_loss),
                comment=_safe_str(payload.get("comment")),
                strategy_key=_safe_str(payload.get("strategy_key")),
                strategy_code=_safe_str(payload.get("strategy_code")),
            )
        )
    return trades


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    return None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ingest_telemetry(config: TelemetryConfig | None = None) -> dict[str, pd.DataFrame]:
    """Ingest telemetry into AI-friendly DataFrames.

    Returns a dictionary with keys like "guard_snapshots" and "micro_overrides".
    """

    config = config or TelemetryConfig.from_env()
    telemetry_path = config.telemetry_path
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    records = list(_iter_jsonl(telemetry_path))
    guard_snapshots = _extract_guard_snapshots(records)
    micro_overrides = _extract_micro_overrides(records)
    position_snapshots = _extract_position_snapshots(records)
    trade_closed = _extract_trade_closed(records)
    risk_presets = _extract_risk_preset_applied(records)

    datasets: dict[str, pd.DataFrame] = {}
    if guard_snapshots:
        datasets["guard_snapshots"] = pd.DataFrame([vars(s) for s in guard_snapshots])
    if micro_overrides:
        datasets["micro_overrides"] = pd.DataFrame([_flatten_micro_override(o) for o in micro_overrides])
    if position_snapshots:
        datasets["position_snapshots"] = pd.DataFrame([vars(s) for s in position_snapshots])
    if trade_closed:
        datasets["trade_closed"] = pd.DataFrame([vars(t) for t in trade_closed])
    if risk_presets:
        datasets["risk_preset_applied"] = pd.DataFrame([vars(r) for r in risk_presets])
    return datasets


def _flatten_micro_override(override: MicroOverride) -> dict[str, object]:
    base = vars(override).copy()
    extra = base.pop("extra", None) or {}
    for key, value in extra.items():
        base[f"extra_{key}"] = value
    return base


def _safe_holding_minutes(duration_seconds: int | None) -> float | None:
    if duration_seconds is None:
        return None
    try:
        return round(duration_seconds / 60.0, 4)
    except TypeError:
        return None


def _derive_outcome(net_result: float | None) -> str | None:
    if net_result is None:
        return None
    if net_result > 0:
        return "win"
    if net_result < 0:
        return "loss"
    return "flat"


def _compute_rr_ratio(
    direction: str | None,
    price_open: float | None,
    price_close: float | None,
    stop_loss: float | None,
) -> float | None:
    if direction not in {"buy", "sell"}:
        return None
    if price_open is None or stop_loss is None or price_close is None:
        return None
    if direction == "buy":
        risk = price_open - stop_loss if stop_loss is not None else None
        reward = price_close - price_open if price_close is not None else None
    else:
        risk = stop_loss - price_open if stop_loss is not None else None
        reward = price_open - price_close if price_close is not None else None
    if risk is None or reward is None:
        return None
    if risk <= 0:
        return None
    return round(reward / risk, 4)


def _safe_direction(value: object) -> str | None:
    text = _safe_str(value)
    if text in {"buy", "sell"}:
        return text
    return None
