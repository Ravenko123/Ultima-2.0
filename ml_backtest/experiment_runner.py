"""Skeleton experiment runner for the ML-driven backtester."""

from __future__ import annotations

import json
import math
import statistics
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import pandas as pd

from .config import ExperimentConfig
from .data_loader import MarketDataCache
from .parameter_space import ParameterSample
from .risk import (
    PersonaProfile,
    RiskProfile,
    profile_to_dict,
    resolve_persona_profile,
    resolve_risk_profile,
)
from .results import ExperimentArtifact, ExperimentResult, TrialScorecard

__all__: Tuple[str, ...] = (
    "ExperimentContext",
    "TelemetryWriter",
    "ExperimentRunner",
    "timezone",
)


def _normalise_symbol(symbol: str) -> str:
    return symbol.replace("+", "").upper()


_SYMBOL_POINT_VALUE_SCALE: dict[str, float] = {
    "BTCUSD": 1e-3,
    "XAUUSD": 0.1,
}


@dataclass(slots=True)
class ExperimentContext:
    """Runtime helpers shared across experiment phases."""

    config: ExperimentConfig
    sample: ParameterSample
    run_id: str
    started_at: datetime
    telemetry_path: Path
    risk_profile: RiskProfile
    persona_profile: PersonaProfile


class TelemetryWriter:
    """Thin JSONL writer that mirrors the live telemetry schema."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def emit(self, event: str, *, timestamp: datetime | None = None, **payload: object) -> None:
        timestamp = timestamp or datetime.now(timezone.utc)
        record = {
            "ts": timestamp.isoformat(),
            "event": event,
            "payload": payload,
        }
        record.update(payload)
        json.dump(record, self._handle, separators=(",", ":"))
        self._handle.write("\n")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ExperimentRunner:
    """Executes a backtest experiment for a given parameter sample."""

    def __init__(self, config: ExperimentConfig, cache: MarketDataCache) -> None:
        self.config = config
        self.cache = cache

    def run(self, sample: ParameterSample) -> ExperimentResult:
        run_id = self._build_run_id()
        telemetry_path = self.config.telemetry.run_directory(run_id) / "telemetry.jsonl"
        started_at = datetime.now(timezone.utc)
        risk_name = sample.values.get("risk_preset") or (self.config.risk_presets[0] if self.config.risk_presets else None)
        persona_name = sample.values.get("persona") or (self.config.personas[0] if self.config.personas else None)
        risk_profile = resolve_risk_profile(risk_name if isinstance(risk_name, str) else None)
        persona_profile = resolve_persona_profile(persona_name if isinstance(persona_name, str) else None)
        drawdown_override_pct: float | None = None
        raw_drawdown = sample.values.get("drawdown_limit_pct")
        if raw_drawdown is not None:
            try:
                drawdown_override_pct = max(0.0, float(raw_drawdown))
            except (TypeError, ValueError):
                drawdown_override_pct = None

        ctx = ExperimentContext(
            config=self.config,
            sample=sample,
            run_id=run_id,
            started_at=started_at,
            telemetry_path=telemetry_path,
            risk_profile=risk_profile,
            persona_profile=persona_profile,
        )

        start, end = self.config.time_range.resolve(now=started_at)
        datasets = self.cache.load_all(start=start, end=end)

        trades = []
        total_mtf_rejections = 0
        with TelemetryWriter(telemetry_path) as writer:
            writer.emit(
                "run_context",
                timestamp=started_at,
                run_id=run_id,
                sample=sample.as_dict(),
                risk_preset=risk_profile.name,
                risk_details=profile_to_dict(risk_profile),
                persona=persona_profile.name,
                persona_details=profile_to_dict(persona_profile),
                drawdown_override_pct=drawdown_override_pct,
            )
            ticket_counter = 1
            for symbol, frame in datasets.items():
                symbol_trades, ticket_counter, mtf_rejections = self._simulate_symbol(
                    ctx,
                    symbol,
                    frame,
                    ticket_counter,
                    writer,
                )
                trades.extend(symbol_trades)
                total_mtf_rejections += mtf_rejections

            equity_snapshots = self._build_equity_curve(trades, initial_balance=self.config.initial_balance)
            for snapshot in equity_snapshots:
                timestamp = snapshot["timestamp"]
                payload = {key: value for key, value in snapshot.items() if key != "timestamp"}
                payload["run_id"] = run_id
                writer.emit("equity_curve_point", timestamp=timestamp, **payload)
            writer.emit(
                "mtf_filter_summary",
                run_id=run_id,
                total_rejections=total_mtf_rejections,
            )

        scorecard = self._score_trades(
            trades,
            risk_profile=risk_profile,
            drawdown_override_pct=drawdown_override_pct,
            mtf_reject_count=total_mtf_rejections,
        )
        artifacts = [
            ExperimentArtifact(
                run_id=run_id,
                telemetry_file=telemetry_path,
            )
        ]
        result = ExperimentResult(
            sample=sample,
            scorecard=scorecard,
            artifacts=artifacts,
            started_at=started_at,
        )
        result.mark_finished()
        return result

    def _simulate_symbol(
        self,
        ctx: ExperimentContext,
        symbol: str,
        frame: pd.DataFrame,
        ticket_start: int,
        writer: TelemetryWriter,
    ) -> tuple[list[dict[str, object]], int, int]:
        sample = ctx.sample
        risk_profile = ctx.risk_profile
        persona_profile = ctx.persona_profile
        if frame.empty:
            return [], ticket_start, 0

        frame = frame.sort_values("time").reset_index(drop=True)
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["high"] = pd.to_numeric(frame.get("high", frame["close"]), errors="coerce")
        frame["low"] = pd.to_numeric(frame.get("low", frame["close"]), errors="coerce")
        frame = frame.dropna(subset=["time", "close", "high", "low"]).reset_index(drop=True)
        if frame.empty:
            return [], ticket_start, 0

        fast_window = int(sample.values.get("fast_window", 12)) + persona_profile.fast_window_bias
        fast_window = max(2, fast_window)
        slow_window = int(sample.values.get("slow_window", max(24, fast_window + 4))) + persona_profile.slow_window_bias
        if slow_window <= fast_window:
            slow_window = fast_window + 4
        atr_window = int(sample.values.get("atr_window", 14)) + persona_profile.atr_window_bias
        atr_window = max(2, atr_window)
        take_mult = float(sample.values.get("take_profit_atr", 1.5))
        stop_mult = float(sample.values.get("stop_loss_atr", 1.0))
        max_holding = max(1, int(sample.values.get("max_holding_bars", 12)))
        volume = float(sample.values.get("base_volume", 0.1))
        point_value = float(sample.values.get("pip_value", 1000.0))
        commission_per_lot = float(sample.values.get("commission_per_lot", -0.5))
        spread_bps = float(sample.values.get("spread_bps", 5.0))
        slippage_bps = float(sample.values.get("slippage_bps", 2.0))
        breakeven_trigger = max(0.0, float(sample.values.get("breakeven_atr_trigger", 0.0)))
        trailing_start = max(0.0, float(sample.values.get("trailing_start_atr", 0.0)))
        trailing_distance = max(0.0, float(sample.values.get("trailing_distance_atr", 0.0)))
        mtf_state = str(sample.values.get("mtf_confirmation", "on")).strip().lower()
        mtf_enabled = mtf_state not in {"off", "0", "false", "no"}
        mtf_short_window = max(1, int(sample.values.get("mtf_short_window", 5)))
        mtf_long_window = max(mtf_short_window + 1, int(sample.values.get("mtf_long_window", 10)))

        symbol_key = _normalise_symbol(symbol)
        point_value_scale = _SYMBOL_POINT_VALUE_SCALE.get(symbol_key, 1.0)

        take_mult *= risk_profile.take_profit_multiplier * persona_profile.take_profit_multiplier
        stop_mult *= risk_profile.stop_loss_multiplier * persona_profile.stop_loss_multiplier
        volume *= risk_profile.volume_multiplier * persona_profile.volume_multiplier
        max_holding = max(1, int(round(max_holding * persona_profile.max_holding_multiplier)))
        spread_bps = max(0.0, spread_bps + risk_profile.spread_bps)
        slippage_bps = max(0.0, slippage_bps + risk_profile.slippage_bps)

        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        times = frame["time"].tolist()

        fast_ma = close.rolling(fast_window, min_periods=fast_window).mean()
        slow_ma = close.rolling(slow_window, min_periods=slow_window).mean()
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(atr_window, min_periods=atr_window).mean()

        mtf_lookup = None
        if mtf_enabled:
            mtf_lookup = self._build_mtf_bias_lookup(
                frame,
                short_window=mtf_short_window,
                long_window=mtf_long_window,
            )
            if mtf_lookup is None:
                mtf_enabled = False
        mtf_reject_count = 0

        class Position:
            __slots__ = (
                "ticket",
                "direction",
                "entry_price",
                "entry_exec_price",
                "entry_time",
                "stop_loss",
                "take_profit",
                "atr_at_entry",
                "commission",
                "bars_open",
                "highest_price",
                "lowest_price",
                "mtf_bias",
            )

            def __init__(
                self,
                *,
                ticket: int,
                direction: str,
                entry_price: float,
                entry_exec_price: float,
                entry_time: datetime,
                stop_loss: float,
                take_profit: float,
                atr_at_entry: float,
                commission: float,
                mtf_bias: str,
            ) -> None:
                self.ticket = ticket
                self.direction = direction
                self.entry_price = entry_price
                self.entry_exec_price = entry_exec_price
                self.entry_time = entry_time
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.atr_at_entry = atr_at_entry
                self.commission = commission
                self.bars_open = 0
                self.highest_price = entry_exec_price
                self.lowest_price = entry_exec_price
                self.mtf_bias = mtf_bias

        def _bps_to_price(base_price: float, bps: float) -> float:
            return base_price * (bps / 10_000.0)

        trades: list[dict[str, object]] = []
        position: Position | None = None
        ticket = ticket_start

        start_index = max(fast_window, slow_window, atr_window)
        for idx in range(start_index, len(frame)):
            timestamp = times[idx]
            atr_value = float(atr.iloc[idx]) if not math.isnan(atr.iloc[idx]) else None
            if atr_value is None or atr_value <= 0:
                continue

            fast_val = float(fast_ma.iloc[idx]) if not math.isnan(fast_ma.iloc[idx]) else None
            slow_val = float(slow_ma.iloc[idx]) if not math.isnan(slow_ma.iloc[idx]) else None
            if fast_val is None or slow_val is None:
                continue

            if position is None:
                prev_fast = float(fast_ma.iloc[idx - 1]) if not math.isnan(fast_ma.iloc[idx - 1]) else None
                prev_slow = float(slow_ma.iloc[idx - 1]) if not math.isnan(slow_ma.iloc[idx - 1]) else None
                if prev_fast is None or prev_slow is None:
                    continue

                direction: str | None = None
                if fast_val > slow_val and prev_fast <= prev_slow:
                    direction = "buy"
                elif fast_val < slow_val and prev_fast >= prev_slow:
                    direction = "sell"

                if direction is None:
                    continue

                mtf_bias = "neutral"
                if mtf_lookup is not None:
                    try:
                        mtf_bias = str(mtf_lookup(timestamp))
                    except Exception:
                        mtf_bias = "neutral"
                if mtf_enabled:
                    if direction == "buy" and mtf_bias == "bearish":
                        mtf_reject_count += 1
                        writer.emit(
                            "mtf_filter_reject",
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=direction,
                            mtf_bias=mtf_bias,
                            run_id=ctx.run_id,
                        )
                        continue
                    if direction == "sell" and mtf_bias == "bullish":
                        mtf_reject_count += 1
                        writer.emit(
                            "mtf_filter_reject",
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=direction,
                            mtf_bias=mtf_bias,
                            run_id=ctx.run_id,
                        )
                        continue

                raw_entry_price = float(close.iloc[idx])
                spread = _bps_to_price(raw_entry_price, spread_bps)
                slip = _bps_to_price(raw_entry_price, slippage_bps)
                if direction == "buy":
                    exec_price = raw_entry_price + spread + slip
                    stop_loss = exec_price - stop_mult * atr_value
                    take_profit = exec_price + take_mult * atr_value
                else:
                    exec_price = raw_entry_price - spread - slip
                    stop_loss = exec_price + stop_mult * atr_value
                    take_profit = exec_price - take_mult * atr_value

                position = Position(
                    ticket=ticket,
                    direction=direction,
                    entry_price=raw_entry_price,
                    entry_exec_price=exec_price,
                    entry_time=timestamp,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    atr_at_entry=atr_value,
                    commission=volume * commission_per_lot,
                    mtf_bias=mtf_bias,
                )
                ticket += 1
                continue

            # Manage open position
            position.bars_open += 1
            exit_reason: str | None = None
            exit_price: float | None = None
            high_price = float(high.iloc[idx])
            low_price = float(low.iloc[idx])
            close_price = float(close.iloc[idx])
            slip = _bps_to_price(close_price, slippage_bps)

            if position.direction == "buy":
                position.highest_price = max(position.highest_price, high_price)
                if breakeven_trigger > 0.0 and position.stop_loss < position.entry_exec_price:
                    if position.highest_price - position.entry_exec_price >= breakeven_trigger * position.atr_at_entry:
                        position.stop_loss = position.entry_exec_price
                if trailing_start > 0.0 and trailing_distance > 0.0:
                    if position.highest_price - position.entry_exec_price >= trailing_start * position.atr_at_entry:
                        trail_stop = position.highest_price - trailing_distance * position.atr_at_entry
                        position.stop_loss = max(position.stop_loss, trail_stop)
            else:
                position.lowest_price = min(position.lowest_price, low_price)
                if breakeven_trigger > 0.0 and position.stop_loss > position.entry_exec_price:
                    if position.entry_exec_price - position.lowest_price >= breakeven_trigger * position.atr_at_entry:
                        position.stop_loss = position.entry_exec_price
                if trailing_start > 0.0 and trailing_distance > 0.0:
                    if position.entry_exec_price - position.lowest_price >= trailing_start * position.atr_at_entry:
                        trail_stop = position.lowest_price + trailing_distance * position.atr_at_entry
                        position.stop_loss = min(position.stop_loss, trail_stop)

            if position.direction == "buy":
                hit_stop = low_price <= position.stop_loss
                hit_target = high_price >= position.take_profit
                if hit_stop:
                    exit_price = position.stop_loss - slip
                    exit_reason = "stop"
                elif hit_target:
                    exit_price = position.take_profit - slip
                    exit_reason = "target"
            else:
                hit_stop = high_price >= position.stop_loss
                hit_target = low_price <= position.take_profit
                if hit_stop:
                    exit_price = position.stop_loss + slip
                    exit_reason = "stop"
                elif hit_target:
                    exit_price = position.take_profit + slip
                    exit_reason = "target"

            if exit_price is None and position.bars_open >= max_holding:
                exit_reason = "timeout"
                if position.direction == "buy":
                    exit_price = close_price - slip
                else:
                    exit_price = close_price + slip

            if exit_price is None:
                continue

            exit_price = max(exit_price, 0.0)
            direction = position.direction
            price_diff = exit_price - position.entry_exec_price
            effective_point_value = point_value * point_value_scale
            if direction == "buy":
                gross_result = price_diff * effective_point_value * volume
            else:
                gross_result = -price_diff * effective_point_value * volume

            net_result = gross_result + position.commission
            duration_seconds = int((timestamp - position.entry_time).total_seconds())
            trade = {
                "ticket": position.ticket,
                "position_id": position.ticket,
                "symbol": symbol,
                "direction": direction,
                "volume": volume,
                "entry_time": position.entry_time.isoformat(),
                "exit_time": timestamp.isoformat(),
                "duration_seconds": max(duration_seconds, 0),
                "price_open": position.entry_exec_price,
                "price_close": exit_price,
                "raw_price_open": position.entry_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "atr_at_entry": position.atr_at_entry,
                "price_diff_points": price_diff * 10_000 * point_value_scale,
                "profit": gross_result,
                "swap": 0.0,
                "commission": position.commission,
                "net_result": net_result,
                "exit_reason": exit_reason,
                "bars_held": position.bars_open,
                "comment": "mlbt_ma_atr",
                "strategy_key": "mlbt_ma_atr",
                "strategy_code": "mlbt_ma_atr",
                "risk_preset": risk_profile.name,
                "persona": persona_profile.name,
                "mtf_bias": position.mtf_bias,
            }
            trade["outcome"] = "win" if trade["net_result"] > 0 else ("loss" if trade["net_result"] < 0 else "flat")
            trades.append(trade)
            writer.emit("trade_closed", timestamp=timestamp, **trade)
            position = None

        return trades, ticket, mtf_reject_count

    def _build_mtf_bias_lookup(
        self,
        frame: pd.DataFrame,
        *,
        short_window: int,
        long_window: int,
    freq: str = "1h",
    ) -> Callable[[datetime], str] | None:
        if frame.empty:
            return None

        closes = (
            frame.set_index("time")
            .sort_index()["close"]
            .astype(float)
            .resample(freq)
            .last()
            .dropna()
        )
        if closes.empty:
            return None

        ma_short = closes.rolling(short_window, min_periods=short_window).mean()
        ma_long = closes.rolling(long_window, min_periods=long_window).mean()

        bias = pd.Series(data="neutral", index=closes.index, dtype="object")
        bullish_mask = (ma_short > ma_long) & (closes > ma_short)
        bearish_mask = (ma_short < ma_long) & (closes < ma_short)
        bias.loc[bullish_mask] = "bullish"
        bias.loc[bearish_mask] = "bearish"
        bias = bias.ffill().fillna("neutral")

        bias_pairs = []
        for ts, val in bias.items():
            if ts is None or pd.isna(ts) or val is None:
                continue
            bias_pairs.append((ts.to_pydatetime(), str(val)))
        if not bias_pairs:
            return None
        bias_times, bias_values = zip(*bias_pairs)
        bias_times = list(bias_times)
        bias_values = list(bias_values)

        def lookup(ts: datetime) -> str:
            pos = bisect_right(bias_times, ts) - 1
            if pos >= 0:
                return bias_values[pos]
            return "neutral"

        return lookup

    def _score_trades(
        self,
        trades: Sequence[Dict[str, object]],
        *,
        risk_profile: RiskProfile,
        drawdown_override_pct: float | None,
        mtf_reject_count: int = 0,
    ) -> TrialScorecard:
        trade_count = len(trades)
        if trade_count == 0:
            metrics = {
                "trade_count": 0.0,
                "mtf_filter_rejections": float(max(mtf_reject_count, 0)),
            }
            return TrialScorecard(metrics=metrics, constraints={"min_trades": False})

        sorted_trades = sorted(trades, key=lambda trade: trade.get("exit_time", ""))
        initial_balance = float(self.config.initial_balance)
        equity = initial_balance
        peak_equity = initial_balance

        net_results = [float(trade["net_result"]) for trade in sorted_trades]
        wins = sum(1 for trade in sorted_trades if trade.get("outcome") == "win")
        losses = sum(1 for trade in sorted_trades if trade.get("outcome") == "loss")
        total_net = sum(net_results)
        durations = [int(trade.get("duration_seconds", 0)) for trade in sorted_trades]

        gross_profit = sum(max(result, 0.0) for result in net_results)
        gross_loss = sum(min(result, 0.0) for result in net_results)
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)
        elif gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = 0.0

        max_drawdown = 0.0
        for trade_result in net_results:
            equity += trade_result
            if equity > peak_equity:
                peak_equity = equity
            if peak_equity > 0:
                drawdown = (peak_equity - equity) / peak_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        avg_net = statistics.mean(net_results) if net_results else 0.0
        win_rate = wins / trade_count if trade_count else 0.0
        loss_rate = losses / trade_count if trade_count else 0.0
        avg_duration = statistics.mean(durations) if durations else 0.0
        return_pct = ((equity - initial_balance) / initial_balance * 100.0) if initial_balance else 0.0

        drawdown_limit = risk_profile.max_drawdown_pct if risk_profile.max_drawdown_pct is not None else 0.35
        if drawdown_override_pct is not None:
            override_fraction = float(drawdown_override_pct) / 100.0
            if override_fraction > 0:
                drawdown_limit = override_fraction

        metrics = {
            "trade_count": float(trade_count),
            "wins": float(wins),
            "losses": float(losses),
            "net_profit": float(total_net),
            "avg_net_result": float(avg_net),
            "win_rate": float(win_rate),
            "loss_rate": float(loss_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_drawdown * 100.0),
            "final_equity": float(equity),
            "return_pct": float(return_pct),
            "avg_duration_sec": float(avg_duration),
            "risk_volume_multiplier": float(risk_profile.volume_multiplier),
            "risk_take_profit_multiplier": float(risk_profile.take_profit_multiplier),
            "risk_stop_loss_multiplier": float(risk_profile.stop_loss_multiplier),
            "risk_spread_bps": float(risk_profile.spread_bps),
            "risk_slippage_bps": float(risk_profile.slippage_bps),
            "risk_drawdown_limit_pct": float(drawdown_limit * 100.0),
            "mtf_filter_rejections": float(max(mtf_reject_count, 0)),
        }
        if drawdown_override_pct is not None and drawdown_override_pct > 0:
            metrics["drawdown_override_pct"] = float(drawdown_override_pct)
        if trade_count >= 2:
            metrics["net_std"] = float(statistics.pstdev(net_results))
            if initial_balance:
                returns = [result / initial_balance for result in net_results]
                if len(returns) >= 2 and any(value != 0.0 for value in returns):
                    metrics["net_sharpe"] = float(
                        statistics.mean(returns) / (statistics.pstdev(returns) or 1e-9) * math.sqrt(len(returns))
                    )
                else:
                    metrics["net_sharpe"] = 0.0

        constraints = {
            "min_trades": trade_count >= 10,
            "max_drawdown_acceptable": max_drawdown <= drawdown_limit,
        }
        return TrialScorecard(metrics=metrics, constraints=constraints)

    def _build_equity_curve(
        self,
        trades: Sequence[Dict[str, object]],
        *,
        initial_balance: float,
    ) -> list[dict[str, object]]:
        if not trades:
            return []

        equity = float(initial_balance)
        peak = float(initial_balance)
        snapshots: list[dict[str, object]] = []
        for trade in sorted(trades, key=lambda trade: trade.get("exit_time", "")):
            net_result = float(trade.get("net_result", 0.0))
            exit_time_raw = trade.get("exit_time")
            timestamp: datetime
            if isinstance(exit_time_raw, datetime):
                timestamp = exit_time_raw
            elif isinstance(exit_time_raw, str):
                try:
                    timestamp = datetime.fromisoformat(exit_time_raw)
                except ValueError:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            equity += net_result
            if equity > peak:
                peak = equity
            drawdown = 0.0 if peak <= 0 else (peak - equity) / peak
            snapshots.append(
                {
                    "timestamp": timestamp,
                    "equity": float(equity),
                    "drawdown_pct": float(drawdown * 100.0),
                    "net_result": float(net_result),
                    "ticket": trade.get("ticket"),
                    "risk_preset": trade.get("risk_preset"),
                    "persona": trade.get("persona"),
                }
            )

        return snapshots

    @staticmethod
    def _build_run_id() -> str:
        now = datetime.now(timezone.utc)
        return f"run_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
