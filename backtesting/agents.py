import numpy as np
import pandas as pd

class TradingAgent:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        # Defaults for PnL scaling; can be overridden per symbol from outside
        self._tick_size = 0.01
        self._tick_value = 1.0
        self._lots = 1.0
        # NEW: trading hours (24/7 by default)
        self._trade_start_hour = 0     # 0..23
        self._trade_end_hour = 24      # 1..24 (24 means end-of-day)
        self._trade_24_7 = True

    def _within_trading_window(self, dt) -> bool:
        """Return True if entries are allowed at timestamp dt based on configured trading hours."""
        if getattr(self, '_trade_24_7', True):
            return True
        if dt is None:
            return True
        hour = getattr(dt, 'hour', None)
        if hour is None:
            return True
        start = int(getattr(self, '_trade_start_hour', 0))
        end = int(getattr(self, '_trade_end_hour', 24))
        start = max(0, min(23, start))
        end = max(1, min(24, end))
        if start < end:
            return start <= hour < end
        # Overnight window (e.g., 22 -> 6)
        return hour >= start or hour < end

    @staticmethod
    def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR series using True Range rolling mean. No lookahead: ATR at i uses bars up to i."""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _simulate_long_with_sl_tp(
        self,
        df: pd.DataFrame,
        signal_series: pd.Series,
        atr_series: pd.Series,
        atr_period: int,
        sl_atr_mult: float,
        tp_atr_mult: float,
        intrabar_priority: str = 'SL',
        bias_series: pd.Series | None = None,
    ):
        """
        Simulate long and short strategy:
        - Enter long when signal == 1, enter short when signal == -1 (at next open).
        - Manage SL/TP intrabar; if both hit, use intrabar_priority ('SL' or 'TP').
        - Exit on opposite signal at next open (no flip in the same bar).
        No lookahead: decisions at bar i execute at open of bar i+1, ATR from bar i.
        Entries are blocked outside configured trading hours; exits always occur.
        """
        intrabar_priority = (intrabar_priority or 'SL').upper()
        if intrabar_priority not in ('SL', 'TP'):
            intrabar_priority = 'SL'

        trades = []
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = None
        sl_price = None
        tp_price = None
        balance = self.initial_balance
        balance_curve = [balance]

        # NEW: equity curve tracking
        equity_times = []
        equity_values = []
        time_get = (lambda idx: df['time'].iloc[idx]) if 'time' in df.columns else (lambda idx: idx)

        # Start after indicators/ATR available
        start_idx = max(atr_period or 0, 1)

        # Baseline equity point
        equity_times.append(time_get(start_idx))
        equity_values.append(balance)

        for i in range(start_idx, len(df) - 1):
            prev_signal = signal_series.iloc[i]
            next_open = df['open'].iloc[i + 1]

            # Gate entries by trading window using execution bar time
            if position == 0:
                exec_time = df['time'].iloc[i + 1] if 'time' in df.columns else None
                if not self._within_trading_window(exec_time):
                    # Do not enter outside window
                    continue

                # High timeframe bias filter: block entries that don't match bias
                if bias_series is not None and not pd.isna(bias_series.iloc[i]):
                    b = float(bias_series.iloc[i])
                else:
                    b = None

                if prev_signal == 1:
                    if b is not None and b <= 0:
                        # Long signal but non-long bias: skip entry
                        continue
                    # Enter long
                    position = 1
                    entry_price = next_open
                    atr_val = atr_series.iloc[i]
                    if pd.notna(atr_val) and atr_val > 0:
                        sl_dist = sl_atr_mult * atr_val
                        tp_dist = tp_atr_mult * atr_val
                        sl_price = entry_price - sl_dist
                        tp_price = entry_price + tp_dist
                    else:
                        sl_price = None
                        tp_price = None
                    continue
                elif prev_signal == -1:
                    if b is not None and b >= 0:
                        # Short signal but non-short bias: skip entry
                        continue
                    # Enter short
                    position = -1
                    entry_price = next_open
                    atr_val = atr_series.iloc[i]
                    if pd.notna(atr_val) and atr_val > 0:
                        sl_dist = sl_atr_mult * atr_val
                        tp_dist = tp_atr_mult * atr_val
                        sl_price = entry_price + sl_dist  # SL above for shorts
                        tp_price = entry_price - tp_dist  # TP below for shorts
                    else:
                        sl_price = None
                        tp_price = None
                    continue

            # Manage long
            if position == 1:
                exited = False
                exit_price = None

                # First: process exit on opposite signal at the next open (no intrabar peeking)
                if prev_signal == -1:
                    exit_price = next_open
                    exited = True

                if not exited:
                    # Then: manage intrabar SL/TP using this bar's high/low
                    bar_high = df['high'].iloc[i + 1]
                    bar_low = df['low'].iloc[i + 1]

                    hit_sl = sl_price is not None and bar_low <= sl_price
                    hit_tp = tp_price is not None and bar_high >= tp_price

                    if hit_sl and hit_tp:
                        exit_price = sl_price if intrabar_priority == 'SL' else tp_price
                        exited = True
                    elif hit_sl:
                        exit_price = sl_price
                        exited = True
                    elif hit_tp:
                        exit_price = tp_price
                        exited = True

                if exited:
                    tick_size = getattr(self, '_tick_size', 0.01) or 0.01
                    tick_value = getattr(self, '_tick_value', 1.0) or 1.0
                    lots = getattr(self, '_lots', 1.0) or 1.0
                    ticks = (exit_price - entry_price) / tick_size
                    profit = ticks * tick_value * lots
                    trades.append(profit)
                    balance += profit
                    balance_curve.append(balance)
                    equity_times.append(time_get(i + 1))
                    equity_values.append(balance)
                    position = 0
                    entry_price = sl_price = tp_price = None

            # Manage short
            elif position == -1:
                exited = False
                exit_price = None

                # First: process exit on opposite signal at the next open
                if prev_signal == 1:
                    exit_price = next_open
                    exited = True

                if not exited:
                    # Then: manage intrabar SL/TP
                    bar_high = df['high'].iloc[i + 1]
                    bar_low = df['low'].iloc[i + 1]

                    hit_sl = sl_price is not None and bar_high >= sl_price  # SL above
                    hit_tp = tp_price is not None and bar_low <= tp_price   # TP below

                    if hit_sl and hit_tp:
                        exit_price = sl_price if intrabar_priority == 'SL' else tp_price
                        exited = True
                    elif hit_sl:
                        exit_price = sl_price
                        exited = True
                    elif hit_tp:
                        exit_price = tp_price
                        exited = True

                if exited:
                    tick_size = getattr(self, '_tick_size', 0.01) or 0.01
                    tick_value = getattr(self, '_tick_value', 1.0) or 1.0
                    lots = getattr(self, '_lots', 1.0) or 1.0
                    ticks = (entry_price - exit_price) / tick_size  # reverse for shorts
                    profit = ticks * tick_value * lots
                    trades.append(profit)
                    balance += profit
                    balance_curve.append(balance)
                    equity_times.append(time_get(i + 1))
                    equity_values.append(balance)
                    position = 0
                    entry_price = sl_price = tp_price = None

        # Close any open position at last close
        if position != 0 and entry_price is not None:
            last_close = df['close'].iloc[-1]
            tick_size = getattr(self, '_tick_size', 0.01) or 0.01
            tick_value = getattr(self, '_tick_value', 1.0) or 1.0
            lots = getattr(self, '_lots', 1.0) or 1.0
            if position == 1:
                ticks = (last_close - entry_price) / tick_size
            else:
                ticks = (entry_price - last_close) / tick_size
            profit = ticks * tick_value * lots
            trades.append(profit)
            balance += profit
            balance_curve.append(balance)
            equity_times.append(time_get(len(df) - 1))
            equity_values.append(balance)

        total_profit = float(sum(trades))
        win_trades = len([t for t in trades if t > 0])
        win_rate = (win_trades / len(trades) * 100) if trades else 0.0
        max_drawdown = 0.0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        # NEW: additional metrics
        loss_trades = len(trades) - win_trades
        gross_profit = float(sum(t for t in trades if t > 0))
        gross_loss = float(sum(-t for t in trades if t < 0))  # positive number
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        avg_trade = (total_profit / len(trades)) if trades else 0.0
        avg_win = (gross_profit / win_trades) if win_trades > 0 else 0.0
        avg_loss = (gross_loss / loss_trades) if loss_trades > 0 else 0.0
        # Expectancy: average P&L per trade
        expectancy = avg_trade
        # Trade-based Sharpe (proxy): mean/std of trade P&L * sqrt(N)
        if len(trades) > 1:
            mean_t = float(np.mean(trades))
            std_t = float(np.std(trades, ddof=1))
            trade_sharpe = (mean_t / std_t) * np.sqrt(len(trades)) if std_t > 0 else float('inf') if mean_t > 0 else 0.0
        else:
            trade_sharpe = 0.0
        # Max drawdown percent relative to peak
        max_dd_pct = (max_drawdown / peak * 100.0) if peak > 0 else 0.0
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance,
            # NEW: equity curve points for plotting
            'equity_times': equity_times,
            'equity_values': equity_values,
            # NEW: richer metrics for analysis/export
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'trade_sharpe': trade_sharpe,
            'max_drawdown_pct': max_dd_pct,
            # Optional: raw trades for deeper analysis
            'trades_list': trades
        }

    def ma_crossover(self, df, fast_period, slow_period, atr_period=None, sl_mult=None, tp_mult=None, priority=None, bias_series=None):
        df['fast_ma'] = df['close'].rolling(fast_period).mean()
        df['slow_ma'] = df['close'].rolling(slow_period).mean()
        df['signal'] = 0
        df.loc[df.index[slow_period:], 'signal'] = np.where(
            df['fast_ma'][slow_period:] > df['slow_ma'][slow_period:], 1, 0
        )
        df['signal'] = df['signal'].shift(1)
        atr_period = atr_period if atr_period is not None else getattr(self, '_atr_period', 14)
        sl_mult = sl_mult if sl_mult is not None else getattr(self, '_sl_mult', 2.0)
        tp_mult = tp_mult if tp_mult is not None else getattr(self, '_tp_mult', 3.0)
        priority = (priority if priority is not None else getattr(self, '_intrabar_priority', 'SL'))
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority, bias_series=bias_series)

    def mean_reversion(self, df, ma_period, num_std, atr_period=None, sl_mult=None, tp_mult=None, priority=None, bias_series=None):
        df['ma'] = df['close'].rolling(ma_period).mean()
        df['std'] = df['close'].rolling(ma_period).std()
        df['upper'] = df['ma'] + num_std * df['std']
        df['lower'] = df['ma'] - num_std * df['std']
        df['signal'] = 0
        df.loc[df.index[ma_period:], 'signal'] = np.where(df['close'][ma_period:] < df['lower'][ma_period:], 1, 0)
        df.loc[df.index[ma_period:], 'signal'] = np.where(df['close'][ma_period:] > df['upper'][ma_period:], -1, df['signal'][ma_period:])
        df['signal'] = df['signal'].shift(1)
        atr_period = atr_period if atr_period is not None else getattr(self, '_atr_period', 14)
        sl_mult = sl_mult if sl_mult is not None else getattr(self, '_sl_mult', 2.0)
        tp_mult = tp_mult if tp_mult is not None else getattr(self, '_tp_mult', 3.0)
        priority = (priority if priority is not None else getattr(self, '_intrabar_priority', 'SL'))
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority, bias_series=bias_series)

    def momentum_trend(self, df, ma_period=50, roc_period=10, atr_period=None, sl_mult=None, tp_mult=None, priority=None, bias_series=None):
        df['ma'] = df['close'].rolling(ma_period).mean()
        df['roc'] = df['close'].pct_change(roc_period)
        df['signal'] = 0
        df.loc[(df['close'] > df['ma']) & (df['roc'] > 0), 'signal'] = 1
        df.loc[(df['close'] < df['ma']) & (df['roc'] < 0), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        atr_period = atr_period if atr_period is not None else getattr(self, '_atr_period', 14)
        sl_mult = sl_mult if sl_mult is not None else getattr(self, '_sl_mult', 2.0)
        tp_mult = tp_mult if tp_mult is not None else getattr(self, '_tp_mult', 3.0)
        priority = (priority if priority is not None else getattr(self, '_intrabar_priority', 'SL'))
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority, bias_series=bias_series)

    def breakout(self, df, lookback=20, atr_period=None, sl_mult=None, tp_mult=None, priority=None, bias_series=None):
        df['highest'] = df['high'].rolling(lookback).max()
        df['lowest'] = df['low'].rolling(lookback).min()
        df['signal'] = 0
        df.loc[df['close'] > df['highest'].shift(1), 'signal'] = 1
        df.loc[df['close'] < df['lowest'].shift(1), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        atr_period = atr_period if atr_period is not None else getattr(self, '_atr_period', 14)
        sl_mult = sl_mult if sl_mult is not None else getattr(self, '_sl_mult', 2.0)
        tp_mult = tp_mult if tp_mult is not None else getattr(self, '_tp_mult', 3.0)
        priority = (priority if priority is not None else getattr(self, '_intrabar_priority', 'SL'))
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority, bias_series=bias_series)

    def donchian_channel(self, df, channel_length=20, exit_length=None, confirm_bars: int = 1,
                          atr_buffer_mult: float = 0.0, atr_period=None, sl_mult=None, tp_mult=None,
                          priority=None, bias_series=None):
        """
        Improved Donchian:
        - Entry on breakout of channel_length highs/lows using close price vs previous bar's band.
        - Optional ATR buffer: require close > band + k*ATR (and vice versa) to reduce whipsaw.
        - Optional confirmation: condition must hold for 'confirm_bars' consecutive bars before signaling.
        - Optional separate exit_length: if provided, uses opposite breakout of exit_length as exit/reverse trigger.

        Backward-compatible defaults replicate previous behavior when exit_length=None, confirm_bars=1, atr_buffer_mult=0.0.
        """
        channel_length = int(channel_length or 20)
        exit_len = int(exit_length) if exit_length not in (None, '', False) else channel_length
        confirm_bars = max(1, int(confirm_bars or 1))
        atr_period = atr_period if atr_period is not None else getattr(self, '_atr_period', 14)
        sl_mult = sl_mult if sl_mult is not None else getattr(self, '_sl_mult', 2.0)
        tp_mult = tp_mult if tp_mult is not None else getattr(self, '_tp_mult', 3.0)
        priority = (priority if priority is not None else getattr(self, '_intrabar_priority', 'SL'))

        # Bands for entry and exit
        df['dc_high_entry'] = df['high'].rolling(channel_length).max()
        df['dc_low_entry'] = df['low'].rolling(channel_length).min()
        df['dc_high_exit'] = df['high'].rolling(exit_len).max()
        df['dc_low_exit'] = df['low'].rolling(exit_len).min()

        # ATR for optional buffer and for SL/TP calc later
        atr_series = self.compute_atr_series(df, atr_period)
        buffer_series = atr_series * float(atr_buffer_mult or 0.0)

        # Entry conditions with buffer applied to previous band
        long_raw = df['close'] > (df['dc_high_entry'].shift(1) + buffer_series.shift(1).fillna(0.0))
        short_raw = df['close'] < (df['dc_low_entry'].shift(1) - buffer_series.shift(1).fillna(0.0))

        if confirm_bars > 1:
            # Require sustained condition for confirm_bars
            long_cond = long_raw.rolling(confirm_bars).apply(lambda x: 1.0 if np.all(x) else 0.0, raw=False).astype(bool)
            short_cond = short_raw.rolling(confirm_bars).apply(lambda x: 1.0 if np.all(x) else 0.0, raw=False).astype(bool)
        else:
            long_cond = long_raw
            short_cond = short_raw

        # Exit triggers (reverse breakout on shorter band if provided)
        long_exit = df['close'] < df['dc_low_exit'].shift(1)
        short_exit = df['close'] > df['dc_high_exit'].shift(1)

        # Build unified signal series: +1 for long entry, -1 for short entry, and allow reverse exits
        sig = pd.Series(0, index=df.index)
        sig[long_cond] = 1
        sig[short_cond] = -1
        # Encode exits as opposite signals to exit/reverse via simulator logic
        sig[long_exit] = -1
        sig[short_exit] = 1
        # Shift to execute at next open per simulator convention
        df['signal'] = sig.shift(1).fillna(0)

        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority, bias_series=bias_series)
