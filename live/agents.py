import numpy as np
import pandas as pd
from datetime import datetime

# Factory mapping for creating agents from bestconfig strategy keys
STRATEGY_CLASS_MAP = {
    'ma_crossover': 'MACrossoverAgent',
    'mean_reversion': 'MeanReversionAgent',
    'momentum_trend': 'MomentumTrendAgent',
    'breakout': 'BreakoutAgent',
    'donchian_channel': 'DonchianChannelAgent',
}

# --- Trading window helpers ---
def within_trading_window(dt: datetime | None, trade_24_7: bool = True, start_hour: int = 0, end_hour: int = 24) -> bool:
    """Return True if entries are allowed at timestamp dt based on configured trading hours.
    Handles overnight windows (e.g., 22 -> 6). If dt is None, returns True.
    """
    if trade_24_7:
        return True
    if dt is None:
        return True
    try:
        hour = int(getattr(dt, 'hour', None))
    except Exception:
        return True
    start = max(0, min(23, int(start_hour)))
    end = max(1, min(24, int(end_hour)))
    if start < end:
        return start <= hour < end
    # Overnight window (e.g., 22 -> 6)
    return hour >= start or hour < end

def is_within_agent_window(agent, dt: datetime | None) -> bool:
    """Check agent attributes (_trade_24_7, _trade_start_hour, _trade_end_hour) to decide trading permission."""
    t247 = bool(getattr(agent, '_trade_24_7', True))
    sh = int(getattr(agent, '_trade_start_hour', 0))
    eh = int(getattr(agent, '_trade_end_hour', 24))
    return within_trading_window(dt, t247, sh, eh)

class MeanReversionAgent:
    def __init__(self, ma_period=20, num_std=2):
        self.ma_period = ma_period
        self.num_std = num_std
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.ma_period:
            return None
        close = df['close'].iloc[-1]
        ma = df['close'].rolling(self.ma_period).mean().iloc[-1]
        std = df['close'].rolling(self.ma_period).std().iloc[-1]
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        if close < lower:
            signal = 'buy'
        elif close > upper:
            signal = 'sell'
        else:
            signal = None
        self.last_signal = signal
        return signal

class MACrossoverAgent:
    def __init__(self, fast_period=10, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.slow_period:
            return None
        fast_ma = df['close'].rolling(self.fast_period).mean().iloc[-2]
        slow_ma = df['close'].rolling(self.slow_period).mean().iloc[-2]
        prev_fast_ma = df['close'].rolling(self.fast_period).mean().iloc[-3]
        prev_slow_ma = df['close'].rolling(self.slow_period).mean().iloc[-3]
        signal = None
        if prev_fast_ma < prev_slow_ma and fast_ma > slow_ma:
            signal = 'buy'
        elif prev_fast_ma > prev_slow_ma and fast_ma < slow_ma:
            signal = 'sell'
        self.last_signal = signal
        return signal

class MomentumTrendAgent:
    def __init__(self, ma_period=50, roc_period=10):
        self.ma_period = ma_period
        self.roc_period = roc_period
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < max(self.ma_period, self.roc_period) + 1:
            return None
        ma = df['close'].rolling(self.ma_period).mean().iloc[-2]
        roc = df['close'].pct_change(self.roc_period).iloc[-2]
        close = df['close'].iloc[-2]
        signal = None
        if close > ma and roc > 0:
            signal = 'buy'
        elif close < ma and roc < 0:
            signal = 'sell'
        self.last_signal = signal
        return signal

class BreakoutAgent:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.lookback + 2:
            return None
        highest = df['high'].rolling(self.lookback).max().iloc[-2]
        lowest = df['low'].rolling(self.lookback).min().iloc[-2]
        close = df['close'].iloc[-2]
        signal = None
        if close > highest:
            signal = 'buy'
        elif close < lowest:
            signal = 'sell'
        self.last_signal = signal
        return signal

class DonchianChannelAgent:
    def __init__(self, channel_length=20, exit_length=None, confirm_bars: int = 1, atr_buffer_mult: float = 0.0, atr_period: int | None = None):
        self.channel_length = int(channel_length)
        self.exit_length = int(exit_length) if exit_length not in (None, '', False) else int(channel_length)
        self.confirm_bars = max(1, int(confirm_bars))
        self.atr_buffer_mult = float(atr_buffer_mult or 0.0)
        self.atr_period = int(atr_period) if atr_period else None
        self.last_signal = None

    def get_signal(self, df):
        n = int(self.channel_length)
        m = int(self.exit_length)
        if len(df) < max(n, m) + max(2, self.confirm_bars):
            return None
        # Compute bands
        dc_high_entry = df['high'].rolling(n).max()
        dc_low_entry = df['low'].rolling(n).min()
        dc_high_exit = df['high'].rolling(m).max()
        dc_low_exit = df['low'].rolling(m).min()
        # ATR buffer: derive a safe ATR period (prefer explicit self.atr_period; else _atr_period; else 14)
        ap_candidate = self.atr_period if self.atr_period not in (None, '') else getattr(self, '_atr_period', 14)
        try:
            ap = int(ap_candidate)
        except Exception:
            ap = 14
        high = df['high']; low = df['low']; close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(ap).mean()
        buf = atr * float(self.atr_buffer_mult)
        # Entry conditions based on last completed bar (-2)
        idx = -2
        long_raw_series = close > (dc_high_entry.shift(1) + buf.shift(1).fillna(0.0))
        short_raw_series = close < (dc_low_entry.shift(1) - buf.shift(1).fillna(0.0))
        def sustained(series, bars):
            if bars <= 1:
                return bool(series.iloc[idx])
            window = series.iloc[idx - bars + 1: idx + 1]
            return bool(window.all()) if len(window) == bars else False
        long_ok = sustained(long_raw_series, self.confirm_bars)
        short_ok = sustained(short_raw_series, self.confirm_bars)
        # Exit triggers (reverse)
        long_exit = bool(close.iloc[idx] < dc_low_exit.shift(1).iloc[idx])
        short_exit = bool(close.iloc[idx] > dc_high_exit.shift(1).iloc[idx])
        signal = None
        if long_ok:
            signal = 'buy'
        elif short_ok:
            signal = 'sell'
        # Allow explicit reverse on exits if no entry
        if signal is None:
            if long_exit:
                signal = 'sell'
            elif short_exit:
                signal = 'buy'
        self.last_signal = signal
        return signal


def create_agent(strategy_key: str, params: dict):
    """Instantiate a live agent class based on a bestconfig strategy key and parameters.

    Supported strategy keys: ma_crossover, mean_reversion, momentum_trend, breakout, donchian_channel
    """
    name = STRATEGY_CLASS_MAP.get(strategy_key)
    if not name:
        return None
    # Normalize params per known constructor signatures
    try:
        if name == 'MACrossoverAgent':
            return MACrossoverAgent(fast_period=int(params.get('fast', 5)), slow_period=int(params.get('slow', 20)))
        if name == 'MeanReversionAgent':
            return MeanReversionAgent(ma_period=int(params.get('ma_period', 20)), num_std=int(params.get('num_std', 2)))
        if name == 'MomentumTrendAgent':
            return MomentumTrendAgent(ma_period=int(params.get('ma_period', 50)), roc_period=int(params.get('roc_period', 10)))
        if name == 'BreakoutAgent':
            return BreakoutAgent(lookback=int(params.get('lookback', 20)))
        if name == 'DonchianChannelAgent':
            # Support both 'channel_length' and 'lookback' for convenience; add advanced params
            length = int(params.get('channel_length', params.get('lookback', 20)))
            exit_len = params.get('exit_length', None)
            confirm_bars = int(params.get('confirm_bars', 1))
            atr_buf = float(params.get('atr_buffer_mult', 0.0))
            atr_period_val = params.get('atr_period', None)
            try:
                atr_period_num = int(atr_period_val) if atr_period_val not in (None, '') else None
            except Exception:
                atr_period_num = None
            return DonchianChannelAgent(channel_length=length, exit_length=exit_len, confirm_bars=confirm_bars, atr_buffer_mult=atr_buf, atr_period=atr_period_num)
    except Exception:
        return None
    return None