import numpy as np

class MeanReversionAgent:
    def __init__(self, ma_period=10, num_std=1):
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
    def __init__(self, fast_period=5, slow_period=40):
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
    def __init__(self, ma_period=30, roc_period=5):
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
    def __init__(self, lookback=10):
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
    def __init__(self, channel_length=10):
        self.channel_length = channel_length
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.channel_length + 2:
            return None
        donchian_high = df['high'].rolling(self.channel_length).max().iloc[-2]
        donchian_low = df['low'].rolling(self.channel_length).min().iloc[-2]
        channel_width = (donchian_high - donchian_low)
        close = df['close'].iloc[-2]
        signal = None
        if channel_width <= close * 0.0005:
            signal = None
        elif close <= donchian_low:
            signal = 'buy'
        elif close >= donchian_high:
            signal = 'sell'
        self.last_signal = signal
        return signal


class ImpulseScalpAgent:
    def __init__(
        self,
        ema_fast: int = 3,
        ema_slow: int = 8,
        rsi_period: int = 7,
        vol_floor: float = 0.0006,
        rsi_upper: float = 68.0,
        rsi_lower: float = 32.0,
    ):
        self.ema_fast = max(1, int(ema_fast))
        self.ema_slow = max(self.ema_fast + 1, int(ema_slow))
        self.rsi_period = max(2, int(rsi_period))
        self.vol_floor = float(max(0.0, vol_floor))
        self.rsi_upper = float(rsi_upper)
        self.rsi_lower = float(rsi_lower)
        self.last_signal = None

    def get_signal(self, df):
        min_length = max(self.ema_slow, self.rsi_period) + 5
        if len(df) < min_length:
            return None

        close = df['close']
        fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        cross = fast - slow
        curr_cross = cross.iloc[-2]
        prev_cross = cross.iloc[-3]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan) + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = float(rsi.iloc[-2])

        vol_window = close.pct_change().rolling(self.rsi_period * 2).std()
        vol_value = float(vol_window.iloc[-2]) if not np.isnan(vol_window.iloc[-2]) else 0.0
        if vol_value < self.vol_floor:
            self.last_signal = None
            return None

        signal = None
        momentum = close.iloc[-2] - close.iloc[-4]
        if prev_cross <= 0 and curr_cross > 0 and rsi_value < self.rsi_upper and momentum > 0:
            signal = 'buy'
        elif prev_cross >= 0 and curr_cross < 0 and rsi_value > self.rsi_lower and momentum < 0:
            signal = 'sell'

        self.last_signal = signal
        return signal