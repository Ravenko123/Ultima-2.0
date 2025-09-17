import numpy as np

class TradingAgent:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance

    def ma_crossover(self, df, fast_period, slow_period):
        df['fast_ma'] = df['close'].rolling(fast_period).mean()
        df['slow_ma'] = df['close'].rolling(slow_period).mean()
        df['signal'] = 0
        df.loc[df.index[slow_period:], 'signal'] = np.where(
            df['fast_ma'][slow_period:] > df['slow_ma'][slow_period:], 1, 0
        )
        df['signal'] = df['signal'].shift(1)
        trades = []
        position = 0
        entry_price = 0
        balance = self.initial_balance
        balance_curve = [balance]
        for i in range(1, len(df)-1):
            prev_signal = df['signal'].iloc[i]
            next_open = df['open'].iloc[i+1]
            if prev_signal == 1 and position == 0:
                position = 1
                entry_price = next_open
            elif prev_signal == 0 and position == 1:
                profit = next_open - entry_price
                trades.append(profit)
                balance += profit
                balance_curve.append(balance)
                position = 0
        total_profit = sum(trades)
        win_trades = len([t for t in trades if t > 0])
        win_rate = win_trades / len(trades) * 100 if trades else 0
        max_drawdown = 0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }

    def mean_reversion(self, df, ma_period, num_std):
        df['ma'] = df['close'].rolling(ma_period).mean()
        df['std'] = df['close'].rolling(ma_period).std()
        df['upper'] = df['ma'] + num_std * df['std']
        df['lower'] = df['ma'] - num_std * df['std']
        df['signal'] = 0
        df.loc[df.index[ma_period:], 'signal'] = np.where(df['close'][ma_period:] < df['lower'][ma_period:], 1, 0)
        df.loc[df.index[ma_period:], 'signal'] = np.where(df['close'][ma_period:] > df['upper'][ma_period:], -1, df['signal'][ma_period:])
        df['signal'] = df['signal'].shift(1)
        trades = []
        position = 0
        entry_price = 0
        balance = self.initial_balance
        balance_curve = [balance]
        for i in range(1, len(df)-1):
            prev_signal = df['signal'].iloc[i]
            next_open = df['open'].iloc[i+1]
            if prev_signal == 1 and position == 0:
                position = 1
                entry_price = next_open
            elif prev_signal == -1 and position == 1:
                profit = next_open - entry_price
                trades.append(profit)
                balance += profit
                balance_curve.append(balance)
                position = 0
        total_profit = sum(trades)
        win_trades = len([t for t in trades if t > 0])
        win_rate = win_trades / len(trades) * 100 if trades else 0
        max_drawdown = 0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }

    def momentum_trend(self, df, ma_period=50, roc_period=10):
        df['ma'] = df['close'].rolling(ma_period).mean()
        df['roc'] = df['close'].pct_change(roc_period)
        df['signal'] = 0
        df.loc[(df['close'] > df['ma']) & (df['roc'] > 0), 'signal'] = 1
        df.loc[(df['close'] < df['ma']) & (df['roc'] < 0), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        trades = []
        position = 0
        entry_price = 0
        balance = self.initial_balance
        balance_curve = [balance]
        for i in range(1, len(df)-1):
            prev_signal = df['signal'].iloc[i]
            next_open = df['open'].iloc[i+1]
            if prev_signal == 1 and position == 0:
                position = 1
                entry_price = next_open
            elif prev_signal == -1 and position == 1:
                profit = next_open - entry_price
                trades.append(profit)
                balance += profit
                balance_curve.append(balance)
                position = 0
        total_profit = sum(trades)
        win_trades = len([t for t in trades if t > 0])
        win_rate = win_trades / len(trades) * 100 if trades else 0
        max_drawdown = 0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }

    def breakout(self, df, lookback=20):
        df['highest'] = df['high'].rolling(lookback).max()
        df['lowest'] = df['low'].rolling(lookback).min()
        df['signal'] = 0
        df.loc[df['close'] > df['highest'].shift(1), 'signal'] = 1
        df.loc[df['close'] < df['lowest'].shift(1), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        trades = []
        position = 0
        entry_price = 0
        balance = self.initial_balance
        balance_curve = [balance]
        for i in range(1, len(df)-1):
            prev_signal = df['signal'].iloc[i]
            next_open = df['open'].iloc[i+1]
            if prev_signal == 1 and position == 0:
                position = 1
                entry_price = next_open
            elif prev_signal == -1 and position == 1:
                profit = next_open - entry_price
                trades.append(profit)
                balance += profit
                balance_curve.append(balance)
                position = 0
        total_profit = sum(trades)
        win_trades = len([t for t in trades if t > 0])
        win_rate = win_trades / len(trades) * 100 if trades else 0
        max_drawdown = 0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }

    def donchian_channel(self, df, channel_length=20):
        df['donchian_high'] = df['high'].rolling(channel_length).max()
        df['donchian_low'] = df['low'].rolling(channel_length).min()
        df['signal'] = 0
        df.loc[df['close'] > df['donchian_high'].shift(1), 'signal'] = 1
        df.loc[df['close'] < df['donchian_low'].shift(1), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        trades = []
        position = 0
        entry_price = 0
        balance = self.initial_balance
        balance_curve = [balance]
        for i in range(1, len(df)-1):
            prev_signal = df['signal'].iloc[i]
            next_open = df['open'].iloc[i+1]
            if prev_signal == 1 and position == 0:
                position = 1
                entry_price = next_open
            elif prev_signal == -1 and position == 1:
                profit = next_open - entry_price
                trades.append(profit)
                balance += profit
                balance_curve.append(balance)
                position = 0
        total_profit = sum(trades)
        win_trades = len([t for t in trades if t > 0])
        win_rate = win_trades / len(trades) * 100 if trades else 0
        max_drawdown = 0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }
