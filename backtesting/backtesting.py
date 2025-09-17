import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agents import TradingAgent
from itertools import product

# Settings
timeframe = mt5.TIMEFRAME_M15
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "XAUUSD", "USDJPY"]  # Add more as needed
start_date = datetime.now() - timedelta(days=60)  # 60 days backtest
end_date = datetime.now()

# Strategy parameters
ma_crossover_fast = 5
ma_crossover_slow = 20
meanrev_ma_period = 30
meanrev_num_std = 3
momentum_ma_period = 50
momentum_roc_period = 5
breakout_lookback = 10

# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

def get_data(symbol):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print(f"No data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

results = {}
agent = TradingAgent(initial_balance=10000)
for symbol in symbols:
    df = get_data(symbol)
    min_lookback = max(ma_crossover_slow, meanrev_ma_period, momentum_ma_period, breakout_lookback)
    if df is not None and len(df) > min_lookback:
        res_ma = agent.ma_crossover(df.copy(), ma_crossover_fast, ma_crossover_slow)
        res_mr = agent.mean_reversion(df.copy(), meanrev_ma_period, meanrev_num_std)
        res_mom = agent.momentum_trend(df.copy(), momentum_ma_period, momentum_roc_period)
        res_brk = agent.breakout(df.copy(), breakout_lookback)
        res_don = agent.donchian_channel(df.copy(), breakout_lookback)
        results[symbol] = {
            'ma_crossover': res_ma,
            'mean_reversion': res_mr,
            'momentum_trend': res_mom,
            'breakout': res_brk,
            'donchian_channel': res_don
        }
        print(f"{symbol} MA Crossover: Trades={res_ma['trades']}, WinRate={res_ma['win_rate']:.2f}%, Profit={res_ma['total_profit']:.2f}")
        print(f"{symbol} Mean Reversion: Trades={res_mr['trades']}, WinRate={res_mr['win_rate']:.2f}%, Profit={res_mr['total_profit']:.2f}")
        print(f"{symbol} Momentum/Trend: Trades={res_mom['trades']}, WinRate={res_mom['win_rate']:.2f}%, Profit={res_mom['total_profit']:.2f}")
        print(f"{symbol} Breakout: Trades={res_brk['trades']}, WinRate={res_brk['win_rate']:.2f}%, Profit={res_brk['total_profit']:.2f}")
        print(f"{symbol} Donchian Channel: Trades={res_don['trades']}, WinRate={res_don['win_rate']:.2f}%, Profit={res_don['total_profit']:.2f}")
    else:
        print(f"Not enough data for {symbol}")


# # Summary statistics
# def print_summary(results, strategy):
#     total_profit = sum(r[strategy]['total_profit'] for r in results.values())
#     total_trades = sum(r[strategy]['trades'] for r in results.values())
#     total_win_trades = sum(
#         r[strategy]['trades'] * r[strategy]['win_rate'] / 100 for r in results.values()
#     )
#     avg_winrate = (total_win_trades / total_trades * 100) if total_trades else 0
#     profit_ratio = (total_profit / total_trades) if total_trades else 0
#     max_drawdown = max((r[strategy]['max_drawdown'] for r in results.values()), default=0)
#     final_balance = sum(r[strategy]['final_balance'] for r in results.values())
#     print(f"\n{strategy.replace('_', ' ').title()} SUMMARY:")
#     print(f"Initial Balance (per symbol): 10000")
#     print(f"Total Final Balance: {final_balance:.2f}")
#     print(f"Total Profit: {total_profit:.2f}")
#     print(f"Total Trades: {total_trades}")
#     print(f"Total Winrate: {avg_winrate:.2f}%")
#     print(f"Profit Ratio (Profit/Trade): {profit_ratio:.4f}")
#     print(f"Max Drawdown (worst symbol): {max_drawdown:.2f}")


# Print per-symbol statistics
def print_per_symbol_stats(results, strategy):
    print(f"\n{strategy.replace('_', ' ').title()} Per-Symbol Statistics:")
    print(f"{'Symbol':<8} {'Trades':>6} {'WinRate%':>9} {'Profit':>10} {'FinalBal':>10} {'MaxDD':>10}")
    for symbol, res in results.items():
        r = res[strategy]
        print(f"{symbol:<8} {r['trades']:>6} {r['win_rate']:>9.2f} {r['total_profit']:>10.2f} {r['final_balance']:>10.2f} {r['max_drawdown']:>10.2f}")


# Save per-symbol stats to CSV

# def save_per_symbol_stats_to_excel(results, strategy, filename):
#     import pandas as pd
#     data = []
#     for symbol, res in results.items():
#         r = res[strategy]
#         data.append({
#             'Symbol': symbol,
#             'Trades': r['trades'],
#             'WinRate': round(r['win_rate'], 2),
#             'Profit': round(r['total_profit'], 2),
#             'FinalBalance': round(r['final_balance'], 2),
#             'MaxDrawdown': round(r['max_drawdown'], 2)
#         })
#     df = pd.DataFrame(data)
#     df.to_excel(filename, index=False)

# save_per_symbol_stats_to_excel(results, 'ma_crossover', 'ma_crossover_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'mean_reversion', 'mean_reversion_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'momentum_trend', 'momentum_trend_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'breakout', 'breakout_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'donchian_channel', 'donchian_channel_stats.xlsx')

print_per_symbol_stats(results, 'ma_crossover')
print_per_symbol_stats(results, 'mean_reversion')
print_per_symbol_stats(results, 'momentum_trend')
print_per_symbol_stats(results, 'breakout')
print_per_symbol_stats(results, 'donchian_channel')

# print_summary(results, 'ma_crossover')
# print_summary(results, 'mean_reversion')
# print_summary(results, 'momentum_trend')
# print_summary(results, 'breakout')

# Parameter grids
ma_crossover_grid = [(f, s) for f, s in product([5, 10, 20], [20, 50, 100]) if f < s]
meanrev_grid = [(p, n) for p in [10, 20, 30] for n in [1, 2, 3]]
momentum_grid = [(p, r) for p in [20, 50, 100] for r in [5, 10, 20]]
breakout_grid = [l for l in [10, 20, 50]]
donchian_grid = [l for l in [10, 20, 50]]

best_params = {}
for strategy, grid in [
    ('ma_crossover', ma_crossover_grid),
    ('mean_reversion', meanrev_grid),
    ('momentum_trend', momentum_grid),
    ('breakout', breakout_grid),
    ('donchian_channel', donchian_grid)
]:
    best_profit = -np.inf
    best_param = None
    for params in grid:
        total_profit = 0
        for symbol in symbols:
            df = get_data(symbol)
            if df is None:
                continue
            if strategy == 'ma_crossover':
                res = agent.ma_crossover(df.copy(), params[0], params[1])
            elif strategy == 'mean_reversion':
                res = agent.mean_reversion(df.copy(), params[0], params[1])
            elif strategy == 'momentum_trend':
                res = agent.momentum_trend(df.copy(), params[0], params[1])
            elif strategy == 'breakout':
                res = agent.breakout(df.copy(), params)
            elif strategy == 'donchian_channel':
                res = agent.donchian_channel(df.copy(), params)
            else:
                continue
            total_profit += res['total_profit']
        if total_profit > best_profit:
            best_profit = total_profit
            best_param = params
    best_params[strategy] = (best_param, best_profit)

print("\nBest parameters for each strategy (by total profit):")
for strat, (params, profit) in best_params.items():
    print(f"{strat}: params={params}, total_profit={profit:.2f}")

mt5.shutdown()