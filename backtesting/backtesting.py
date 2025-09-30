import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agents import TradingAgent
from itertools import product
from time import time
from functools import lru_cache

# Settings
timeframe = mt5.TIMEFRAME_M15
symbols = ["EURUSD+", "USDJPY+", "GBPUSD+", "GBPJPY+", "XAUUSD+"]  # Add more as needed
start_date = datetime.now() - timedelta(days=60)  # 60 days backtest
end_date = datetime.now()

# ATR exit settings (default fallback; individual strategies override below)
ATR_PERIOD = 5
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 2.0
INTRABAR_PRIORITY = 'TP'  # 'SL' or 'TP' if both hit within the same bar

# Strategy parameters
ma_crossover_fast = 5
ma_crossover_slow = 30
meanrev_ma_period = 10
meanrev_num_std = 1
momentum_ma_period = 30
momentum_roc_period = 5
breakout_lookback = 15
donchian_channel_length = 15

STRATEGY_ATR_CONFIG = {
    'ma_crossover': {'period': 5, 'sl_mult': 2.0, 'tp_mult': 2.0, 'priority': 'TP'},
    'mean_reversion': {'period': 5, 'sl_mult': 2.25, 'tp_mult': 4.0, 'priority': 'TP'},
    'momentum_trend': {'period': 5, 'sl_mult': 3.0, 'tp_mult': 4.0, 'priority': 'TP'},
    'breakout': {'period': 5, 'sl_mult': 1.5, 'tp_mult': 3.5, 'priority': 'TP'},
    'donchian_channel': {'period': 5, 'sl_mult': 1.5, 'tp_mult': 3.5, 'priority': 'TP'},
}

DEFAULT_ATR_SETTINGS = {
    'period': ATR_PERIOD,
    'sl_mult': SL_ATR_MULTIPLIER,
    'tp_mult': TP_ATR_MULTIPLIER,
    'priority': INTRABAR_PRIORITY,
}

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


symbol_data = {symbol: get_data(symbol) for symbol in symbols}
symbol_specs = {}

results = {}
agent = TradingAgent(initial_balance=10000)
# Inject ATR/exit params into agent (a simple way to pass defaults)
agent._atr_period = ATR_PERIOD
agent._sl_mult = SL_ATR_MULTIPLIER
agent._tp_mult = TP_ATR_MULTIPLIER
agent._intrabar_priority = INTRABAR_PRIORITY
min_lookback = max(ma_crossover_slow, meanrev_ma_period, momentum_ma_period, breakout_lookback, donchian_channel_length)

for symbol in symbols:
    df = symbol_data.get(symbol)
    if df is None:
        print(f"No data for {symbol}")
        continue
    if len(df) <= min_lookback:
        print(f"Not enough data for {symbol}")
        continue

    # Pull tick specs from MT5 for realistic PnL scaling
    sinfo = mt5.symbol_info(symbol)
    if sinfo is not None:
        tick_size = getattr(sinfo, 'trade_tick_size', sinfo.point if hasattr(sinfo, 'point') else 0.01) or 0.01
        tick_value = getattr(sinfo, 'trade_tick_value', 1.0) or 1.0
        lots = getattr(sinfo, 'trade_lots', None)
        agent._tick_size = tick_size
        agent._tick_value = tick_value
        agent._lots = 1.0 if lots is None else lots
        symbol_specs[symbol] = {
            'tick_size': tick_size,
            'tick_value': tick_value,
            'lots': agent._lots,
        }

    def run_with_atr(strategy_key, func, *args):
        cfg = STRATEGY_ATR_CONFIG.get(strategy_key, DEFAULT_ATR_SETTINGS)
        agent._atr_period = cfg['period']
        agent._sl_mult = cfg['sl_mult']
        agent._tp_mult = cfg['tp_mult']
        agent._intrabar_priority = cfg.get('priority', INTRABAR_PRIORITY)
        return func(df.copy(), *args)

    res_ma = run_with_atr('ma_crossover', agent.ma_crossover, ma_crossover_fast, ma_crossover_slow)
    res_mr = run_with_atr('mean_reversion', agent.mean_reversion, meanrev_ma_period, meanrev_num_std)
    res_mom = run_with_atr('momentum_trend', agent.momentum_trend, momentum_ma_period, momentum_roc_period)
    res_brk = run_with_atr('breakout', agent.breakout, breakout_lookback)
    res_don = run_with_atr('donchian_channel', agent.donchian_channel, donchian_channel_length)
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


def freeze_value(value):
    if isinstance(value, dict):
        return 'dict', tuple(sorted((k, freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return 'tuple', tuple(freeze_value(v) for v in value)
    return 'scalar', value


def thaw_value(frozen):
    kind, data = frozen
    if kind == 'dict':
        return {k: thaw_value(v) for k, v in data}
    if kind == 'tuple':
        return tuple(thaw_value(v) for v in data)
    return data


def run_strategy(agent, strategy, df, params):
    local_df = df.copy()
    if strategy == 'ma_crossover':
        fast, slow = params
        return agent.ma_crossover(local_df, fast, slow)
    if strategy == 'mean_reversion':
        ma_period, num_std = params
        return agent.mean_reversion(local_df, ma_period, num_std)
    if strategy == 'momentum_trend':
        ma_period, roc_period = params
        return agent.momentum_trend(local_df, ma_period, roc_period)
    if strategy == 'breakout':
        lookback = params
        return agent.breakout(local_df, lookback)
    if strategy == 'donchian_channel':
        channel_length = params
        return agent.donchian_channel(local_df, channel_length)
    raise ValueError(f"Unsupported strategy '{strategy}'")


@lru_cache(maxsize=None)
def evaluate_symbol_strategy(strategy, params_key, atr_key, symbol):
    df = symbol_data.get(symbol)
    if df is None or df.empty:
        return None

    params = thaw_value(params_key)
    atr_settings = thaw_value(atr_key)

    agent = TradingAgent(initial_balance=10000)
    agent._atr_period = atr_settings.get('period', ATR_PERIOD)
    agent._sl_mult = atr_settings.get('sl_mult', SL_ATR_MULTIPLIER)
    agent._tp_mult = atr_settings.get('tp_mult', TP_ATR_MULTIPLIER)
    agent._intrabar_priority = atr_settings.get('priority', INTRABAR_PRIORITY)

    specs = symbol_specs.get(symbol)
    if specs:
        agent._tick_size = specs['tick_size']
        agent._tick_value = specs['tick_value']
        agent._lots = specs['lots']

    return run_strategy(agent, strategy, df, params)


def aggregate_strategy_metrics(strategy, params, atr_settings):
    params_key = freeze_value(params)
    atr_key = freeze_value(atr_settings)
    total_profit = 0.0
    total_trades = 0
    win_trades = 0.0
    max_drawdown = 0.0
    per_symbol = {}

    for symbol in symbols:
        res = evaluate_symbol_strategy(strategy, params_key, atr_key, symbol)
        if not res:
            continue
        total_profit += res['total_profit']
        total_trades += res['trades']
        win_trades += res['trades'] * res['win_rate'] / 100.0
        max_drawdown = max(max_drawdown, res['max_drawdown'])
        per_symbol[symbol] = res

    if total_trades == 0:
        avg_win_rate = 0.0
        profit_per_trade = 0.0
    else:
        avg_win_rate = (win_trades / total_trades) * 100.0
        profit_per_trade = total_profit / total_trades

    return {
        'strategy': strategy,
        'params': params if not isinstance(params, dict) else dict(params),
        'atr': dict(atr_settings),
        'total_profit': total_profit,
        'total_trades': total_trades,
        'avg_win_rate': avg_win_rate,
        'profit_per_trade': profit_per_trade,
        'max_drawdown': max_drawdown,
        'per_symbol': per_symbol,
    }


def refine_parameter_grid(strategy, params):
    refined = set()
    if strategy == 'ma_crossover':
        fast, slow = params
        for f in range(max(2, fast - 5), fast + 6, 5):
            for s in range(max(f + 5, slow - 10), slow + 11, 5):
                if f < s:
                    refined.add((f, s))
    elif strategy == 'mean_reversion':
        ma_period, num_std = params
        ma_candidates = range(max(5, ma_period - 5), ma_period + 6, 5)
        std_candidates = [max(0.5, num_std - 0.5), num_std, num_std + 0.5]
        for m in ma_candidates:
            for n in std_candidates:
                refined.add((m, round(n, 2)))
    elif strategy == 'momentum_trend':
        ma_period, roc_period = params
        ma_candidates = range(max(20, ma_period - 10), ma_period + 11, 10)
        roc_candidates = range(max(2, roc_period - 5), roc_period + 6, 5)
        for m in ma_candidates:
            for r in roc_candidates:
                refined.add((m, r))
    elif strategy in ('breakout', 'donchian_channel'):
        lookback = params
        for lb in range(max(5, lookback - 5), lookback + 6, 5):
            refined.add(lb)
    return refined


def refine_atr_grid(atr_settings):
    base_period = atr_settings.get('period', ATR_PERIOD)
    base_sl = atr_settings.get('sl_mult', SL_ATR_MULTIPLIER)
    base_tp = atr_settings.get('tp_mult', TP_ATR_MULTIPLIER)
    base_priority = atr_settings.get('priority', INTRABAR_PRIORITY)

    periods = {base_period}
    for delta in (-2, 2):
        candidate = base_period + delta
        if candidate >= 3:
            periods.add(candidate)

    sl_values = {round(base_sl + delta, 2) for delta in (-0.5, 0.0, 0.5) if base_sl + delta >= 0.5}
    tp_values = {round(base_tp + delta, 2) for delta in (-0.5, 0.0, 0.5) if base_tp + delta >= 0.5}
    priorities = {base_priority}

    return [
        {
            'period': period,
            'sl_mult': sl,
            'tp_mult': tp,
            'priority': priority,
        }
        for period in periods
        for sl in sl_values
        for tp in tp_values
        for priority in priorities
    ]


def search_best_combinations(strategy, param_grid, atr_grid, top_n=5, refine_top=3, max_evaluations=500):
    evaluated = set()
    all_results = []
    eval_count = 0
    dots_cycle = ['.', '..', '...', '....']
    status_index = 0
    last_status_time = 0.0

    def evaluate_combo(params, atr):
        frozen_key = (freeze_value(params), freeze_value(atr))
        if frozen_key in evaluated:
            return None
        nonlocal eval_count
        if max_evaluations and eval_count >= max_evaluations:
            return None
        evaluated.add(frozen_key)
        metrics = aggregate_strategy_metrics(strategy, params, atr)
        if metrics['total_trades'] == 0:
            return None
        all_results.append(metrics)
        eval_count += 1
        emit_progress()
        return metrics

    def emit_progress(force=False):
        nonlocal status_index, last_status_time
        now = time()
        if force or now - last_status_time >= 1.5:
            msg = f"  testing{dots_cycle[status_index % len(dots_cycle)]}"
            print(msg.ljust(20), end='\r', flush=True)
            status_index += 1
            last_status_time = now

    for params in param_grid:
        for atr in atr_grid:
            evaluate_combo(params, atr)
            if max_evaluations and eval_count >= max_evaluations:
                break
        if max_evaluations and eval_count >= max_evaluations:
            break

    all_results.sort(key=lambda x: x['total_profit'], reverse=True)

    if refine_top and all_results:
        seeds = all_results[:min(refine_top, len(all_results))]
        refined_params = set()
        refined_atrs = set()
        for seed in seeds:
            refined_params.update(refine_parameter_grid(strategy, seed['params']))
            for refined_atr in refine_atr_grid(seed['atr']):
                refined_atrs.add(tuple(sorted(refined_atr.items())))

        refined_atr_dicts = [dict(items) for items in refined_atrs] if refined_atrs else []

        for params in refined_params:
            atr_candidates = refined_atr_dicts if refined_atr_dicts else atr_grid
            for atr in atr_candidates:
                evaluate_combo(params, atr)
                if max_evaluations and eval_count >= max_evaluations:
                    break
            if max_evaluations and eval_count >= max_evaluations:
                break

        all_results.sort(key=lambda x: x['total_profit'], reverse=True)

    emit_progress(force=True)
    print("  testing... done!".ljust(20))

    top_results = all_results[:min(top_n, len(all_results))]
    return top_results, all_results, eval_count

# Parameter grids
ma_crossover_fast_values = [5, 10, 15, 20]
ma_crossover_slow_values = [30, 40, 50, 60, 80, 100]
ma_crossover_grid = [(f, s) for f, s in product(ma_crossover_fast_values, ma_crossover_slow_values) if f < s]

meanrev_ma_values = [10, 15, 20, 25, 30]
meanrev_std_values = [1.0, 1.5, 2.0]
meanrev_grid = [(p, round(n, 2)) for p, n in product(meanrev_ma_values, meanrev_std_values)]

momentum_ma_values = [30, 40, 50, 60, 70, 80, 90, 100]
momentum_roc_values = [5, 10, 15, 20, 25, 30]
momentum_grid = [(p, r) for p, r in product(momentum_ma_values, momentum_roc_values)]

breakout_grid = list(range(10, 61, 5))
donchian_grid = list(range(10, 61, 5))

# ATR parameter grids for fine tuning
atr_period_grid = [5, 7, 10, 14, 21]
sl_mult_grid = [1.5, 2.0, 2.25, 2.5, 2.75, 3.0]
tp_mult_grid = [2.0, 2.5, 3.0, 3.5, 4.0]
priority_grid = ['SL', 'TP']

strategy_labels = {
    'ma_crossover': 'MA Crossover',
    'mean_reversion': 'Mean Reversion',
    'momentum_trend': 'Momentum Trend',
    'breakout': 'Breakout',
    'donchian_channel': 'Donchian Channel',
}

print("\nImproved parameter sweep results (top combinations by total profit):")
top_results_per_strategy = {}
best_profit_map = {}

base_atr_grid = [DEFAULT_ATR_SETTINGS]
joint_atr_grid = [
    {'period': ap, 'sl_mult': sl, 'tp_mult': tp, 'priority': prio}
    for ap in atr_period_grid
    for sl in sl_mult_grid
    for tp in tp_mult_grid
    for prio in priority_grid
]

for strategy, grid in [
    ('ma_crossover', ma_crossover_grid),
    ('mean_reversion', meanrev_grid),
    ('momentum_trend', momentum_grid),
    ('breakout', breakout_grid),
    ('donchian_channel', donchian_grid)
]:
    strategy_name = strategy_labels[strategy]
    print(f"\n{strategy_name}:")

    top_basic, _, basic_evals = search_best_combinations(
        strategy,
        grid,
        base_atr_grid,
        top_n=5,
        refine_top=3,
        max_evaluations=150,
    )
    print("  Top parameter sets with default ATR:")
    for idx, result in enumerate(top_basic, start=1):
        params = result['params']
        print(
            f"    {idx}. params={params}, profit={result['total_profit']:.2f}, "
            f"trades={result['total_trades']}, win%={result['avg_win_rate']:.2f}, maxDD={result['max_drawdown']:.2f}"
        )

    top_joint, all_joint, joint_evals = search_best_combinations(
        strategy,
        grid,
        joint_atr_grid,
        top_n=5,
        refine_top=5,
        max_evaluations=350,
    )
    print("  Top parameter + ATR combinations:")
    for idx, result in enumerate(top_joint, start=1):
        params = result['params']
        atr = result['atr']
        print(
            f"    {idx}. params={params}, ATR={atr}, profit={result['total_profit']:.2f}, "
            f"trades={result['total_trades']}, win%={result['avg_win_rate']:.2f}, maxDD={result['max_drawdown']:.2f}"
        )

    print(
        f"  Evaluations performed: default={basic_evals}, joint={joint_evals}, total={basic_evals + joint_evals}"
    )

    if top_joint:
        best_profit_map[strategy_name] = top_joint[0]['total_profit']
        top_results_per_strategy[strategy_name] = top_joint[0]

if best_profit_map:
    priority_order = sorted(best_profit_map.items(), key=lambda item: item[1], reverse=True)
    print("\nSuggested priority order based on aggregated profits:")
    for idx, (strategy_name, profit) in enumerate(priority_order, start=1):
        print(f"  {idx}. {strategy_name} -> total_profit={profit:.2f}")

    print("\nTop recommendations (strategy, params, ATR):")
    for strategy_name, result in top_results_per_strategy.items():
        atr = result['atr']
        print(
            f"  {strategy_name}: params={result['params']}, ATR_PERIOD={atr['period']}, "
            f"SLx={atr['sl_mult']}, TPx={atr['tp_mult']}, priority={atr['priority']}, profit={result['total_profit']:.2f}"
        )

mt5.shutdown()