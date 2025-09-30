import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from agents import MeanReversionAgent, MACrossoverAgent, MomentumTrendAgent, BreakoutAgent, DonchianChannelAgent
import time
import re

# Settings
symbols = ["EURUSD+", "USDJPY+", "GBPUSD+", "GBPJPY+", "XAUUSD+"]
timeframe = mt5.TIMEFRAME_M15
bars = 100  # Number of bars to fetch

# Risk/target settings (ATR-based) - OPTIMIZED FROM BACKTESTING
ATR_PERIOD = 5  # Base period from latest optimization (strategies currently share this value)
SL_ATR_MULTIPLIER = 2.0  # Default fallback; individual strategies can override
TP_ATR_MULTIPLIER = 2.0  # Default fallback; individual strategies can override
RESPECT_STOPS_LEVEL = True  # Enforce broker minimal stop distance if provided
ALLOW_HEDGING = False  # When False, wait for existing positions to close before taking opposite trades

# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

# Agent settings for each strategy - OPTIMIZED FROM BACKTESTING
agent_definitions = [
    {
        'label': 'Breakout',
        'cls': BreakoutAgent,
        'params': {'lookback': 15},
        'sl_mult': 1.5,
        'tp_mult': 3.5,
        'priority': 3,
    },
    {
        'label': 'Donchian Channel',
        'cls': DonchianChannelAgent,
        'params': {'channel_length': 15},
        'sl_mult': 1.5,
        'tp_mult': 3.5,
        'priority': 4,
    },
    {
        'label': 'MA Crossover',
        'cls': MACrossoverAgent,
        'params': {'fast_period': 5, 'slow_period': 30},
        'sl_mult': 2.0,
        'tp_mult': 2.0,
        'priority': 1,
    },
    {
        'label': 'Momentum Trend',
        'cls': MomentumTrendAgent,
        'params': {'ma_period': 30, 'roc_period': 5},
        'sl_mult': 3.0,
        'tp_mult': 4.0,
        'priority': 5,
    },
    {
        'label': 'Mean Reversion',
        'cls': MeanReversionAgent,
        'params': {'ma_period': 10, 'num_std': 1.0},
        'sl_mult': 2.25,
        'tp_mult': 4.0,
        'priority': 2,
    },
]

# For each symbol, create a dict of agents
agent_dict = {
    symbol: [
        {
            'label': definition['label'],
            'agent': definition['cls'](**definition['params']),
            'sl_mult': definition.get('sl_mult', SL_ATR_MULTIPLIER),
            'tp_mult': definition.get('tp_mult', TP_ATR_MULTIPLIER),
            'priority': definition.get('priority', 999),
        }
        for definition in agent_definitions
    ]
    for symbol in symbols
}

def ensure_connection_ready() -> bool:
    """Verify the terminal connection and attempt a reinitialize if needed."""
    terminal_info = mt5.terminal_info()
    if terminal_info is None or not getattr(terminal_info, 'connected', False):
        print("MT5 terminal not connected. Attempting to reinitialize...")
        if not mt5.initialize():
            code, message = mt5.last_error()
            print(f"Reinitialize failed: {code} | {message}")
            return False
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not getattr(terminal_info, 'connected', False):
            print("Unable to establish MT5 connection after reinitialize attempt.")
            return False
    return True


def prepare_symbol(symbol: str):
    """Ensure a symbol is visible/tradable and return its updated info."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not getattr(symbol_info, 'visible', True):
        if not mt5.symbol_select(symbol, True):
            code, message = mt5.last_error()
            print(f"Unable to select {symbol}: {code} | {message}")
            return None
        symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        print(f"Symbol info still unavailable for {symbol} after selection.")
        return None

    trade_mode = getattr(symbol_info, 'trade_mode', None)
    if trade_mode in (mt5.SYMBOL_TRADE_MODE_DISABLED, mt5.SYMBOL_TRADE_MODE_CLOSEONLY):
        print(f"Trading disabled for {symbol} (trade_mode={trade_mode}).")
        return None

    return symbol_info


def normalize_volume(volume: float, symbol_info) -> float:
    """Clamp and snap the requested volume to the broker's allowed step."""
    step = getattr(symbol_info, 'volume_step', 0.01) or 0.01
    min_volume = getattr(symbol_info, 'volume_min', step) or step
    max_volume = getattr(symbol_info, 'volume_max', volume) or volume
    normalized = max(min_volume, min(volume, max_volume))
    steps = round(normalized / step)
    normalized = steps * step
    decimals = len(str(step).split('.')[-1]) if '.' in str(step) else 0
    return round(normalized, decimals)


def resolve_filling_type(symbol_info) -> int:
    """Select a filling mode supported by the symbol, falling back safely."""
    default = getattr(mt5, 'ORDER_FILLING_IOC', 0)
    filling_mask = getattr(symbol_info, 'trade_fillings', 0) or 0
    options = []

    if filling_mask & 0x02 and hasattr(mt5, 'ORDER_FILLING_IOC'):
        options.append(mt5.ORDER_FILLING_IOC)
    if filling_mask & 0x01 and hasattr(mt5, 'ORDER_FILLING_FOK'):
        options.append(mt5.ORDER_FILLING_FOK)
    if filling_mask & 0x04 and hasattr(mt5, 'ORDER_FILLING_RETURN'):
        options.append(mt5.ORDER_FILLING_RETURN)

    if options:
        return options[0]

    mode = getattr(symbol_info, 'trade_filling_mode', None)
    if mode is not None:
        mapping = {
            0: getattr(mt5, 'ORDER_FILLING_FOK', default),
            1: getattr(mt5, 'ORDER_FILLING_IOC', default),
            2: getattr(mt5, 'ORDER_FILLING_RETURN', default),
        }
        return mapping.get(mode, default)

    return default


def sanitize_comment(comment: str | None, fallback: str = "Ultima") -> str:
    """Ensure order comments meet MT5 constraints (ASCII, <=31 chars)."""
    base = comment if comment else fallback
    ascii_comment = base.encode('ascii', errors='ignore').decode('ascii')
    ascii_comment = re.sub(r'[^A-Za-z0-9]+', '-', ascii_comment).strip('-')
    if not ascii_comment:
        ascii_comment = fallback
    max_len = 31
    if len(ascii_comment) > max_len:
        ascii_comment = ascii_comment[:max_len]
    ascii_comment = ascii_comment.rstrip('-')
    if not ascii_comment:
        ascii_comment = fallback[:max_len]
    return ascii_comment


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD):
    """Compute ATR on completed bars and return the last ATR value (float) or None if insufficient data."""
    try:
        if df is None or len(df) < period + 2:
            return None
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(period).mean()
        atr_val = atr_series.iloc[-2]  # use last completed bar
        if pd.isna(atr_val):
            return None
        return float(atr_val)
    except Exception:
        return None


def send_order(
    symbol,
    signal,
    lot=None,
    deviation=5,
    comment="",
    atr: float | None = None,
    sl_mult: float | None = None,
    tp_mult: float | None = None,
):
    if not ensure_connection_ready():
        print("Aborting send_order because MT5 connection is not ready.")
        return

    # Fetch symbol details and tick once
    symbol_info = prepare_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        print(f"Cannot send order for {symbol}: symbol_info or tick is None (symbol_info={symbol_info}, tick={tick})")
        return

    # Set lot size based on symbol
    if lot is None:
        if symbol.upper() == 'XAUUSD+':
            lot = 0.05
        elif symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            lot = 1  # For stocks, usually 1
        else:
            lot = 0.25
    lot = normalize_volume(lot, symbol_info)
    price = tick.ask if signal == 'buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL

    # Compute SL/TP from ATR if available
    sl = None
    tp = None
    effective_sl_mult = sl_mult if sl_mult is not None else SL_ATR_MULTIPLIER
    effective_tp_mult = tp_mult if tp_mult is not None else TP_ATR_MULTIPLIER
    if atr is not None and atr > 0:
        sl_dist = effective_sl_mult * atr
        tp_dist = effective_tp_mult * atr
        if signal == 'buy':
            sl = price - sl_dist
            tp = price + tp_dist
        else:  # sell
            sl = price + sl_dist
            tp = price - tp_dist

        # Respect minimal stop distance in points if requested
        if RESPECT_STOPS_LEVEL and hasattr(symbol_info, 'trade_stops_level') and symbol_info.trade_stops_level:
            point = symbol_info.point
            min_dist = symbol_info.trade_stops_level * point
            # Adjust SL
            if sl is not None and abs(price - sl) < min_dist:
                if signal == 'buy':
                    sl = price - min_dist
                else:
                    sl = price + min_dist
            # Adjust TP
            if tp is not None and abs(price - tp) < min_dist:
                if signal == 'buy':
                    tp = price + min_dist
                else:
                    tp = price - min_dist

        # Round to symbol digits
        digits = getattr(symbol_info, 'digits', 5)
        sl = round(sl, digits) if sl is not None else None
        tp = round(tp, digits) if tp is not None else None

    if atr is None:
        print(f"ATR unavailable for {symbol}; sending order without SL/TP.")
    else:
        print(f"{symbol} ATR={atr:.6f} | SL={sl} | TP={tp} (SLx={effective_sl_mult}, TPx={effective_tp_mult})")

    filling_type = resolve_filling_type(symbol_info)

    sanitized_comment = sanitize_comment(comment)
    if sanitized_comment != (comment or ""):
        print(f"Adjusted order comment to '{sanitized_comment}'")

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': lot,
        'type': order_type,
        'price': price,
        'deviation': deviation,
        'magic': 123456,
        'comment': sanitized_comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': filling_type,
    }
    # Attach SL/TP if computed
    if sl is not None:
        request['sl'] = sl
    if tp is not None:
        request['tp'] = tp
    result = mt5.order_send(request)
    if result is None:
        code, message = mt5.last_error()
        print(f"OrderSend returned None for {symbol}. Last error: {code} | {message}")
        print(f"Request: {request}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"OrderSend failed for {symbol}: {result.retcode}")
        print(f"Reason: {result.comment}")
    else:
        print(f"OrderSend success for {symbol}: {signal} at {price} | SL={request.get('sl')} TP={request.get('tp')}")

try:
    while True:
        if not ensure_connection_ready():
            print("Waiting before retrying MT5 connection check...")
            time.sleep(5)
            continue
        print(f"\nChecking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for symbol in symbols:
            symbol_info = prepare_symbol(symbol)
            if symbol_info is None:
                print(f"Skipping {symbol} because symbol preparation failed.")
                continue
            today = datetime.now()
            from_date = today - timedelta(minutes=bars*15)
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, today)
            if rates is None:
                print(f"No data for {symbol}")
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Compute ATR once per symbol for this cycle
            atr_value = compute_atr(df, ATR_PERIOD)
            if atr_value is not None:
                print(f"{symbol} computed ATR({ATR_PERIOD}) = {atr_value:.6f}")
            else:
                print(f"{symbol} ATR could not be computed (insufficient data).")
            positions = mt5.positions_get(symbol=symbol) or []
            have_buy = any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
            have_sell = any(p.type == mt5.POSITION_TYPE_SELL for p in positions)

            all_candidates = []

            for agent_info in agent_dict[symbol]:
                agent = agent_info['agent']
                strategy_name = agent_info.get('label') or type(agent).__name__.replace('Agent', '').replace('_', ' ').strip() or 'Strategy'
                sl_mult = agent_info.get('sl_mult', SL_ATR_MULTIPLIER)
                tp_mult = agent_info.get('tp_mult', TP_ATR_MULTIPLIER)
                priority = agent_info.get('priority', 999)

                signal = agent.get_signal(df)
                print(f"{symbol} {strategy_name} Signal: {signal}")

                if signal in ('buy', 'sell'):
                    all_candidates.append({
                        'signal': signal,
                        'label': strategy_name,
                        'priority': priority,
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult,
                    })

            if all_candidates:
                # Pick the absolute highest priority signal regardless of direction
                best_candidate = min(all_candidates, key=lambda c: c['priority'])
                signal_type = best_candidate['signal']
                
                if signal_type == 'buy':
                    if have_sell and not ALLOW_HEDGING:
                        print(f"{symbol}: skipping {best_candidate['label']} buy (priority {best_candidate['priority']}) because opposite position is open")
                    elif have_buy:
                        print(f"{symbol}: already in buy position, ignoring {best_candidate['label']} (priority {best_candidate['priority']})")
                    else:
                        print(f"{symbol}: executing BUY via {best_candidate['label']} (priority {best_candidate['priority']})")
                        send_order(
                            symbol,
                            'buy',
                            comment=f"{best_candidate['label']} P{best_candidate['priority']}",
                            atr=atr_value,
                            sl_mult=best_candidate['sl_mult'],
                            tp_mult=best_candidate['tp_mult'],
                        )
                        have_buy = True
                elif signal_type == 'sell':
                    if have_buy and not ALLOW_HEDGING:
                        print(f"{symbol}: skipping {best_candidate['label']} sell (priority {best_candidate['priority']}) because opposite position is open")
                    elif have_sell:
                        print(f"{symbol}: already in sell position, ignoring {best_candidate['label']} (priority {best_candidate['priority']})")
                    else:
                        print(f"{symbol}: executing SELL via {best_candidate['label']} (priority {best_candidate['priority']})")
                        send_order(
                            symbol,
                            'sell',
                            comment=f"{best_candidate['label']} P{best_candidate['priority']}",
                            atr=atr_value,
                            sl_mult=best_candidate['sl_mult'],
                            tp_mult=best_candidate['tp_mult'],
                        )
                        have_sell = True
                
                # Now skip any remaining lower-priority signals
                remaining_candidates = [c for c in all_candidates if c != best_candidate]
                for candidate in remaining_candidates:
                    print(f"{symbol}: skipping {candidate['label']} {candidate['signal']} (priority {candidate['priority']}) - lower priority than executed {best_candidate['label']} (priority {best_candidate['priority']})")
        time.sleep(300)  # Wait 5 minutes
finally:
    mt5.shutdown()
