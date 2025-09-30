import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from agents import MeanReversionAgent, MACrossoverAgent, MomentumTrendAgent, BreakoutAgent, DonchianChannelAgent
import time

# Settings
symbols = ["USDJPY", "XAUUSD"]
timeframe = mt5.TIMEFRAME_M15
bars = 100  # Number of bars to fetch

# Risk/target settings (ATR-based)
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 3.0
RESPECT_STOPS_LEVEL = True  # Enforce broker minimal stop distance if provided

# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

# Agent settings for each strategy
agent_classes = [
    (MACrossoverAgent, {'fast_period': 5, 'slow_period': 20}),
    (MeanReversionAgent, {'ma_period': 30, 'num_std': 3}),
    (MomentumTrendAgent, {'ma_period': 50, 'roc_period': 5}),
    (BreakoutAgent, {'lookback': 10}),
    (DonchianChannelAgent, {'channel_length': 10}),
]

# For each symbol, create a dict of agents
agent_dict = {
    symbol: [cls(**params) for cls, params in agent_classes]
    for symbol in symbols
}

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


def send_order(symbol, signal, lot=None, deviation=5, comment="", atr: float | None = None):
    # Fetch symbol details and tick once
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        print(f"Cannot send order for {symbol}: symbol_info or tick is None (symbol_info={symbol_info}, tick={tick})")
        return

    # Set lot size based on symbol
    if lot is None:
        if symbol.upper() == 'XAUUSD':
            lot = 0.01
        elif symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            lot = 1  # For stocks, usually 1
        else:
            lot = 0.1
    price = tick.ask if signal == 'buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL

    # Compute SL/TP from ATR if available
    sl = None
    tp = None
    if atr is not None and atr > 0:
        sl_dist = SL_ATR_MULTIPLIER * atr
        tp_dist = TP_ATR_MULTIPLIER * atr
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
        print(f"{symbol} ATR={atr:.6f} | SL={sl} | TP={tp}")

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': lot,
        'type': order_type,
        'price': price,
        'deviation': deviation,
        'magic': 123456,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,  # Use Fill or Kill
    }
    # Attach SL/TP if computed
    if sl is not None:
        request['sl'] = sl
    if tp is not None:
        request['tp'] = tp
    result = mt5.order_send(request)
    if result is None:
        print(f"OrderSend returned None for {symbol}. Check connection, symbol, and request parameters.")
        print(f"Request: {request}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"OrderSend failed for {symbol}: {result.retcode}")
        print(f"Reason: {result.comment}")
    else:
        print(f"OrderSend success for {symbol}: {signal} at {price} | SL={request.get('sl')} TP={request.get('tp')}")

try:
    while True:
        print(f"\nChecking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for symbol in symbols:
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
            for agent in agent_dict[symbol]:
                strategy_name = type(agent).__name__.replace('Agent', '').replace('_', ' ').strip() or 'Strategy'
                signal = agent.get_signal(df)
                print(f"{symbol} {strategy_name} Signal: {signal}")
                positions = mt5.positions_get(symbol=symbol)
                have_buy = positions and any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
                have_sell = positions and any(p.type == mt5.POSITION_TYPE_SELL for p in positions)
                # Close opposite position if needed
                if signal == 'buy':
                    # Close sell positions before opening buy
                    if have_sell:
                        for p in positions:
                            if p.type == mt5.POSITION_TYPE_SELL:
                                close_request = {
                                    'action': mt5.TRADE_ACTION_DEAL,
                                    'symbol': symbol,
                                    'volume': p.volume,
                                    'type': mt5.ORDER_TYPE_BUY,
                                    'position': p.ticket,
                                    'price': mt5.symbol_info_tick(symbol).ask,
                                    'deviation': 5,
                                    'magic': 123456,
                                    'comment': f'close sell by {strategy_name} agent',
                                    'type_time': mt5.ORDER_TIME_GTC,
                                    'type_filling': mt5.ORDER_FILLING_IOC,
                                }
                                mt5.order_send(close_request)
                                print(f"Closed sell position for {symbol}")
                    if not have_buy:
                        send_order(symbol, 'buy', comment=f"{strategy_name} signal", atr=atr_value)
                elif signal == 'sell':
                    # Close buy positions before opening sell
                    if have_buy:
                        for p in positions:
                            if p.type == mt5.POSITION_TYPE_BUY:
                                close_request = {
                                    'action': mt5.TRADE_ACTION_DEAL,
                                    'symbol': symbol,
                                    'volume': p.volume,
                                    'type': mt5.ORDER_TYPE_SELL,
                                    'position': p.ticket,
                                    'price': mt5.symbol_info_tick(symbol).bid,
                                    'deviation': 5,
                                    'magic': 123456,
                                    'comment': f'close buy by {strategy_name} agent',
                                    'type_time': mt5.ORDER_TIME_GTC,
                                    'type_filling': mt5.ORDER_FILLING_IOC,
                                }
                                mt5.order_send(close_request)
                                print(f"Closed buy position for {symbol}")
                    if not have_sell:
                        send_order(symbol, 'sell', comment=f"{strategy_name} signal", atr=atr_value)
        time.sleep(300)  # Wait 5 minutes
finally:
    mt5.shutdown()
