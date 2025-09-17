import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from agents import MeanReversionAgent, MACrossoverAgent, MomentumTrendAgent, BreakoutAgent, DonchianChannelAgent
import time

# Settings
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "XAUUSD", "USDJPY"]
timeframe = mt5.TIMEFRAME_M15
bars = 100  # Number of bars to fetch

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

def send_order(symbol, signal, lot=None, deviation=5, comment=""):
    # Set lot size based on symbol
    if lot is None:
        if symbol.upper() == 'XAUUSD':
            lot = 0.01
        elif symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            lot = 1  # For stocks, usually 1
        else:
            lot = 0.1
    price = mt5.symbol_info_tick(symbol).ask if signal == 'buy' else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
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
    result = mt5.order_send(request)
    if result is None:
        print(f"OrderSend returned None for {symbol}. Check connection, symbol, and request parameters.")
        print(f"Request: {request}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"OrderSend failed for {symbol}: {result.retcode}")
        print(f"Reason: {result.comment}")
    else:
        print(f"OrderSend success for {symbol}: {signal} at {price}")

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
                        send_order(symbol, 'buy', comment=f"{strategy_name} signal")
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
                        send_order(symbol, 'sell', comment=f"{strategy_name} signal")
        time.sleep(300)  # Wait 5 minutes
finally:
    mt5.shutdown()
