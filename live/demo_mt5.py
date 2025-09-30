import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
from agents import MeanReversionAgent, MACrossoverAgent, MomentumTrendAgent, BreakoutAgent, DonchianChannelAgent
import time
import re
import json
from collections import defaultdict

# Strategy comment registry for MT5-safe order comments
STRATEGY_COMMENT_REGISTRY = {
    "momentum trend": ("MT", "momentum_trend"),
    "ma crossover": ("MA", "ma_crossover"),
    "mean reversion": ("MR", "mean_reversion"),
    "breakout": ("BO", "breakout"),
    "donchian channel": ("DC", "donchian_channel"),
}

COMMENT_CODE_TO_PERF_KEY = {code: perf_key for code, perf_key in STRATEGY_COMMENT_REGISTRY.values()}
COMMENT_CODE_CACHE = dict(COMMENT_CODE_TO_PERF_KEY)


def _generate_comment_code(label: str) -> str:
    initials = ''.join(word[0].upper() for word in re.split(r"\s+", label) if word)
    if not initials:
        return "ST"
    return initials[:3]


def build_order_comment(strategy_label: str, signal: str, priority: float) -> str:
    """Create a compact MT5-safe comment encoding strategy, side, and priority."""
    label_key = strategy_label.strip().lower()
    code = STRATEGY_COMMENT_REGISTRY.get(label_key, (None, None))[0]
    perf_key = STRATEGY_COMMENT_REGISTRY.get(label_key, (None, None))[1]
    if code is None:
        code = _generate_comment_code(strategy_label)
        perf_key = label_key.replace(' ', '_') if label_key else "strategy"
    COMMENT_CODE_CACHE[code] = perf_key

    side = 'B' if signal.lower() == 'buy' else 'S'
    priority_tag = max(0, min(99, int(round(priority))))
    return f"{code}{side}{priority_tag:02d}"

# Strategy Performance Tracking
PERFORMANCE_FILE = "strategy_performance.json"
strategy_performance = defaultdict(lambda: {
    'total_trades': 0,
    'winning_trades': 0,
    'total_pnl': 0.0,
    'win_rate': 0.0,
    'avg_win': 0.0,
    'avg_loss': 0.0,
    'profit_factor': 0.0,
    'last_updated': None
})

# Performance tracking settings
PERFORMANCE_LOOKBACK_DAYS = 30  # Track performance over last 30 days
MIN_TRADES_FOR_ADAPTATION = 10  # Minimum trades before adjusting weights

def load_strategy_performance():
    """Load strategy performance from file."""
    global strategy_performance
    try:
        with open(PERFORMANCE_FILE, 'r') as f:
            data = json.load(f)
            strategy_performance.update(data)
        print(f"📊 Loaded strategy performance data from {PERFORMANCE_FILE}")
    except FileNotFoundError:
        print(f"📊 No existing performance file found, starting fresh tracking")
    except Exception as e:
        print(f"Error loading performance data: {e}")

def save_strategy_performance():
    """Save strategy performance to file."""
    try:
        with open(PERFORMANCE_FILE, 'w') as f:
            json.dump(dict(strategy_performance), f, indent=2)
    except Exception as e:
        print(f"Error saving performance data: {e}")


def update_strategy_performance(strategy_name: str, symbol: str, pnl: float):
    """Update performance metrics for a strategy."""
    key = f"{strategy_name}_{symbol}"
    perf = strategy_performance[key]
    
    perf['total_trades'] += 1
    perf['total_pnl'] += pnl
    perf['last_updated'] = datetime.now().isoformat()
    
    if pnl > 0:
        perf['winning_trades'] += 1
    
    # Calculate metrics
    if perf['total_trades'] > 0:
        perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
    
    # For detailed win/loss analysis, we'd need to track individual trades
    # For now, use simplified profit factor approximation
    if perf['total_trades'] >= MIN_TRADES_FOR_ADAPTATION:
        perf['profit_factor'] = max(0.1, perf['total_pnl'] / max(1.0, perf['total_trades']))
    
    save_strategy_performance()
    print(f"📈 Updated performance for {strategy_name} on {symbol}: PnL={pnl:.2f}, WR={perf['win_rate']:.1%}, Trades={perf['total_trades']}")

def monitor_closed_positions():
    """Monitor recently closed positions and update strategy performance."""
    try:
        # Get deals from last 24 hours
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now()
        
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return
        
        # Track processed deals to avoid double counting
        processed_deals_file = "processed_deals.txt"
        try:
            with open(processed_deals_file, 'r') as f:
                processed_deals = set(line.strip() for line in f)
        except FileNotFoundError:
            processed_deals = set()
        
        new_processed = []
        
        for deal in deals:
            deal_id = str(deal.ticket)
            if deal_id in processed_deals:
                continue
            
            # Only process exit deals (position close)
            if deal.entry != 1:  # 1 = entry, 0 = exit
                continue
            
            symbol = deal.symbol
            profit = deal.profit
            comment = (deal.comment or "").strip()
            performance_key = None

            # New compact comment format: CODE + side + priority (e.g., MTB16)
            match = re.match(r"^([A-Z]{1,3})([BS])(\d{2})", comment)
            if match:
                code = match.group(1)
                performance_key = COMMENT_CODE_CACHE.get(code)
            elif " P" in comment:
                # Legacy format: "Strategy Name P123"
                strategy_name = comment.split(" P")[0].replace('-', ' ').strip()
                performance_key = strategy_name.lower().replace(' ', '_')

            if performance_key and symbol:
                update_strategy_performance(performance_key, symbol, profit)
                new_processed.append(deal_id)
        
        # Save processed deals
        if new_processed:
            with open(processed_deals_file, 'a') as f:
                for deal_id in new_processed:
                    f.write(f"{deal_id}\n")
                    
    except Exception as e:
        print(f"Error monitoring closed positions: {e}")


def get_strategy_performance_weight(strategy_name: str, symbol: str) -> float:
    """Get performance-based weight multiplier for strategy."""
    key = f"{strategy_name}_{symbol}"
    perf = strategy_performance[key]
    
    if perf['total_trades'] < MIN_TRADES_FOR_ADAPTATION:
        return 1.0  # Use default weight until enough data
    
    # Base weight on profit factor and win rate
    profit_factor = perf.get('profit_factor', 0.0)
    win_rate = perf.get('win_rate', 0.0)
    
    # Conservative adaptation: good performance gets slight boost, poor performance gets reduced
    if profit_factor > 0.1 and win_rate > 0.4:
        return min(1.5, 1.0 + (profit_factor * 0.5) + (win_rate * 0.3))  # Max 1.5x boost
    elif profit_factor < -0.05 or win_rate < 0.3:
        return max(0.3, 1.0 + profit_factor)  # Reduce to min 0.3x
    else:
        return 1.0  # Neutral performance

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

# Enhanced risk management
ACCOUNT_RISK_PER_TRADE = 0.30  # Mirror aggressive backtest profile for live trading
MIN_LOT_SIZE = 0.01  # Minimum position size
MAX_LOT_SIZE = 5.0   # Maximum position size cap

# Dynamic risk multiplier bounds
RISK_MULTIPLIER_MIN = 1.0
RISK_MULTIPLIER_MAX = 3.5

# Trade quality filters
SPREAD_POINTS_LIMIT = 10            # Maximum raw spread in points before skipping
SPREAD_ATR_RATIO_LIMIT = 0.45       # Spread must remain under 45% of current ATR

# Micro timeframe momentum confirmation (sniper entry enhancer)
ENABLE_MICRO_MOMENTUM_CONFIRMATION = True
MICRO_MOMENTUM_TIMEFRAME = mt5.TIMEFRAME_M5
MICRO_MOMENTUM_LOOKBACK = 6
MICRO_MOMENTUM_BASE_THRESHOLD = 0.0006
MICRO_MOMENTUM_MIN_THRESHOLD = 0.0002
MICRO_MOMENTUM_MAX_THRESHOLD = 0.0025
MICRO_MOMENTUM_DYNAMIC_MULTIPLIER = 0.55
MICRO_MOMENTUM_SOFT_PASS_RATIO = 0.45


def get_account_balance() -> float:
    """Get the current account balance from MT5."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info, using fallback balance of $1000")
            return 1000.0
        
        equity = getattr(account_info, 'equity', None)
        balance = float(account_info.balance)
        effective_balance = float(equity) if equity is not None else balance
        print(f"Current account equity: ${effective_balance:.2f} (balance: ${balance:.2f})")
        return effective_balance
    except Exception as e:
        print(f"Error getting account balance: {e}, using fallback balance of $1000")
        return 1000.0


def is_market_session_active() -> bool:
    """Check if we're in an active trading session (avoid low-liquidity periods)."""
    try:
        # Get current UTC time
        now_utc = datetime.now(pytz.UTC)
        current_hour = now_utc.hour
        current_weekday = now_utc.weekday()  # 0=Monday, 6=Sunday
        
        # Avoid weekends (Friday 22:00 UTC to Sunday 22:00 UTC)
        if current_weekday == 6:  # Sunday
            return current_hour >= 22  # Only after 22:00 UTC (Sydney open)
        elif current_weekday == 5:  # Saturday
            return False  # No trading on Saturday
        elif current_weekday == 4 and current_hour >= 22:  # Friday after 22:00 UTC
            return False  # Weekend starts
        
        # Active trading sessions (UTC times):
        # Sydney: 22:00-07:00 UTC
        # Tokyo: 00:00-09:00 UTC  
        # London: 08:00-17:00 UTC
        # New York: 13:00-22:00 UTC
        
        # Avoid dead zones: 07:00-08:00 UTC (between Sydney/Tokyo and London)
        if 7 <= current_hour < 8:
            return False
        
        # Most active periods (overlaps):
        # London-New York overlap: 13:00-17:00 UTC (Best)
        # Tokyo-London overlap: 08:00-09:00 UTC (Good)
        # Always active: 22:00-07:00, 08:00-22:00 (avoid only 07:00-08:00)
        
        return True
        
    except Exception as e:
        print(f"Error checking market session: {e}, defaulting to allow trading")
        return True


def get_session_info() -> str:
    """Get current trading session name for logging."""
    try:
        now_utc = datetime.now(pytz.UTC)
        hour = now_utc.hour
        
        if 22 <= hour or hour < 7:
            return "Sydney/Tokyo"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 17:
            return "London-NY Overlap"
        elif 17 <= hour < 22:
            return "New York"
        else:
            return "Dead Zone"
    except Exception:
        return "Unknown"


# Instrument-specific session priorities (based on underlying market activity)
INSTRUMENT_SESSION_PRIORITY = {
    "EURUSD+": {
        "London": 5,           # Best: EUR/USD most active during European session
        "London-NY Overlap": 5, # Best: Highest liquidity period for majors
        "New York": 4,         # Good: Still active during US session
        "Sydney/Tokyo": 2,     # Poor: Low EUR activity during Asian session
    },
    "GBPUSD+": {
        "London": 5,           # Best: GBP home session, highest volatility
        "London-NY Overlap": 5, # Best: Peak liquidity for GBP/USD
        "New York": 4,         # Good: Continued activity in US session
        "Sydney/Tokyo": 1,     # Very poor: GBP barely trades in Asia
    },
    "USDJPY+": {
        "Sydney/Tokyo": 5,     # Best: JPY home session, highest volatility
        "London": 4,           # Good: European traders active
        "London-NY Overlap": 4, # Good: Still decent liquidity
        "New York": 3,         # Fair: End of USD day, lower JPY activity
    },
    "GBPJPY+": {
        "Sydney/Tokyo": 4,     # Good: JPY side active
        "London": 5,           # Best: GBP side most active + some JPY overlap
        "London-NY Overlap": 4, # Good: High volatility cross
        "New York": 2,         # Poor: Both sides winding down
    },
    "XAUUSD+": {
        "London": 5,           # Best: Major gold trading center (London)
        "London-NY Overlap": 5, # Best: Peak institutional activity
        "New York": 4,         # Good: US market participation
        "Sydney/Tokyo": 3,     # Fair: Asian demand, but lower liquidity
    }
}

# Minimum session priority to allow trading (1-5 scale)
MIN_SESSION_PRIORITY = 3  # Only trade when session priority >= 3

# Correlation management - prevent over-concentration in similar moves
CORRELATION_GROUPS = {
    "EUR_PAIRS": ["EURUSD+"],                    # EUR strength/weakness
    "GBP_PAIRS": ["GBPUSD+", "GBPJPY+"],        # GBP strength/weakness  
    "JPY_PAIRS": ["USDJPY+", "GBPJPY+"],        # JPY strength/weakness
    "USD_MAJORS": ["EURUSD+", "GBPUSD+", "USDJPY+"], # USD strength/weakness
    "SAFE_HAVEN": ["USDJPY+", "XAUUSD+"],       # Risk-off correlation
}

MAX_CORRELATED_POSITIONS = 2  # Max positions in same correlation group
CORRELATION_POSITION_LIMIT = 0.6  # Reduce position size if multiple correlated trades

# Market Regime Detection - Dynamic strategy adaptation
REGIME_LOOKBACK_PERIODS = 50  # Bars to analyze for regime detection
TREND_THRESHOLD = 0.3         # ADX threshold for trending markets
VOLATILITY_THRESHOLD = 1.5    # ATR multiplier for volatile markets

# Strategy weights by market regime (sum should = 1.0 for each regime)
REGIME_STRATEGY_WEIGHTS = {
    'TRENDING': {
        'ma_crossover': 0.35,      # Best in trends
        'momentum_trend': 0.30,    # Excellent trend following
        'breakout': 0.25,          # Good for trend continuation
        'mean_reversion': 0.05,    # Poor in strong trends
        'donchian_channel': 0.05,  # Poor in strong trends
    },
    'RANGING': {
        'mean_reversion': 0.40,    # Excellent in sideways markets
        'donchian_channel': 0.25,  # Good for range trading
        'ma_crossover': 0.15,      # Moderate in ranges
        'momentum_trend': 0.10,    # Poor in ranges
        'breakout': 0.10,          # Poor in ranges (many false breakouts)
    },
    'VOLATILE': {
        'breakout': 0.35,          # Good for explosive moves
        'momentum_trend': 0.25,    # Good for volatility expansion
        'ma_crossover': 0.20,      # Moderate in volatile conditions
        'donchian_channel': 0.15,  # Moderate adaptability
                'mean_reversion': 0.05,    # Poor in high volatility
    }
}

# News/Economic Event Filtering - avoid trading around major announcements
NEWS_BLACKOUT_TIMES = {
    # Major USD events (all times in UTC)
    'USD_MAJOR': [
        (8, 30),   # 8:30 UTC - US Economic data (NFP, CPI, etc.)
        (14, 0),   # 2:00 PM UTC - FOMC meetings  
        (18, 0),   # 6:00 PM UTC - Fed speeches
    ],
    'EUR_MAJOR': [
        (7, 0),    # 7:00 UTC - ECB interest rate decisions
        (8, 0),    # 8:00 UTC - EU inflation data
        (9, 0),    # 9:00 UTC - German/EU economic data
    ],
    'GBP_MAJOR': [
        (6, 0),    # 6:00 UTC - BOE interest rate decisions
        (8, 30),   # 8:30 UTC - UK employment/inflation data
        (9, 30),   # 9:30 UTC - UK GDP data
    ],
    'JPY_MAJOR': [
        (23, 50),  # 11:50 PM UTC - BOJ interest rate decisions  
        (0, 30),   # 12:30 AM UTC - Japanese economic data
    ]
}

# News/Economic Event Filtering - DISABLED (too restrictive for active trading)
ENABLE_NEWS_FILTERING = False  # Disable news filtering to allow more trading opportunities
NEWS_BLACKOUT_MINUTES = 30  # Minutes before/after news to avoid trading
NEWS_BOOST_MINUTES = 90     # Minutes after major news for increased activity

# Multi-timeframe confirmation settings
ENABLE_MTF_CONFIRMATION = True  # Enable multi-timeframe signal confirmation
MTF_TIMEFRAMES = [
    mt5.TIMEFRAME_M15,  # Primary timeframe (current)
    mt5.TIMEFRAME_H1,   # Higher timeframe for trend confirmation
]
MTF_LOOKBACK_BARS = 20  # Bars to fetch for higher timeframe analysis

# Advanced Stop Management - HIGH IMPACT ON RETURNS
ENABLE_ADVANCED_STOPS = True    # Enable advanced stop management
BREAKEVEN_TRIGGER = 1.0         # Move SL to breakeven after 1x ATR profit
TRAILING_START = 1.5            # Start trailing after 1.5x ATR profit  
TRAILING_DISTANCE = 1.0         # Trail SL 1x ATR behind price
PARTIAL_TAKE_PROFIT = 0.5       # Take 50% profit at 1:1 risk/reward
PARTIAL_TP_RATIO = 1.0          # Take partial profit at 1x risk


def get_symbol_currency_exposure(symbol: str) -> list:
    """Get list of currencies this symbol is exposed to for news filtering."""
    if symbol == "EURUSD+":
        return ["EUR", "USD"]
    elif symbol == "GBPUSD+":
        return ["GBP", "USD"]
    elif symbol == "USDJPY+":
        return ["USD", "JPY"]
    elif symbol == "GBPJPY+":
        return ["GBP", "JPY"]
    elif symbol == "XAUUSD+":
        return ["USD"]  # Gold primarily affected by USD
    else:
        return ["USD"]  # Default to USD exposure


def is_news_blackout_period(symbol: str) -> tuple[bool, str]:
    """Check if current time is within news blackout period for symbol's currencies."""
    # Skip news filtering if disabled
    if not ENABLE_NEWS_FILTERING:
        return False, "News filtering disabled"
    
    try:
        now_utc = datetime.now(pytz.UTC)
        current_hour = now_utc.hour
        current_minute = now_utc.minute
        current_time = current_hour * 60 + current_minute  # Convert to minutes
        
        print(f"🕐 Current UTC time: {current_hour:02d}:{current_minute:02d} (checking {symbol})")
        
        currencies = get_symbol_currency_exposure(symbol)
        
        for currency in currencies:
            news_times = NEWS_BLACKOUT_TIMES.get(f"{currency}_MAJOR", [])
            
            for news_hour, news_minute in news_times:
                news_time = news_hour * 60 + news_minute
                
                # Calculate time difference (handle same day only for now)
                time_diff = abs(current_time - news_time)
                
                print(f"  📰 {currency} news at {news_hour:02d}:{news_minute:02d} - time diff: {time_diff} min (limit: {NEWS_BLACKOUT_MINUTES})")
                
                # Only consider blackout if we're within the same day and close to news time
                if time_diff <= NEWS_BLACKOUT_MINUTES:
                    return True, f"{currency} news at {news_hour:02d}:{news_minute:02d}"
        
        return False, ""
    except Exception as e:
        print(f"Error checking news blackout: {e}")
        return False, ""


def manage_position_stops(symbol: str, atr_value: float):
    """Advanced stop management: breakeven, trailing, partial profits."""
    if not ENABLE_ADVANCED_STOPS or not atr_value:
        return
    
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        
        for position in positions:
            ticket = position.ticket
            position_type = position.type  # 0=buy, 1=sell
            open_price = position.price_open
            current_price = position.price_current
            volume = position.volume
            sl = position.sl
            tp = position.tp
            
            # Calculate current profit in ATR terms
            if position_type == 0:  # Buy position
                profit_distance = current_price - open_price
                is_profitable = profit_distance > 0
            else:  # Sell position  
                profit_distance = open_price - current_price
                is_profitable = profit_distance > 0
            
            if not is_profitable:
                continue  # Skip losing positions
            
            profit_atr = profit_distance / atr_value if atr_value > 0 else 0
            
            # 1. Move to breakeven after 1x ATR profit
            if profit_atr >= BREAKEVEN_TRIGGER and sl != open_price:
                new_sl = open_price
                print(f"💰 {symbol} Moving to breakeven (profit: {profit_atr:.1f}x ATR)")
                modify_position_sl(ticket, new_sl)
            
            # 2. Start trailing after 1.5x ATR profit
            elif profit_atr >= TRAILING_START:
                if position_type == 0:  # Buy position
                    trail_sl = current_price - (TRAILING_DISTANCE * atr_value)
                    if trail_sl > sl:  # Only move SL up
                        print(f"📈 {symbol} Trailing stop: {trail_sl:.5f} (profit: {profit_atr:.1f}x ATR)")
                        modify_position_sl(ticket, trail_sl)
                else:  # Sell position
                    trail_sl = current_price + (TRAILING_DISTANCE * atr_value)
                    if trail_sl < sl:  # Only move SL down (for sell)
                        print(f"📉 {symbol} Trailing stop: {trail_sl:.5f} (profit: {profit_atr:.1f}x ATR)")
                        modify_position_sl(ticket, trail_sl)
            
            # 3. Partial profit taking at 1:1 ratio
            if profit_atr >= PARTIAL_TP_RATIO and volume > 0.01:
                partial_volume = volume * PARTIAL_TAKE_PROFIT
                if partial_volume >= 0.01:  # Minimum trade size
                    print(f"💵 {symbol} Taking partial profit: {partial_volume:.2f} lots at {profit_atr:.1f}x ATR")
                    close_partial_position(ticket, partial_volume)
                    
    except Exception as e:
        print(f"Error managing stops for {symbol}: {e}")


def modify_position_sl(ticket: int, new_sl: float):
    """Modify stop loss for an existing position."""
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": position.tp,  # Keep existing TP
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        else:
            print(f"Failed to modify SL for ticket {ticket}: {result.comment if result else 'No result'}")
            return False
            
    except Exception as e:
        print(f"Error modifying SL for ticket {ticket}: {e}")
        return False


def close_partial_position(ticket: int, volume: float):
    """Close partial position for profit taking."""
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        symbol = position.symbol
        
        # Determine order type (opposite of position type)
        if position.type == 0:  # Close buy position with sell order
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:  # Close sell position with buy order
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "comment": "Partial profit",
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        else:
            print(f"Failed to close partial position for ticket {ticket}: {result.comment if result else 'No result'}")
            return False
            
    except Exception as e:
        print(f"Error closing partial position for ticket {ticket}: {e}")
        return False


def get_mtf_trend_bias(symbol: str) -> str:
    """Get higher timeframe trend bias for signal confirmation."""
    try:
        if not ENABLE_MTF_CONFIRMATION:
            return 'neutral'  # No filtering if MTF disabled
        
        # Get higher timeframe data (H1)
        today = datetime.now()
        from_date = today - timedelta(hours=MTF_LOOKBACK_BARS)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, from_date, today)
        
        if rates is None or len(rates) < 10:
            return 'neutral'  # Insufficient data
        
        df_h1 = pd.DataFrame(rates)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        
        # Simple trend detection using MA
        close = df_h1['close']
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(10).mean()
        
        current_price = close.iloc[-1]
        ma_short_current = ma_short.iloc[-1]
        ma_long_current = ma_long.iloc[-1]
        
        if pd.isna(ma_short_current) or pd.isna(ma_long_current):
            return 'neutral'
        
        # Determine trend bias
        if ma_short_current > ma_long_current and current_price > ma_short_current:
            return 'bullish'
        elif ma_short_current < ma_long_current and current_price < ma_short_current:
            return 'bearish'
        else:
            return 'neutral'
            
    except Exception as e:
        print(f"Error getting MTF bias for {symbol}: {e}")
        return 'neutral'


def confirm_signal_with_mtf(signal: str, symbol: str) -> bool:
    """Confirm M15 signal with higher timeframe bias."""
    if not ENABLE_MTF_CONFIRMATION:
        return True  # No filtering if disabled
    
    mtf_bias = get_mtf_trend_bias(symbol)
    
    # Allow signal if:
    # 1. MTF is neutral (no strong bias)
    # 2. Signal aligns with MTF bias
    # 3. MTF bias detection failed (neutral)
    
    if mtf_bias == 'neutral':
        return True
    elif signal == 'buy' and mtf_bias == 'bullish':
        return True
    elif signal == 'sell' and mtf_bias == 'bearish':
        return True
    else:
        return False


def confirm_with_micro_momentum(
    symbol: str,
    signal: str,
    regime: str,
    atr_value: float | None,
    reference_price: float | None,
) -> tuple[bool, float, float, bool]:
    """Use micro timeframe momentum bias to refine entries."""
    if not ENABLE_MICRO_MOMENTUM_CONFIRMATION:
        return True, 0.0, 0.0, False

    try:
        request_count = MICRO_MOMENTUM_LOOKBACK + 3
        rates = mt5.copy_rates_from_pos(symbol, MICRO_MOMENTUM_TIMEFRAME, 0, request_count)
        if rates is None or len(rates) < MICRO_MOMENTUM_LOOKBACK + 2:
            return True, 0.0, 0.0, False  # Skip if insufficient data

        df = pd.DataFrame(rates)
        close = df['close'].astype(float)
        momentum_series = close.pct_change().rolling(MICRO_MOMENTUM_LOOKBACK).sum()
        momentum_score = momentum_series.iloc[-1]

        if pd.isna(momentum_score):
            return True, 0.0, 0.0, False

        # Derive adaptive threshold based on ATR and market regime
        threshold = MICRO_MOMENTUM_BASE_THRESHOLD
        if atr_value and reference_price:
            atr_pct = max(1e-9, atr_value / reference_price)
            dynamic = atr_pct * MICRO_MOMENTUM_DYNAMIC_MULTIPLIER
            threshold = max(MICRO_MOMENTUM_MIN_THRESHOLD, min(MICRO_MOMENTUM_MAX_THRESHOLD, dynamic))

        if regime == 'TRENDING':
            threshold *= 0.75
        elif regime == 'VOLATILE':
            threshold *= 0.85
        threshold = max(MICRO_MOMENTUM_MIN_THRESHOLD, threshold)

        soft_pass = False

        if signal == 'buy':
            if momentum_score >= threshold:
                return True, float(momentum_score), threshold, soft_pass
            elif momentum_score >= threshold * MICRO_MOMENTUM_SOFT_PASS_RATIO:
                soft_pass = True
                return True, float(momentum_score), threshold, soft_pass
        else:
            if momentum_score <= -threshold:
                return True, float(momentum_score), threshold, soft_pass
            elif momentum_score <= -threshold * MICRO_MOMENTUM_SOFT_PASS_RATIO:
                soft_pass = True
                return True, float(momentum_score), threshold, soft_pass

        return False, float(momentum_score), threshold, soft_pass
    except Exception as e:
        print(f"Error computing micro momentum for {symbol}: {e}")
        return True, 0.0, MICRO_MOMENTUM_MIN_THRESHOLD, False


def get_instrument_session_priority(symbol: str, session: str) -> int:
    """Get the trading priority for a symbol during a specific session."""
    return INSTRUMENT_SESSION_PRIORITY.get(symbol, {}).get(session, 3)


def should_trade_instrument_in_session(symbol: str) -> bool:
    """Check if we should trade this instrument in the current session."""
    try:
        session = get_session_info()
        if session == "Dead Zone":
            return False
        
        priority = get_instrument_session_priority(symbol, session)
        return priority >= MIN_SESSION_PRIORITY
    except Exception:
        return True  # Default to allow if error


def get_correlated_open_positions(symbol: str) -> list:
    """Get list of open positions in symbols correlated with the given symbol."""
    try:
        correlated_positions = []
        
        # Find which correlation groups this symbol belongs to
        symbol_groups = []
        for group_name, group_symbols in CORRELATION_GROUPS.items():
            if symbol in group_symbols:
                symbol_groups.append(group_name)
        
        # Get all open positions
        all_positions = mt5.positions_get() or []
        
        # Check each position to see if it's in a correlated group
        for position in all_positions:
            pos_symbol = position.symbol
            for group_name in symbol_groups:
                if pos_symbol in CORRELATION_GROUPS[group_name] and pos_symbol != symbol:
                    correlated_positions.append({
                        'symbol': pos_symbol,
                        'type': 'buy' if position.type == 0 else 'sell',
                        'volume': position.volume,
                        'group': group_name
                    })
        
        return correlated_positions
    except Exception as e:
        print(f"Error checking correlated positions: {e}")
        return []


def should_limit_correlation_exposure(symbol: str, signal: str) -> tuple[bool, float]:
    """Check if we should limit position size due to correlation exposure.
    
    Returns:
        (should_limit: bool, size_multiplier: float)
    """
    try:
        correlated_positions = get_correlated_open_positions(symbol)
        
        if not correlated_positions:
            return False, 1.0  # No correlation limits
        
        # Count positions in same direction (could amplify moves)
        same_direction_count = 0
        for pos in correlated_positions:
            if pos['type'] == signal:
                same_direction_count += 1
        
        total_correlated = len(correlated_positions)
        
        # Apply limits based on correlation exposure
        if total_correlated >= MAX_CORRELATED_POSITIONS:
            print(f"🔗 {symbol}: Blocking trade - too many correlated positions ({total_correlated}/{MAX_CORRELATED_POSITIONS})")
            return True, 0.0  # Block the trade
        
        elif same_direction_count >= 1:
            print(f"🔗 {symbol}: Reducing position size - {same_direction_count} same-direction correlated positions")
            return False, CORRELATION_POSITION_LIMIT  # Reduce position size
        
        else:
            print(f"🔗 {symbol}: Correlation check OK - {total_correlated} correlated positions, different directions")
            return False, 1.0  # Normal position size
            
    except Exception as e:
        print(f"Error in correlation check: {e}")
        return False, 1.0  # Default to allow if error


def calculate_risk_multiplier(session_priority: int, regime_weight: float, performance_weight: float) -> float:
    """Blend session quality, regime alignment, and performance into a risk multiplier."""
    multiplier = 1.0

    # Session-based boost
    if session_priority >= 5:
        multiplier += 0.15
    elif session_priority == 4:
        multiplier += 0.05

    # Regime alignment: baseline weight ~0.2. Encourage strong fits, penalize weak.
    baseline_regime = 0.2
    multiplier += (regime_weight - baseline_regime) * 0.4

    # Performance-based adjustment (win-rate/profit factor derived weight around 1.0 - 1.5)
    multiplier += (performance_weight - 1.0) * 0.3

    # Clamp to configured bounds
    multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, multiplier))
    return round(multiplier, 2)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average Directional Index (ADX) for trend strength."""
    try:
        if len(df) < period + 1:
            return None
            
        high = df['high']
        low = df['low'] 
        close = df['close']
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = ((high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)).fillna(0)
        dm_minus = ((low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)).fillna(0)
        
        # Smooth the values
        tr_smooth = tr.rolling(period).mean()
        dm_plus_smooth = dm_plus.rolling(period).mean()
        dm_minus_smooth = dm_minus.rolling(period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None
    except Exception:
        return None


def detect_market_regime(df: pd.DataFrame) -> str:
    """Detect market regime: TRENDING, RANGING, or VOLATILE."""
    try:
        if len(df) < REGIME_LOOKBACK_PERIODS:
            return 'RANGING'  # Default to ranging if insufficient data
        
        # Calculate trend strength (ADX)
        adx = calculate_adx(df)
        
        # Calculate volatility (current ATR vs historical average)
        current_atr = compute_atr(df, period=14)
        historical_atr = compute_atr(df.iloc[:-10], period=14)  # ATR from 10 bars ago
        
        if current_atr is None or historical_atr is None or historical_atr == 0:
            return 'RANGING'
        
        volatility_ratio = current_atr / historical_atr
        
        # Regime classification logic
        if adx is not None and adx > TREND_THRESHOLD * 100:  # ADX > 30
            if volatility_ratio > VOLATILITY_THRESHOLD:
                return 'VOLATILE'  # High trend + high volatility = volatile trending
            else:
                return 'TRENDING'  # High trend + normal volatility = clean trend
        else:
            if volatility_ratio > VOLATILITY_THRESHOLD:
                return 'VOLATILE'  # Low trend + high volatility = choppy/volatile
            else:
                return 'RANGING'   # Low trend + normal volatility = sideways
                
    except Exception as e:
        print(f"Error detecting market regime: {e}")
        return 'RANGING'  # Safe default


def get_regime_strategy_weight(strategy_name: str, regime: str) -> float:
    """Get the weight for a strategy in the current market regime."""
    regime_weights = REGIME_STRATEGY_WEIGHTS.get(regime, REGIME_STRATEGY_WEIGHTS['RANGING'])
    return regime_weights.get(strategy_name, 0.2)  # Default 20% if strategy not found


def should_trade_strategy_in_regime(strategy_name: str, regime: str) -> bool:
    """Check if a strategy should be active in the current regime."""
    weight = get_regime_strategy_weight(strategy_name, regime)
    return weight >= 0.15  # Only trade strategies with 15%+ weight


# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

# Load strategy performance data
load_strategy_performance()

# Agent settings for each strategy - OPTIMIZED FROM BACKTESTING
agent_definitions = [
    {
        'label': 'Breakout',
        'cls': BreakoutAgent,
        'params': {'lookback': 10},
        'sl_mult': 2.5,
        'tp_mult': 3.0,
        'priority': 3,
        'atr_period': 14,
    },
    {
        'label': 'Donchian Channel',
        'cls': DonchianChannelAgent,
        'params': {'channel_length': 10},
        'sl_mult': 3.0,
        'tp_mult': 3.5,
        'priority': 4,
        'atr_period': 5,
    },
    {
        'label': 'MA Crossover',
        'cls': MACrossoverAgent,
        'params': {'fast_period': 5, 'slow_period': 40},
        'sl_mult': 2.0,
        'tp_mult': 2.0,
        'priority': 2,
        'atr_period': 5,
    },
    {
        'label': 'Momentum Trend',
        'cls': MomentumTrendAgent,
        'params': {'ma_period': 30, 'roc_period': 5},
        'sl_mult': 2.75,
        'tp_mult': 2.0,
        'priority': 5,
        'atr_period': 5,
    },
    {
        'label': 'Mean Reversion',
        'cls': MeanReversionAgent,
        'params': {'ma_period': 10, 'num_std': 1.0},
        'sl_mult': 2.75,
        'tp_mult': 4.0,
        'priority': 1,
        'atr_period': 7,
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


def calculate_position_size(
    symbol_info,
    atr_value: float,
    sl_mult: float,
    risk_multiplier: float = 1.0,
    account_balance: float | None = None,
) -> float:
    """Calculate position size based on volatility and risk management."""
    if account_balance is None or account_balance <= 0:
        account_balance = get_account_balance()

    if not atr_value or atr_value <= 0:
        # Fallback to conservative fixed sizing if no ATR
        name = getattr(symbol_info, "name", "") if symbol_info is not None else ""
        if isinstance(name, str) and "XAUUSD" in name.upper():
            return 0.1
        return 0.5

    # Risk amount in account currency (bounded multiplier)
    bounded_multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, risk_multiplier))
    risk_amount = account_balance * ACCOUNT_RISK_PER_TRADE * bounded_multiplier

    # Expected loss per unit if SL is hit
    base_tick_size = getattr(symbol_info, 'point', 0.01) if symbol_info is not None else 0.01
    tick_size = getattr(symbol_info, 'trade_tick_size', base_tick_size) or base_tick_size
    tick_value = getattr(symbol_info, 'trade_tick_value', 1.0) or 1.0

    sl_distance = atr_value * sl_mult
    ticks_at_risk = sl_distance / tick_size if tick_size else 0.0
    loss_per_unit = ticks_at_risk * tick_value

    if loss_per_unit <= 0:
        return MIN_LOT_SIZE

    # Calculate optimal position size
    optimal_size = risk_amount / loss_per_unit

    # Apply limits
    position_size = max(MIN_LOT_SIZE, min(optimal_size, MAX_LOT_SIZE))

    return round(position_size, 2)


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


def calculate_atr(symbol: str, timeframe=mt5.TIMEFRAME_H1, period: int = ATR_PERIOD) -> float:
    """Calculate ATR for a symbol by fetching recent price data."""
    try:
        # Get historical data for ATR calculation
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 10)
        if rates is None or len(rates) < period + 1:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Use existing compute_atr function
        return compute_atr(df, period)
    except Exception as e:
        print(f"Error calculating ATR for {symbol}: {e}")
        return None


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
    risk_multiplier: float = 1.0,
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

    spread = (tick.ask - tick.bid) if tick else None
    point = getattr(symbol_info, 'point', None) or 0.0001
    spread_points = (spread / point) if (spread is not None and point) else float('inf')
    atr_reference = atr if atr and atr > 0 else None

    # Set lot size using volatility-based sizing
    if lot is None:
        # Calculate dynamic position size based on volatility
        atr_value = atr if atr and atr > 0 else (calculate_atr(symbol, timeframe=mt5.TIMEFRAME_H1, period=5) or 0.001)
        sl_mult_effective = sl_mult if sl_mult is not None else SL_ATR_MULTIPLIER
        bounded_multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, risk_multiplier))
        lot = calculate_position_size(
            symbol_info,
            atr_value,
            sl_mult_effective,
            risk_multiplier=bounded_multiplier,
        )
        print(f"🎯 {symbol}: base lot {lot:.2f} using risk multiplier {bounded_multiplier:.2f}")

        if atr_reference is None and atr_value:
            atr_reference = atr_value
        
        # Apply correlation management
        should_block, correlation_multiplier = should_limit_correlation_exposure(symbol, signal)
        if should_block:
            print(f"🚫 {symbol}: Trade blocked due to correlation limits")
            return
        elif correlation_multiplier < 1.0:
            lot = lot * correlation_multiplier
            print(f"📉 {symbol}: Position size reduced to {lot:.2f} lots due to correlation exposure")
            
    if spread_points > SPREAD_POINTS_LIMIT:
        print(f"🚫 {symbol}: Spread {spread_points:.1f} points exceeds limit {SPREAD_POINTS_LIMIT}")
        return

    if atr_reference and atr_reference > 0 and spread is not None:
        spread_ratio = spread / atr_reference
        if spread_ratio > SPREAD_ATR_RATIO_LIMIT:
            print(f"🚫 {symbol}: Spread/ATR ratio {spread_ratio:.2f} exceeds limit {SPREAD_ATR_RATIO_LIMIT}")
            return

    lot = normalize_volume(lot, symbol_info)
    print(f"⚖️ {symbol}: Final order volume {lot:.2f} lots")
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

        # Check if we're in an active trading session
        if not is_market_session_active():
            session_name = get_session_info()
            print(f"⏰ Market session filter: Skipping trading during {session_name} (low liquidity period)")
            time.sleep(60)  # Wait 1 minute before checking again
            continue
            
        session_name = get_session_info()
        print(f"\n🌍 Active trading session: {session_name}")
        print(f"Checking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Monitor closed positions for performance tracking
        monitor_closed_positions()
        
        for symbol in symbols:
            symbol_info = prepare_symbol(symbol)
            if symbol_info is None:
                print(f"Skipping {symbol} because symbol preparation failed.")
                continue
                
            # Check instrument-specific session priority
            if not should_trade_instrument_in_session(symbol):
                session_priority = get_instrument_session_priority(symbol, session_name)
                print(f"📊 {symbol}: Skipping (session priority {session_priority}/{MIN_SESSION_PRIORITY} for {session_name})")
                continue
            else:
                session_priority = get_instrument_session_priority(symbol, session_name)
                print(f"📊 {symbol}: Trading allowed (session priority {session_priority}/{MIN_SESSION_PRIORITY} for {session_name})")
            
            # Check for news blackout periods
            is_blackout, news_reason = is_news_blackout_period(symbol)
            if is_blackout:
                print(f"📰 {symbol}: Skipping due to news blackout - {news_reason}")
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

            atr_cache_by_period: dict[int, float | None] = {ATR_PERIOD: atr_value}
                
            # Detect market regime for strategy optimization
            current_regime = detect_market_regime(df)
            print(f"📊 {symbol} Market Regime: {current_regime}")
            
            # Advanced stop management for existing positions
            if atr_value:
                manage_position_stops(symbol, atr_value)
            
            positions = mt5.positions_get(symbol=symbol) or []
            have_buy = any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
            have_sell = any(p.type == mt5.POSITION_TYPE_SELL for p in positions)

            all_candidates = []

            for agent_info in agent_dict[symbol]:
                agent = agent_info['agent']
                strategy_name = agent_info.get('label') or type(agent).__name__.replace('Agent', '').replace('_', ' ').strip() or 'Strategy'
                strategy_key = strategy_name.lower().replace(' ', '_')
                
                # Check if strategy should be active in current regime
                if not should_trade_strategy_in_regime(strategy_key, current_regime):
                    regime_weight = get_regime_strategy_weight(strategy_key, current_regime)
                    print(f"🚫 {symbol} {strategy_name}: Disabled in {current_regime} regime (weight: {regime_weight:.1%})")
                    continue
                
                sl_mult = agent_info.get('sl_mult', SL_ATR_MULTIPLIER)
                tp_mult = agent_info.get('tp_mult', TP_ATR_MULTIPLIER)
                priority = agent_info.get('priority', 999)
                atr_period_override = agent_info.get('atr_period', ATR_PERIOD)
                strategy_atr_value = atr_cache_by_period.get(atr_period_override)
                if strategy_atr_value is None:
                    strategy_atr_value = compute_atr(df, atr_period_override)
                    atr_cache_by_period[atr_period_override] = strategy_atr_value
                if strategy_atr_value is None:
                    print(f"🚫 {symbol} {strategy_name}: Insufficient data for ATR({atr_period_override})")
                    continue
                
                # Adjust priority based on regime and performance (lower number = higher priority)
                regime_weight = get_regime_strategy_weight(strategy_key, current_regime)
                performance_weight = get_strategy_performance_weight(strategy_key, symbol)
                combined_weight = regime_weight * performance_weight
                regime_adjusted_priority = priority / combined_weight  # Higher weight = better priority
                risk_multiplier = calculate_risk_multiplier(session_priority, regime_weight, performance_weight)
                
                signal = agent.get_signal(df)
                micro_momentum_score = None
                micro_momentum_threshold = None
                micro_soft_pass = False
                
                # Multi-timeframe confirmation
                if signal in ('buy', 'sell'):
                    mtf_confirmed = confirm_signal_with_mtf(signal, symbol)
                    if not mtf_confirmed:
                        mtf_bias = get_mtf_trend_bias(symbol)
                        print(f"🔍 {symbol} {strategy_name}: Signal {signal} rejected by MTF filter (H1 bias: {mtf_bias})")
                        continue  # Skip this signal
                    else:
                        mtf_bias = get_mtf_trend_bias(symbol)
                        print(f"✅ {symbol} {strategy_name}: Signal {signal} confirmed by MTF (H1 bias: {mtf_bias})")

                    reference_price = float(df['close'].iloc[-1]) if not df['close'].empty else None
                    micro_pass, micro_score, micro_threshold, micro_soft = confirm_with_micro_momentum(
                        symbol,
                        signal,
                        current_regime,
                        strategy_atr_value,
                        reference_price,
                    )
                    if not micro_pass:
                        print(f"🎯 {symbol} {strategy_name}: Signal {signal} rejected by micro momentum ({micro_score:+.2%} vs {micro_threshold:.2%})")
                        continue
                    else:
                        micro_momentum_score = micro_score
                        micro_momentum_threshold = micro_threshold
                        micro_soft_pass = micro_soft
                        softness_text = " (soft confirm)" if micro_soft else ""
                        print(f"🎯 {symbol} {strategy_name}: Micro momentum aligned {micro_score:+.2%} ≥ {micro_threshold:.2%}{softness_text}")
                
                if micro_momentum_score is not None:
                    softness_text = " soft" if micro_soft_pass else ""
                    micro_text = f", micro {micro_momentum_score:+.2%}/{micro_momentum_threshold:.2%}{softness_text}"
                else:
                    micro_text = ""
                print(f"{symbol} {strategy_name} Final Signal: {signal} (regime: {regime_weight:.1%}, performance: {performance_weight:.1f}x, adj priority: {regime_adjusted_priority:.1f}, risk x{risk_multiplier:.2f}, ATR({atr_period_override})={strategy_atr_value:.6f}{micro_text})")

                if signal in ('buy', 'sell'):
                    all_candidates.append({
                        'signal': signal,
                        'label': strategy_name,
                        'priority': regime_adjusted_priority,  # Use regime-adjusted priority
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult,
                        'regime_weight': regime_weight,
                        'risk_multiplier': risk_multiplier,
                        'micro_momentum': micro_momentum_score,
                        'micro_threshold': micro_momentum_threshold,
                        'micro_soft_pass': micro_soft_pass,
                        'atr_value': strategy_atr_value,
                        'atr_period': atr_period_override,
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
                        order_comment = build_order_comment(best_candidate['label'], 'buy', best_candidate['priority'])
                        print(f"{symbol}: executing BUY via {best_candidate['label']} (priority {best_candidate['priority']}) | comment {order_comment}")
                        send_order(
                            symbol,
                            'buy',
                            comment=order_comment,
                            risk_multiplier=best_candidate['risk_multiplier'],
                            atr=best_candidate['atr_value'],
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
                        order_comment = build_order_comment(best_candidate['label'], 'sell', best_candidate['priority'])
                        print(f"{symbol}: executing SELL via {best_candidate['label']} (priority {best_candidate['priority']}) | comment {order_comment}")
                        send_order(
                            symbol,
                            'sell',
                            comment=order_comment,
                            risk_multiplier=best_candidate['risk_multiplier'],
                            atr=best_candidate['atr_value'],
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
