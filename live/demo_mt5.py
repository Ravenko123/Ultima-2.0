import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
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

# Risk guard settings
RISK_GUARD_FILE = "risk_guard_state.json"
DAILY_MAX_DRAWDOWN = 0.12  # 12% maximum drop from daily peak
WEEKLY_MAX_DRAWDOWN = 0.20  # 20% maximum drop from weekly peak
RISK_RECOVERY_THRESHOLD = 0.05  # reduce risk once drawdown exceeds 5%
RISK_RECOVERY_MIN_FACTOR = 0.35  # never risk more than 35% of base risk when deep in drawdown
RISK_GUARD_ENABLED = False  # Temporarily disabled per latest instructions

risk_guard_state: dict[str, object] = {}
soft_guard_state: dict[str, float | bool] = {
    "blocked": False,
    "throttle": 1.0,
    "drawdown": 0.0,
}

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


def _week_start(date_obj: datetime) -> datetime:
    return (date_obj - timedelta(days=date_obj.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def load_risk_guard_state() -> dict[str, object]:
    global risk_guard_state
    if not RISK_GUARD_ENABLED:
        risk_guard_state = {}
        return risk_guard_state
    try:
        with open(RISK_GUARD_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                risk_guard_state = data
            else:
                risk_guard_state = {}
    except FileNotFoundError:
        risk_guard_state = {}
    except Exception as exc:
        print(f"⚠️  Failed to load risk guard state: {exc}")
        risk_guard_state = {}
    return risk_guard_state


def save_risk_guard_state() -> None:
    if not RISK_GUARD_ENABLED:
        return
    try:
        with open(RISK_GUARD_FILE, "w", encoding="utf-8") as handle:
            json.dump(risk_guard_state, handle, indent=2)
    except Exception as exc:
        print(f"⚠️  Failed to save risk guard state: {exc}")


def _reset_daily_state(now: datetime, equity: float) -> None:
    risk_guard_state["day"] = now.date().isoformat()
    risk_guard_state["daily_start_equity"] = equity
    risk_guard_state["daily_peak_equity"] = equity
    risk_guard_state["daily_drawdown"] = 0.0
    risk_guard_state["daily_blocked"] = False


def _reset_weekly_state(now: datetime, equity: float) -> None:
    week_anchor = _week_start(now)
    risk_guard_state["week_start"] = week_anchor.date().isoformat()
    risk_guard_state["weekly_start_equity"] = equity
    risk_guard_state["weekly_peak_equity"] = equity
    risk_guard_state["weekly_drawdown"] = 0.0
    risk_guard_state["weekly_blocked"] = False


def update_risk_guard(equity: float, now: datetime | None = None) -> None:
    if now is None:
        now = datetime.now()

    if not RISK_GUARD_ENABLED:
        risk_guard_state.clear()
        return

    if not risk_guard_state:
        _reset_daily_state(now, equity)
        _reset_weekly_state(now, equity)

    if risk_guard_state.get("day") != now.date().isoformat():
        _reset_daily_state(now, equity)

    current_week = _week_start(now).date().isoformat()
    if risk_guard_state.get("week_start") != current_week:
        _reset_weekly_state(now, equity)

    risk_guard_state["equity"] = equity

    daily_peak = max(risk_guard_state.get("daily_peak_equity", equity), equity)
    risk_guard_state["daily_peak_equity"] = daily_peak
    if daily_peak > 0:
        daily_drawdown = max(0.0, (daily_peak - equity) / daily_peak)
    else:
        daily_drawdown = 0.0
    risk_guard_state["daily_drawdown"] = daily_drawdown
    if daily_drawdown >= DAILY_MAX_DRAWDOWN:
        if not risk_guard_state.get("daily_blocked"):
            print(f"🚨 Daily drawdown {daily_drawdown:.1%} breached {DAILY_MAX_DRAWDOWN:.0%} limit. Pausing trading until next session.")
        risk_guard_state["daily_blocked"] = True

    weekly_peak = max(risk_guard_state.get("weekly_peak_equity", equity), equity)
    risk_guard_state["weekly_peak_equity"] = weekly_peak
    if weekly_peak > 0:
        weekly_drawdown = max(0.0, (weekly_peak - equity) / weekly_peak)
    else:
        weekly_drawdown = 0.0
    risk_guard_state["weekly_drawdown"] = weekly_drawdown
    if weekly_drawdown >= WEEKLY_MAX_DRAWDOWN:
        if not risk_guard_state.get("weekly_blocked"):
            print(f"🚨 Weekly drawdown {weekly_drawdown:.1%} breached {WEEKLY_MAX_DRAWDOWN:.0%} limit. Holding fire until new week.")
        risk_guard_state["weekly_blocked"] = True

    save_risk_guard_state()


def risk_guard_allow_trade(now: datetime | None = None) -> bool:
    if not RISK_GUARD_ENABLED:
        return True
    if now is None:
        now = datetime.now()
    if not risk_guard_state:
        return True

    if risk_guard_state.get("daily_blocked"):
        if risk_guard_state.get("day") != now.date().isoformat():
            risk_guard_state["daily_blocked"] = False
        else:
            return False

    week_anchor = _week_start(now).date().isoformat()
    if risk_guard_state.get("weekly_blocked"):
        if risk_guard_state.get("week_start") != week_anchor:
            risk_guard_state["weekly_blocked"] = False
        else:
            return False

    return True


def risk_guard_drawdown_factor() -> float:
    if not RISK_GUARD_ENABLED:
        return 1.0
    if not risk_guard_state:
        return 1.0
    daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0))
    weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0))
    dd = max(daily_dd, weekly_dd)
    if dd <= RISK_RECOVERY_THRESHOLD:
        return 1.0
    span = max(1e-6, max(DAILY_MAX_DRAWDOWN, WEEKLY_MAX_DRAWDOWN) - RISK_RECOVERY_THRESHOLD)
    scaled = min(1.0, (dd - RISK_RECOVERY_THRESHOLD) / span)
    return max(RISK_RECOVERY_MIN_FACTOR, 1.0 - scaled)


def evaluate_soft_guard(balance: float | None, equity: float | None) -> dict[str, float | bool | None]:
    """Assess soft guard state based on current balance/equity."""
    if not SOFT_GUARD_ENABLED or balance is None or balance <= 0 or equity is None:
        soft_guard_state.update({"blocked": False, "throttle": 1.0, "drawdown": 0.0})
        return {"blocked": False, "throttle": 1.0, "drawdown": 0.0, "transition": None}

    previous_blocked = bool(soft_guard_state.get("blocked", False))
    previous_throttle = float(soft_guard_state.get("throttle", 1.0) or 1.0)

    drawdown = max(0.0, (balance - equity) / balance)

    blocked = previous_blocked
    if previous_blocked:
        if drawdown <= SOFT_GUARD_RESUME:
            blocked = False
    else:
        if drawdown >= SOFT_GUARD_LIMIT:
            blocked = True

    if blocked:
        throttle = 0.0
    else:
        if drawdown <= SOFT_GUARD_RESUME:
            throttle = 1.0
        else:
            span = max(1e-6, SOFT_GUARD_LIMIT - SOFT_GUARD_RESUME)
            ratio = min(1.0, max(0.0, (drawdown - SOFT_GUARD_RESUME) / span))
            throttle = max(0.35, 1.0 - ratio * 0.65)

    transition: dict[str, float | str] | None = None
    if blocked != previous_blocked:
        transition = {"type": "block" if blocked else "resume"}
    elif abs(throttle - previous_throttle) >= 0.05:
        transition = {"type": "throttle", "value": throttle}

    soft_guard_state.update({"blocked": blocked, "throttle": throttle, "drawdown": drawdown})
    return {"blocked": blocked, "throttle": throttle, "drawdown": drawdown, "transition": transition}


def apply_risk_guard_to_multiplier(multiplier: float) -> float:
    guard_factor = risk_guard_drawdown_factor() if RISK_GUARD_ENABLED else 1.0
    soft_factor = float(soft_guard_state.get("throttle", 1.0) or 1.0) if SOFT_GUARD_ENABLED else 1.0
    combined_factor = guard_factor * soft_factor

    adjusted = multiplier * combined_factor

    min_bound = RISK_MULTIPLIER_MIN
    if RISK_GUARD_ENABLED and guard_factor < 1.0:
        min_bound = min(min_bound, RISK_RECOVERY_MIN_FACTOR)
    if SOFT_GUARD_ENABLED and soft_factor < 1.0:
        min_bound = min(min_bound, max(0.35, soft_factor))

    adjusted = max(min_bound, min(RISK_MULTIPLIER_MAX, adjusted))
    return round(adjusted, 2)


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
ACCOUNT_RISK_PER_TRADE = 0.12  # Reduced base risk to balance fast growth with capital protection
MIN_LOT_SIZE = 0.01  # Minimum position size
MAX_LOT_SIZE = 5.0   # Maximum position size cap

# Dynamic risk multiplier bounds
RISK_MULTIPLIER_MIN = 1.0
RISK_MULTIPLIER_MAX = 3.5

# Scan cadence configuration
SCAN_INTERVAL_SECONDS = 120       # Time between scan loops (seconds)

# Soft drawdown guard (still aggressive but avoids death spirals)
SOFT_GUARD_ENABLED = True
SOFT_GUARD_LIMIT = 0.35           # Block new trades if unrealized DD exceeds 35% of balance
SOFT_GUARD_RESUME = 0.20          # Resume trading once DD recovers below 20%

# Drawdown-aware micro confirmation tuning
DRAWDOWN_RELAXATION_PER_ATR = 0.25   # Relax micro alignment 25% per ATR of drawdown
DRAWDOWN_OVERRIDE_ATR = 1.2          # Allow fallback overrides when drawdown exceeds 1.2 ATR

# Aggressive mode tuning
AGGRESSIVE_MODE = True
AGGRESSIVE_REGIME_THRESHOLD = 0.05  # Minimal regime weight still allowed when aggressive
AGGRESSIVE_MICRO_TOLERANCE = 0.25   # Allow counter-momentum within 25% of threshold
AGGRESSIVE_MICRO_MIN_RATIO = 0.25   # Require at least 25% of base threshold in the trade direction
HIGH_CONFIDENCE_BOOST_CAP = 1.5     # Max boost applied when confidence is exceptional

# Pyramiding (position stacking) configuration
PYRAMID_ENABLED = True
PYRAMID_MAX_ENTRIES_PER_SIDE = 3          # Total entries (initial + adds) allowed per direction
PYRAMID_MIN_CONFIDENCE = 0.9              # Require strong conviction to stack
PYRAMID_MAX_LOSING_POSITIONS = 3          # Allow deeper averaging while managing size
PYRAMID_WINNING_RISK_BOOST = 1.2          # Risk multiplier boost when scaling into profitable legs
PYRAMID_LOSING_RISK_BOOST = 1.0           # Neutral sizing when averaging into drawdown

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

# Signal confidence gating
CONFIDENCE_EXECUTION_THRESHOLD = 0.65  # Slightly looser gate for higher trade frequency


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


def get_account_equity_quiet() -> float:
    """Fetch account equity without emitting log noise."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0
        equity_value = getattr(account_info, "equity", None)
        if equity_value is None:
            return float(account_info.balance)
        return float(equity_value)
    except Exception as exc:
        print(f"⚠️  Failed to fetch account equity: {exc}")
        return 0.0


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

DEFAULT_CORRELATION_LIMITS = {
    "max_same_direction": 2,
    "max_total": 3,
    "size_multiplier": 0.6,
}

CORRELATION_GROUP_LIMITS = {
    "EUR_PAIRS": {"max_same_direction": 3, "max_total": 4, "size_multiplier": 0.65},
    "GBP_PAIRS": {"max_same_direction": 3, "max_total": 4, "size_multiplier": 0.6},
    "JPY_PAIRS": {"max_same_direction": 2, "max_total": 3, "size_multiplier": 0.5},
    "USD_MAJORS": {"max_same_direction": 3, "max_total": 4, "size_multiplier": 0.65},
    "SAFE_HAVEN": {"max_same_direction": 2, "max_total": 2, "size_multiplier": 0.5},
}

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

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            mt5.symbol_select(symbol, True)
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Unable to retrieve symbol info for {symbol}; skipping partial close.")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"No tick data for {symbol}; cannot execute partial close.")
            return False
        
        # Determine order type (opposite of position type)
        if position.type == 0:  # Close buy position with sell order
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:  # Close sell position with buy order
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        max_closeable = float(getattr(position, 'volume', 0.0) or 0.0)
        if max_closeable <= 0:
            return False

        requested_volume = min(volume, max_closeable)
        normalized_volume = normalize_volume(requested_volume, symbol_info)

        if normalized_volume <= 0:
            print(f"Partial close volume normalized to zero for {symbol}; aborting.")
            return False

        if normalized_volume > max_closeable:
            normalized_volume = max_closeable

        residual = max_closeable - normalized_volume
        min_volume = getattr(symbol_info, 'volume_min', 0.01) or 0.01
        step = getattr(symbol_info, 'volume_step', 0.01) or 0.01

        if residual > 0 and residual < min_volume:
            # Leaving less than the minimum causes rejection; adjust to close entire position
            normalized_volume = max_closeable

        filling_type = resolve_filling_type(symbol_info)
        time_type = getattr(mt5, 'ORDER_TIME_GTC', 0)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": normalized_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "comment": "Partial profit",
            "type_filling": filling_type,
            "type_time": time_type,
            "deviation": 10,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        elif result and result.retcode == getattr(mt5, 'TRADE_RETCODE_PLACED', -1):
            print(f"Partial close for ticket {ticket} placed (pending).")
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


def _calculate_drawdown_pressure_atr(
    positions: list | None,
    atr_value: float | None,
    signal: str,
) -> tuple[float, float, float, float]:
    """Estimate drawdown pressure in ATR units for the given signal direction.

    Returns tuple of (same_direction, opposite_direction, worst_single, total_loss)
    expressed in ATR multiples. Values are 0.0 when unavailable.
    """
    if not positions or atr_value is None or atr_value <= 0:
        return 0.0, 0.0, 0.0, 0.0

    same_direction_dd = 0.0
    opposite_direction_dd = 0.0
    worst_dd = 0.0
    total_dd = 0.0

    for pos in positions:
        price_open = float(getattr(pos, 'price_open', 0.0) or 0.0)
        price_current = float(getattr(pos, 'price_current', 0.0) or 0.0)
        if price_open <= 0 or price_current <= 0:
            continue

        if getattr(pos, 'type', None) == mt5.POSITION_TYPE_BUY:
            adverse_move = price_open - price_current
            pos_side = 'buy'
        else:
            adverse_move = price_current - price_open
            pos_side = 'sell'

        if adverse_move <= 0:
            continue  # Position is not under water

        drawdown_atr = adverse_move / atr_value
        worst_dd = max(worst_dd, drawdown_atr)
        total_dd += drawdown_atr

        if (signal == 'buy' and pos_side == 'buy') or (signal == 'sell' and pos_side == 'sell'):
            same_direction_dd = max(same_direction_dd, drawdown_atr)
        else:
            opposite_direction_dd = max(opposite_direction_dd, drawdown_atr)

    return same_direction_dd, opposite_direction_dd, worst_dd, total_dd


def confirm_with_micro_momentum(
    symbol: str,
    signal: str,
    regime: str,
    atr_value: float | None,
    reference_price: float | None,
    open_positions: list | None = None,
) -> tuple[bool, float, float, bool, bool]:
    """Use micro timeframe momentum bias to refine entries."""
    if not ENABLE_MICRO_MOMENTUM_CONFIRMATION:
        return True, 0.0, 0.0, False, False

    try:
        override_used = False
        request_count = MICRO_MOMENTUM_LOOKBACK + 3
        rates = mt5.copy_rates_from_pos(symbol, MICRO_MOMENTUM_TIMEFRAME, 0, request_count)
        if rates is None or len(rates) < MICRO_MOMENTUM_LOOKBACK + 2:
            return True, 0.0, 0.0, False, False  # Skip if insufficient data

        df = pd.DataFrame(rates)
        close = df['close'].astype(float)
        momentum_series = close.pct_change().rolling(MICRO_MOMENTUM_LOOKBACK).sum()
        momentum_score = momentum_series.iloc[-1]

        if pd.isna(momentum_score):
            return True, 0.0, 0.0, False, False

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

        same_dir_dd_atr, opposite_dir_dd_atr, worst_dd_atr, total_dd_atr = _calculate_drawdown_pressure_atr(
            open_positions,
            atr_value,
            signal,
        )
        drawdown_pressure_atr = same_dir_dd_atr if same_dir_dd_atr > 0 else opposite_dir_dd_atr

        if drawdown_pressure_atr > 0:
            relaxation_multiplier = max(0.25, 1.0 - min(2.0, drawdown_pressure_atr) * DRAWDOWN_RELAXATION_PER_ATR)
            if relaxation_multiplier < 1.0:
                threshold *= relaxation_multiplier

        soft_pass = False
        aligned_momentum = momentum_score if signal == 'buy' else -momentum_score
        tolerance = threshold * AGGRESSIVE_MICRO_TOLERANCE
        if drawdown_pressure_atr > 0:
            tolerance *= 1.0 + min(0.6, drawdown_pressure_atr * 0.3)
        alignment_ratio = AGGRESSIVE_MICRO_MIN_RATIO
        if regime == 'VOLATILE':
            alignment_ratio *= 0.5
        elif regime == 'RANGING':
            alignment_ratio *= 0.75
        if drawdown_pressure_atr > 0:
            alignment_ratio *= max(0.4, 1.0 - min(1.5, drawdown_pressure_atr) * 0.2)
        alignment_ratio = max(0.05, alignment_ratio)
        required_alignment = threshold * alignment_ratio

        if signal == 'buy':
            if momentum_score >= threshold:
                return True, float(momentum_score), threshold, soft_pass, override_used
            elif momentum_score >= threshold * MICRO_MOMENTUM_SOFT_PASS_RATIO:
                soft_pass = True
                return True, float(momentum_score), threshold, soft_pass, override_used
        else:
            if momentum_score <= -threshold:
                return True, float(momentum_score), threshold, soft_pass, override_used
            elif momentum_score <= -threshold * MICRO_MOMENTUM_SOFT_PASS_RATIO:
                soft_pass = True
                return True, float(momentum_score), threshold, soft_pass, override_used

        if drawdown_pressure_atr >= DRAWDOWN_OVERRIDE_ATR and aligned_momentum >= -tolerance:
            soft_pass = True
            override_used = True
            print(
                f"🩹 {symbol}: Drawdown override for {signal.upper()} (pressure {drawdown_pressure_atr:.2f} ATR, momentum {aligned_momentum:+.2%} within tolerance)."
            )
            return True, float(momentum_score), threshold, soft_pass, override_used

        if AGGRESSIVE_MODE and regime in ('TRENDING', 'VOLATILE'):
            if aligned_momentum >= required_alignment:
                soft_pass = True
                override_used = True
                tolerance_pct = f"±{tolerance:.2%}"
                required_symbol = "≥" if signal == 'buy' else "≤"
                required_value = required_alignment if signal == 'buy' else -required_alignment
                required_pct = f"{required_symbol} {required_value:.2%}"
                print(
                    f"⚡ {symbol}: Aggressive micro override for {signal.upper()} (aligned {aligned_momentum:+.2%} within tolerance {tolerance_pct}, required {required_pct})."
                )
                return True, float(momentum_score), threshold, soft_pass, override_used

        return False, float(momentum_score), threshold, soft_pass, override_used
    except Exception as e:
        print(f"Error computing micro momentum for {symbol}: {e}")
        return True, 0.0, MICRO_MOMENTUM_MIN_THRESHOLD, False, False


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
        
        group_stats: dict[str, dict[str, float]] = {}
        for pos in correlated_positions:
            group = pos.get('group')
            if not group:
                continue
            stats = group_stats.setdefault(group, {'same': 0, 'opposite': 0, 'total': 0})
            stats['total'] += 1
            if pos['type'] == signal:
                stats['same'] += 1
            else:
                stats['opposite'] += 1

        block_reasons: list[tuple[str, dict[str, float], dict[str, float]]] = []
        scale_reasons: list[tuple[str, dict[str, float], dict[str, float]]] = []

        for group, stats in group_stats.items():
            limits = CORRELATION_GROUP_LIMITS.get(group, DEFAULT_CORRELATION_LIMITS)
            if stats['same'] >= limits['max_same_direction']:
                block_reasons.append((group, stats, limits))
            elif stats['total'] >= limits['max_total']:
                scale_reasons.append((group, stats, limits))
            elif stats['same'] > 0:
                # Still scale down if we already have exposure in the same direction
                scale_reasons.append((group, stats, limits))

        if block_reasons:
            for group, stats, limits in block_reasons:
                print(
                    f"🔗 {symbol}: Blocking trade - {stats['same']} same-direction positions in {group} "
                    f"(limit {limits['max_same_direction']})."
                )
            return True, 0.0

        if scale_reasons:
            size_multiplier = 1.0
            strongest_reason = None
            for group, stats, limits in scale_reasons:
                size_multiplier = min(size_multiplier, limits['size_multiplier'])
                if strongest_reason is None or limits['size_multiplier'] < strongest_reason[2]['size_multiplier']:
                    strongest_reason = (group, stats, limits)

            if strongest_reason:
                group, stats, limits = strongest_reason
                net_bias = stats['same'] - stats['opposite']
                print(
                    f"🔗 {symbol}: Correlation exposure in {group} ({stats['total']}/{limits['max_total']} total, {stats['same']} same-direction, net bias {net_bias:+.0f}). "
                    f"Scaling position by {size_multiplier:.0%}."
                )
            return False, size_multiplier

        print(f"🔗 {symbol}: Correlation check OK - diversified exposure across correlated groups")
        return False, 1.0  # Normal position size
            
    except Exception as e:
        print(f"Error in correlation check: {e}")
        return False, 1.0  # Default to allow if error


def evaluate_pyramiding(
    symbol: str,
    signal: str,
    positions: list,
    candidate_confidence: float,
    atr_value: float | None = None,
) -> tuple[bool, float, str]:
    """Determine whether we can stack another position in the same direction."""
    if not PYRAMID_ENABLED:
        return False, 1.0, "pyramiding disabled"

    if signal not in ("buy", "sell"):
        return False, 1.0, "invalid signal"

    direction_type = mt5.POSITION_TYPE_BUY if signal == "buy" else mt5.POSITION_TYPE_SELL
    same_direction_positions = [p for p in positions if p.type == direction_type]

    if not same_direction_positions:
        return False, 1.0, "no base position"

    if len(same_direction_positions) >= PYRAMID_MAX_ENTRIES_PER_SIDE:
        return False, 1.0, "max stacks reached"

    if candidate_confidence < PYRAMID_MIN_CONFIDENCE:
        return False, 1.0, "confidence below pyramid threshold"

    total_profit = sum(float(getattr(p, "profit", 0.0) or 0.0) for p in same_direction_positions)
    losing_leg = total_profit < 0

    if losing_leg and len(same_direction_positions) >= PYRAMID_MAX_LOSING_POSITIONS:
        return False, 1.0, "losing stack limit reached"

    drawdown_same, drawdown_opposite, worst_drawdown_atr, total_drawdown_atr = _calculate_drawdown_pressure_atr(
        positions,
        atr_value,
        signal,
    )
    drawdown_pressure = drawdown_same if drawdown_same > 0 else max(drawdown_opposite, worst_drawdown_atr * 0.5)

    confidence_bonus = max(0.0, candidate_confidence - PYRAMID_MIN_CONFIDENCE)
    stack_load_ratio = len(same_direction_positions) / max(1, PYRAMID_MAX_ENTRIES_PER_SIDE)

    if losing_leg:
        base_scale = PYRAMID_LOSING_RISK_BOOST
        drawdown_scale = max(0.55, 1.0 - min(2.5, drawdown_pressure) * 0.2)
        risk_boost = base_scale * drawdown_scale
        rationale = "averaging into drawdown"
        if drawdown_pressure > 0:
            rationale += f" ({drawdown_pressure:.2f} ATR adverse)"
    else:
        base_scale = PYRAMID_WINNING_RISK_BOOST
        confidence_scale = 1.0 + min(0.35, confidence_bonus * 0.5)
        risk_boost = base_scale * confidence_scale
        rationale = "scaling a winner"
        if confidence_bonus > 0:
            rationale += f" (confidence +{confidence_bonus:.2f})"

    stack_scale = max(0.6, 1.0 - stack_load_ratio * 0.25)
    risk_boost *= stack_scale
    if stack_load_ratio > 0:
        rationale += f", stack {len(same_direction_positions)}/{PYRAMID_MAX_ENTRIES_PER_SIDE}"

    correlation_block, correlation_multiplier = should_limit_correlation_exposure(symbol, signal)
    if correlation_block:
        return False, 1.0, "correlation limit"

    if correlation_multiplier < 1.0:
        risk_boost *= correlation_multiplier
        rationale += f", correlation {correlation_multiplier:.0%}"

    if drawdown_pressure > 0 and not losing_leg:
        mitigation_scale = max(0.7, 1.0 - min(2.0, drawdown_pressure) * 0.15)
        risk_boost *= mitigation_scale
        rationale += f", drawdown guard {mitigation_scale:.0%}"

    risk_boost = max(0.5, min(risk_boost, RISK_MULTIPLIER_MAX))
    return True, risk_boost, rationale


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
    if weight >= 0.15:
        return True
    if AGGRESSIVE_MODE and weight >= AGGRESSIVE_REGIME_THRESHOLD:
        return True
    return False  # Default: disable very low-weight strategies


# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

# Load strategy performance data
load_strategy_performance()
load_risk_guard_state()

initial_equity = get_account_equity_quiet()
if initial_equity > 0:
    update_risk_guard(initial_equity)
else:
    print("⚠️  Unable to prime risk guard with account equity (0 received).")

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


def calculate_volatility_pressure(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> float:
    """Measure volatility expansion or contraction via ATR ratio."""
    try:
        if df is None or len(df) < long_period + 5:
            return 1.0
        short_atr = compute_atr(df, period=short_period)
        long_atr = compute_atr(df, period=long_period)
        if short_atr is None or long_atr is None or long_atr <= 0:
            return 1.0
        ratio = short_atr / long_atr
        return max(0.5, min(1.8, float(ratio)))
    except Exception:
        return 1.0


def calculate_candle_conviction(df: pd.DataFrame) -> float:
    """Gauge strength of the last completed candle (body size vs wicks)."""
    try:
        if df is None or len(df) < 3:
            return 1.0
        last = df.iloc[-2]
        open_price = float(last['open'])
        close_price = float(last['close'])
        high = float(last['high'])
        low = float(last['low'])
        price_range = max(1e-10, high - low)
        body = abs(close_price - open_price)
        upper_wick = max(0.0, high - max(open_price, close_price))
        lower_wick = max(0.0, min(open_price, close_price) - low)
        wick_ratio = (upper_wick + lower_wick) / price_range
        body_ratio = body / price_range
        directional_bias = 1.08 if close_price > open_price else 0.95
        conviction = body_ratio * (1.0 - 0.5 * wick_ratio) * directional_bias
        return max(0.5, min(1.5, float(conviction)))
    except Exception:
        return 1.0


def calculate_signal_confidence(
    session_priority: int,
    regime_weight: float,
    performance_weight: float,
    guard_factor: float,
    signal_direction: str,
    micro_score: float | None,
    micro_threshold: float | None,
    micro_soft: bool,
    volatility_pressure: float,
    candle_conviction: float,
) -> float:
    """Blend contextual factors into a confidence score around 1.0 baseline."""
    base = 1.0

    # Regime alignment: normalize around 0.3 baseline
    regime_factor = 0.65 + 1.1 * min(1.0, regime_weight / 0.35)
    base *= regime_factor

    # Strategy performance factor (win-rate/profit factor proxy around 1.0-1.5)
    perf_factor = 0.7 + 0.3 * min(1.6, max(0.4, performance_weight))
    base *= perf_factor

    # Session quality scaling (1.0 at top-tier sessions)
    session_factor = 0.7 + 0.3 * min(1.0, max(0.4, session_priority / 5))
    base *= session_factor

    # Guard factor (drawdown recovery throttles)
    guard_factor = max(0.4, min(1.0, guard_factor))
    base *= (0.75 + 0.25 * guard_factor)

    if micro_score is not None and micro_threshold:
        alignment_raw = micro_score if signal_direction == 'buy' else -micro_score
        effective_threshold = max(1e-9, micro_threshold)
        alignment_ratio = alignment_raw / effective_threshold
        if micro_soft and alignment_ratio < 1.0:
            alignment_ratio = max(0.6, alignment_ratio)
        if alignment_ratio >= 1.0:
            micro_factor = min(1.5, 1.0 + 0.25 * (alignment_ratio - 1.0))
        elif alignment_ratio >= 0.0:
            micro_factor = 0.85 + 0.15 * alignment_ratio
        else:
            micro_factor = max(0.55, 1.0 + 0.2 * alignment_ratio)
        base *= micro_factor
    else:
        base *= 0.9  # slight penalty without micro confirmation detail

    base *= max(0.75, min(1.3, volatility_pressure))
    base *= max(0.75, min(1.25, candle_conviction))

    confidence = 0.45 + base * 0.35
    return round(float(min(1.8, confidence)), 3)


def adjust_risk_with_confidence(multiplier: float, confidence: float) -> float:
    """Tilt risk multiplier up/down based on confidence."""
    adjusted = multiplier
    if confidence >= 1.25:
        boost = min(HIGH_CONFIDENCE_BOOST_CAP, confidence + 0.1)
        adjusted *= boost
    elif confidence < 0.9:
        cut = max(0.65, confidence)
        adjusted *= cut
    return adjusted


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

scan_counter = 0

try:
    while True:
        scan_counter += 1
        if not ensure_connection_ready():
            print("Waiting before retrying MT5 connection check...")
            time.sleep(5)
            continue

        cycle_stats: dict[str, int] = defaultdict(int)
        now_local = datetime.now()
        now_utc = datetime.now(timezone.utc)
        print("\n" + "=" * 68)
        print(f"🔁 Scan #{scan_counter} | Local {now_local.strftime('%Y-%m-%d %H:%M:%S')} | UTC {now_utc.strftime('%H:%M:%S')}")

        account_info = mt5.account_info()
        if account_info:
            print(
                "💼 Account snapshot -> "
                f"Balance: {account_info.balance:.2f} | Equity: {account_info.equity:.2f} | "
                f"Free Margin: {account_info.margin_free:.2f} | Margin Used: {account_info.margin:.2f} | "
                f"Open PnL: {account_info.profit:.2f}"
            )
        else:
            print("💼 Account snapshot unavailable (mt5.account_info() returned None).")

        soft_guard_status = evaluate_soft_guard(
            getattr(account_info, 'balance', None) if account_info else None,
            getattr(account_info, 'equity', None) if account_info else None,
        )

        active_positions = mt5.positions_get() or []
        if active_positions:
            position_summary: dict[str, dict[str, float]] = defaultdict(lambda: {'count': 0, 'buy': 0.0, 'sell': 0.0, 'pnl': 0.0})
            for pos in active_positions:
                stats = position_summary[pos.symbol]
                stats['count'] += 1
                stats['pnl'] += float(getattr(pos, 'profit', 0.0) or 0.0)
                volume = float(getattr(pos, 'volume', 0.0) or 0.0)
                if pos.type == mt5.POSITION_TYPE_BUY:
                    stats['buy'] += volume
                else:
                    stats['sell'] += volume
            summary_parts = []
            for sym in sorted(position_summary.keys()):
                stats = position_summary[sym]
                net_vol = stats['buy'] - stats['sell']
                summary_parts.append(
                    f"{sym}: {int(stats['count'])} pos (buy {stats['buy']:.2f}, sell {stats['sell']:.2f}, net {net_vol:+.2f}, pnl {stats['pnl']:.2f})"
                )
            print("📂 Open positions -> " + " | ".join(summary_parts))
        else:
            print("📂 Open positions -> none")

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

        equity_snapshot = get_account_equity_quiet()
        if equity_snapshot > 0:
            update_risk_guard(equity_snapshot)
        else:
            print("⚠️  Account equity unavailable; keeping previous risk guard snapshot.")

        risk_guard_allowed = risk_guard_allow_trade()
        guard_factor = risk_guard_drawdown_factor()
        soft_guard_allowed = not bool(soft_guard_status.get("blocked", False))
        soft_guard_factor = float(soft_guard_status.get("throttle", 1.0) or 1.0)
        combined_guard_factor = guard_factor * (soft_guard_factor if soft_guard_allowed else 0.0)
        guard_trading_allowed = risk_guard_allowed and soft_guard_allowed

        if not RISK_GUARD_ENABLED:
            print("🛡️ Risk guard disabled: no daily/weekly drawdown limits enforced this cycle.")
        elif not risk_guard_allowed:
            daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0))
            weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0))
            print(f"🛑 Risk guard: blocking new entries (daily DD {daily_dd:.1%}, weekly DD {weekly_dd:.1%}).")
        elif guard_factor < 1.0:
            daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0))
            weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0))
            print(f"🛡️ Risk guard moderation: scaling risk to {guard_factor:.0%} (daily DD {daily_dd:.1%}, weekly DD {weekly_dd:.1%}).")

        if SOFT_GUARD_ENABLED:
            transition = soft_guard_status.get("transition")
            soft_dd = float(soft_guard_status.get("drawdown", 0.0) or 0.0)
            if transition:
                t_type = transition.get("type")
                if t_type == "block":
                    print(f"🩸 Soft equity guard: blocking new entries (DD {soft_dd:.1%} of balance).")
                elif t_type == "resume":
                    print(f"✅ Soft equity guard: trading resumed (DD {soft_dd:.1%} of balance).")
                elif t_type == "throttle":
                    throttle_val = float(transition.get("value", soft_guard_factor))
                    print(f"🩹 Soft equity guard: throttling risk to {throttle_val:.0%} (DD {soft_dd:.1%}).")
            elif not soft_guard_allowed:
                print(f"🩸 Soft equity guard: blocking new entries (DD {soft_dd:.1%} of balance).")
        
        for symbol in symbols:
            cycle_stats['symbols_total'] += 1
            symbol_info = prepare_symbol(symbol)
            if symbol_info is None:
                cycle_stats['symbols_failed_prepare'] += 1
                print(f"Skipping {symbol} because symbol preparation failed.")
                continue
                
            # Check instrument-specific session priority
            if not should_trade_instrument_in_session(symbol):
                session_priority = get_instrument_session_priority(symbol, session_name)
                print(f"📊 {symbol}: Skipping (session priority {session_priority}/{MIN_SESSION_PRIORITY} for {session_name})")
                cycle_stats['symbols_skipped_session'] += 1
                continue
            else:
                session_priority = get_instrument_session_priority(symbol, session_name)
                print(f"📊 {symbol}: Trading allowed (session priority {session_priority}/{MIN_SESSION_PRIORITY} for {session_name})")
            
            # Check for news blackout periods
            is_blackout, news_reason = is_news_blackout_period(symbol)
            if is_blackout:
                print(f"📰 {symbol}: Skipping due to news blackout - {news_reason}")
                cycle_stats['symbols_skipped_news'] += 1
                continue
                
            today = datetime.now()
            from_date = today - timedelta(minutes=bars*15)
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, today)
            if rates is None:
                print(f"No data for {symbol}")
                cycle_stats['symbols_skipped_data'] += 1
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Compute ATR once per symbol for this cycle
            atr_value = compute_atr(df, ATR_PERIOD)
            if atr_value is not None:
                print(f"{symbol} computed ATR({ATR_PERIOD}) = {atr_value:.6f}")
            else:
                print(f"{symbol} ATR could not be computed (insufficient data).")
                cycle_stats['atr_missing'] += 1

            atr_cache_by_period: dict[int, float | None] = {ATR_PERIOD: atr_value}
            volatility_pressure = calculate_volatility_pressure(df)
            candle_conviction = calculate_candle_conviction(df)
            print(f"{symbol} Volatility pressure: {volatility_pressure:.2f} | Candle conviction: {candle_conviction:.2f}")
                
            # Detect market regime for strategy optimization
            current_regime = detect_market_regime(df)
            print(f"📊 {symbol} Market Regime: {current_regime}")
            
            # Advanced stop management for existing positions
            if atr_value:
                manage_position_stops(symbol, atr_value)
            
            positions = mt5.positions_get(symbol=symbol) or []
            if positions:
                pos_details = []
                for p in positions:
                    direction = 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL'
                    pos_details.append(
                        f"{direction} {p.volume:.2f}@{p.price_open:.5f}→{p.price_current:.5f} PnL {getattr(p, 'profit', 0.0):+.2f}"
                    )
                print(f"📈 {symbol} Open positions: {' | '.join(pos_details)}")
                cycle_stats['symbols_with_open_positions'] += 1
            else:
                print(f"📈 {symbol}: no open positions")
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
                    cycle_stats['strategies_disabled_regime'] += 1
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
                    cycle_stats['strategies_skipped_atr'] += 1
                    continue
                
                # Adjust priority based on regime and performance (lower number = higher priority)
                regime_weight = get_regime_strategy_weight(strategy_key, current_regime)
                performance_weight = get_strategy_performance_weight(strategy_key, symbol)
                combined_weight = regime_weight * performance_weight
                regime_adjusted_priority = priority / combined_weight  # Higher weight = better priority
                base_risk_multiplier = calculate_risk_multiplier(session_priority, regime_weight, performance_weight)
                
                signal = agent.get_signal(df)
                if signal not in ('buy', 'sell'):
                    cycle_stats['strategies_no_signal'] += 1
                    print(
                        f"ℹ️ {symbol} {strategy_name}: No actionable signal (raw={signal}) | "
                        f"regime weight {regime_weight:.1%} | performance {performance_weight:.2f}× | adj priority {regime_adjusted_priority:.1f}"
                    )
                    continue
                
                micro_momentum_score = None
                micro_momentum_threshold = None
                micro_soft_pass = False
                micro_override = False
                
                # Multi-timeframe confirmation
                if signal in ('buy', 'sell'):
                    mtf_confirmed = confirm_signal_with_mtf(signal, symbol)
                    mtf_bias = get_mtf_trend_bias(symbol)
                    if not mtf_confirmed:
                        if strategy_key == 'mean_reversion' and current_regime == 'RANGING':
                            print(f"🌀 {symbol} {strategy_name}: Overriding MTF bias ({mtf_bias}) in ranging regime for mean reversion signal {signal}.")
                        else:
                            print(f"🔍 {symbol} {strategy_name}: Signal {signal} rejected by MTF filter (H1 bias: {mtf_bias})")
                            continue  # Skip this signal
                    else:
                        print(f"✅ {symbol} {strategy_name}: Signal {signal} confirmed by MTF (H1 bias: {mtf_bias})")

                    reference_price = float(df['close'].iloc[-1]) if not df['close'].empty else None
                    micro_pass, micro_score, micro_threshold, micro_soft, micro_override = confirm_with_micro_momentum(
                        symbol,
                        signal,
                        current_regime,
                        strategy_atr_value,
                        reference_price,
                        positions,
                    )
                    if not micro_pass:
                        if strategy_key == 'mean_reversion' and current_regime == 'RANGING':
                            required_threshold = micro_threshold if signal == 'buy' else -micro_threshold
                            print(
                                f"🎯 {symbol} {strategy_name}: Micro momentum counter-trend ({micro_score:+.2%} vs required {required_threshold:.2%}); allowing fade entry in range."
                            )
                            micro_momentum_score = None
                            micro_momentum_threshold = None
                            micro_soft_pass = True
                            cycle_stats['micro_overrides'] += 1
                        else:
                            required_threshold = micro_threshold if signal == 'buy' else -micro_threshold
                            comparator = "≥" if signal == 'buy' else "≤"
                            print(
                                f"🎯 {symbol} {strategy_name}: Signal {signal} rejected by micro momentum ({micro_score:+.2%} vs required {comparator} {required_threshold:.2%})"
                            )
                            cycle_stats['signals_filtered_micro'] += 1
                            continue
                    else:
                        micro_momentum_score = micro_score
                        micro_momentum_threshold = micro_threshold
                        micro_soft_pass = micro_soft
                        threshold_display = micro_momentum_threshold
                        comparator = "≥"
                        if signal == 'sell':
                            comparator = "≤"
                            threshold_display = -micro_momentum_threshold
                        qualifiers = []
                        if micro_soft:
                            qualifiers.append("soft confirm")
                        if micro_override:
                            qualifiers.append("override")
                        qualifier_text = f" ({', '.join(qualifiers)})" if qualifiers else ""
                        print(
                            f"🎯 {symbol} {strategy_name}: Micro momentum aligned {micro_score:+.2%} {comparator} {threshold_display:.2%}{qualifier_text}"
                        )
                        if micro_override:
                            cycle_stats['micro_overrides'] += 1
                        cycle_stats['micro_confirms'] += 1
                
                if micro_momentum_score is not None:
                    qualifiers = []
                    if micro_soft_pass:
                        qualifiers.append("soft")
                    if micro_override:
                        qualifiers.append("override")
                    qualifier_suffix = f" ({'/'.join(qualifiers)})" if qualifiers else ""
                    threshold_display = micro_momentum_threshold
                    if signal == 'sell':
                        threshold_display = -micro_momentum_threshold
                    micro_text = (
                        f", micro {micro_momentum_score:+.2%}/{threshold_display:.2%}{qualifier_suffix}"
                    )
                else:
                    micro_text = ""
                confidence = calculate_signal_confidence(
                    session_priority,
                    regime_weight,
                    performance_weight,
                    combined_guard_factor,
                    signal,
                    micro_momentum_score,
                    micro_momentum_threshold,
                    micro_soft_pass,
                    volatility_pressure,
                    candle_conviction,
                )

                adjusted_risk_multiplier = adjust_risk_with_confidence(base_risk_multiplier, confidence)
                risk_multiplier = apply_risk_guard_to_multiplier(adjusted_risk_multiplier)

                if signal in ('buy', 'sell') and confidence < CONFIDENCE_EXECUTION_THRESHOLD:
                    print(f"⚖️ {symbol} {strategy_name}: Confidence {confidence:.2f} below threshold {CONFIDENCE_EXECUTION_THRESHOLD:.2f}; skipping signal {signal}.")
                    cycle_stats['signals_filtered_confidence'] += 1
                    signal = None

                print(
                    f"{symbol} {strategy_name} Final Signal: {signal} (regime: {regime_weight:.1%}, performance: {performance_weight:.2f}×, adj priority: {regime_adjusted_priority:.1f}, confidence {confidence:.2f}, risk base x{base_risk_multiplier:.2f} → x{risk_multiplier:.2f}, ATR({atr_period_override})={strategy_atr_value:.6f}{micro_text})"
                )

                if signal in ('buy', 'sell'):
                    cycle_stats['candidates_total'] += 1
                    all_candidates.append({
                        'signal': signal,
                        'label': strategy_name,
                        'priority': regime_adjusted_priority,  # Use regime-adjusted priority
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult,
                        'regime_weight': regime_weight,
                        'base_risk_multiplier': base_risk_multiplier,
                        'risk_multiplier': risk_multiplier,
                        'micro_momentum': micro_momentum_score,
                        'micro_threshold': micro_momentum_threshold,
                        'micro_soft_pass': micro_soft_pass,
                        'atr_value': strategy_atr_value,
                        'atr_period': atr_period_override,
                        'confidence': confidence,
                    })

            if all_candidates:
                if not guard_trading_allowed:
                    cycle_stats['symbols_blocked_guard'] += 1
                    cycle_stats['signals_skipped_guard'] += len(all_candidates)
                    if not risk_guard_allowed and not soft_guard_allowed:
                        reason = "risk + soft guard"
                    elif not risk_guard_allowed:
                        reason = "risk guard"
                    else:
                        reason = "soft guard"
                    print(f"{symbol}: Guard active ({reason}), skipping {len(all_candidates)} candidate signals this cycle.")
                    continue
                # Pick the highest confidence signal, tie-breaker by priority
                best_candidate = max(all_candidates, key=lambda c: (c['confidence'], -c['priority']))
                signal_type = best_candidate['signal']
                
                if signal_type == 'buy':
                    if have_sell and not ALLOW_HEDGING:
                        print(f"{symbol}: skipping {best_candidate['label']} buy (priority {best_candidate['priority']}) because opposite position is open")
                        cycle_stats['signals_skipped_position_conflict'] += 1
                    elif have_buy:
                        allow_stack, risk_scale, rationale = evaluate_pyramiding(
                            symbol,
                            'buy',
                            positions,
                            best_candidate['confidence'],
                            strategy_atr_value,
                        )
                        if allow_stack:
                            stacked_multiplier = min(
                                RISK_MULTIPLIER_MAX,
                                best_candidate['risk_multiplier'] * risk_scale,
                            )
                            order_comment = build_order_comment(best_candidate['label'], 'buy', best_candidate['priority'])
                            print(
                                f"{symbol}: pyramiding BUY via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | {rationale} | comment {order_comment}"
                            )
                            cycle_stats['signals_executed'] += 1
                            cycle_stats['signals_executed_pyramid'] += 1
                            send_order(
                                symbol,
                                'buy',
                                comment=order_comment,
                                risk_multiplier=stacked_multiplier,
                                atr=best_candidate['atr_value'],
                                sl_mult=best_candidate['sl_mult'],
                                tp_mult=best_candidate['tp_mult'],
                            )
                        else:
                            print(f"{symbol}: already in buy position, ignoring {best_candidate['label']} (priority {best_candidate['priority']})")
                            cycle_stats['signals_skipped_duplicate_side'] += 1
                    else:
                        order_comment = build_order_comment(best_candidate['label'], 'buy', best_candidate['priority'])
                        print(f"{symbol}: executing BUY via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | comment {order_comment}")
                        cycle_stats['signals_executed'] += 1
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
                        cycle_stats['signals_skipped_position_conflict'] += 1
                    elif have_sell:
                        allow_stack, risk_scale, rationale = evaluate_pyramiding(
                            symbol,
                            'sell',
                            positions,
                            best_candidate['confidence'],
                            strategy_atr_value,
                        )
                        if allow_stack:
                            stacked_multiplier = min(
                                RISK_MULTIPLIER_MAX,
                                best_candidate['risk_multiplier'] * risk_scale,
                            )
                            order_comment = build_order_comment(best_candidate['label'], 'sell', best_candidate['priority'])
                            print(
                                f"{symbol}: pyramiding SELL via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | {rationale} | comment {order_comment}"
                            )
                            cycle_stats['signals_executed'] += 1
                            cycle_stats['signals_executed_pyramid'] += 1
                            send_order(
                                symbol,
                                'sell',
                                comment=order_comment,
                                risk_multiplier=stacked_multiplier,
                                atr=best_candidate['atr_value'],
                                sl_mult=best_candidate['sl_mult'],
                                tp_mult=best_candidate['tp_mult'],
                            )
                        else:
                            print(f"{symbol}: already in sell position, ignoring {best_candidate['label']} (priority {best_candidate['priority']})")
                            cycle_stats['signals_skipped_duplicate_side'] += 1
                    else:
                        order_comment = build_order_comment(best_candidate['label'], 'sell', best_candidate['priority'])
                        print(f"{symbol}: executing SELL via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | comment {order_comment}")
                        cycle_stats['signals_executed'] += 1
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
                    print(
                        f"{symbol}: skipping {candidate['label']} {candidate['signal']} (conf {candidate['confidence']:.2f}, priority {candidate['priority']}) "
                        f"- beaten by {best_candidate['label']} (conf {best_candidate['confidence']:.2f}, priority {best_candidate['priority']})"
                    )
        print(
            f"📒 Scan #{scan_counter} summary -> "
            f"symbols processed {cycle_stats['symbols_total']}/{len(symbols)} | "
            f"open-symbols {cycle_stats['symbols_with_open_positions']} | "
            f"candidates {cycle_stats['candidates_total']} | executed {cycle_stats['signals_executed']} | pyramids {cycle_stats['signals_executed_pyramid']} | "
            f"confidence-filtered {cycle_stats['signals_filtered_confidence']} | micro-filtered {cycle_stats['signals_filtered_micro']} | "
            f"position-blocked {cycle_stats['signals_skipped_position_conflict']} | duplicates {cycle_stats['signals_skipped_duplicate_side']} | guard-blocked {cycle_stats['signals_skipped_guard']}"
        )
        print(
            "📊 Strategy diagnostics -> "
            f"disabled {cycle_stats['strategies_disabled_regime']} | no-signal {cycle_stats['strategies_no_signal']} | "
            f"ATR-missing {cycle_stats['strategies_skipped_atr']} | micro-overrides {cycle_stats['micro_overrides']} | micro-confirms {cycle_stats['micro_confirms']}"
        )
        print(f"⏱️ Next scan in {SCAN_INTERVAL_SECONDS} seconds...")
        time.sleep(SCAN_INTERVAL_SECONDS)
finally:
    mt5.shutdown()
