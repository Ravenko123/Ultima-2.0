"""
Helper to load and query backtesting best configuration for live trading.

This reads the bestconfig.json produced by the backtester and exposes simple
functions to fetch per-symbol, per-strategy parameters and metadata.

Typical usage:

    from bestconfig_loader import load_bestconfig, find_bestconfig, get_params_for

    path = find_bestconfig()
    cfg = load_bestconfig(path)
    params = get_params_for(cfg, symbol="XAUUSD", strategy="ma_crossover")
    if params:
        strat_params = params.get("strategy_params", {})
        atr = params.get("atr", {})
        lots = params.get("risk", {}).get("lots", 0.1)
        # Use strat_params to configure your live agent
        # Use atr settings to set SL/TP logic if desired
"""
from __future__ import annotations
import json
import os
from typing import Any, Optional, Dict

DEFAULT_NAMES = [
    "bestconfig.json",
]

RELATIVE_TRY_PATHS = [
    # When called from live/, backtester writes into ../backtesting/bestconfig.json
    os.path.join("..", "backtesting", "bestconfig.json"),
    os.path.join(".", "bestconfig.json"),
]


def find_bestconfig(explicit_path: Optional[str] = None) -> Optional[str]:
    """Return a plausible path to bestconfig.json if it exists, else None."""
    if explicit_path:
        p = os.path.abspath(explicit_path)
        return p if os.path.exists(p) else None
    # Try relative candidates
    here = os.path.dirname(__file__)
    for rel in RELATIVE_TRY_PATHS + DEFAULT_NAMES:
        p = os.path.abspath(os.path.join(here, rel))
        if os.path.exists(p):
            return p
    return None


def load_bestconfig(path: Optional[str] = None) -> Dict[str, Any]:
    """Load bestconfig JSON. If path is None, try to locate it automatically."""
    if path is None:
        path = find_bestconfig()
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_symbol_config(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Return the config entry for a given symbol, or empty dict."""
    try:
        return (cfg or {}).get("symbols", {}).get(symbol, {}) or {}
    except Exception:
        return {}


def get_params_for(cfg: Dict[str, Any], symbol: str, strategy: str) -> Optional[Dict[str, Any]]:
    """Return merged parameters for a symbol+strategy or None if missing.

    Structure:
    {
      'strategy_params': {...},
      'atr': {'period': int, 'sl_mult': float, 'tp_mult': float, 'priority': str},
      'trading_window': {'trade_24_7': bool, 'start_hour': int, 'end_hour': int},
      'risk': {'lots': float},
      'timeframe': str,              # e.g., '15M' or MT5 numeric
      'timeframe_code': int,         # MT5 timeframe code
      'days': int,
      'performance': {...},          # optional, if present in file
      'optimized': {...},            # optional, if present in file
      'updated_at': str
    }
    """
    try:
        sym = get_symbol_config(cfg, symbol)
        if strategy not in sym:
            return None
        return sym[strategy]
    except Exception:
        return None


def available_symbols(cfg: Dict[str, Any]) -> list[str]:
    try:
        return sorted(list((cfg or {}).get("symbols", {}).keys()))
    except Exception:
        return []


def available_strategies(cfg: Dict[str, Any], symbol: str, active_only: bool = False) -> list[str]:
    try:
        sym = get_symbol_config(cfg, symbol)
        if not active_only:
            return sorted(list(sym.keys()))
        actives = []
        for k, v in sym.items():
            try:
                if bool((v or {}).get('active', True)):
                    actives.append(k)
            except Exception:
                actives.append(k)
        return sorted(actives)
    except Exception:
        return []
