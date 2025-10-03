"""Risk and persona presets for the ML backtester."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

__all__ = [
    "RiskProfile",
    "PersonaProfile",
    "DEFAULT_RISK_PROFILES",
    "DEFAULT_PERSONA_PROFILES",
    "resolve_risk_profile",
    "resolve_persona_profile",
    "profile_to_dict",
]


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Static risk adjustments applied to simulator parameters."""

    name: str
    volume_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    max_drawdown_pct: float | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class PersonaProfile:
    """Behavioural tuning controls for personas."""

    name: str
    fast_window_bias: int = 0
    slow_window_bias: int = 0
    atr_window_bias: int = 0
    take_profit_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    volume_multiplier: float = 1.0
    max_holding_multiplier: float = 1.0
    description: str | None = None


NEUTRAL_RISK_PROFILE = RiskProfile(name="neutral")
NEUTRAL_PERSONA_PROFILE = PersonaProfile(name="neutral")


DEFAULT_RISK_PROFILES: Mapping[str, RiskProfile] = {
    profile.name: profile
    for profile in (
        NEUTRAL_RISK_PROFILE,
        RiskProfile(
            name="conservative",
            volume_multiplier=0.5,
            take_profit_multiplier=0.85,
            stop_loss_multiplier=0.9,
            spread_bps=1.5,
            slippage_bps=1.0,
            max_drawdown_pct=0.1,
            description="Capital preservation first; tighter stops, smaller size, ~10% drawdown cap.",
        ),
        RiskProfile(
            name="balanced",
            volume_multiplier=0.9,
            take_profit_multiplier=1.05,
            stop_loss_multiplier=1.05,
            spread_bps=0.8,
            slippage_bps=0.8,
            max_drawdown_pct=0.2,
            description="Default balanced profile aimed at steady compounding with ~20% drawdown room.",
        ),
        RiskProfile(
            name="aggressive",
            volume_multiplier=1.3,
            take_profit_multiplier=1.25,
            stop_loss_multiplier=1.2,
            spread_bps=0.5,
            slippage_bps=0.3,
            max_drawdown_pct=0.4,
            description="Swing hard for growth with wider brackets; tolerates ~40% drawdown.",
        ),
        RiskProfile(
            name="ultra",
            volume_multiplier=1.6,
            take_profit_multiplier=1.45,
            stop_loss_multiplier=1.3,
            spread_bps=0.3,
            slippage_bps=0.2,
            max_drawdown_pct=0.60,
            description="Max throttle for stress testing/frontier runs with ~60% drawdown tolerance.",
        ),
    )
}

RISK_ALIASES: Mapping[str, str] = {
    "default": "balanced",
    "moderate": "balanced",
    "medium": "balanced",
    "low": "conservative",
    "conservative": "conservative",
    "high": "aggressive",
    "aggressive": "aggressive",
    "max": "ultra",
    "ultra": "ultra",
}

DEFAULT_PERSONA_PROFILES: Mapping[str, PersonaProfile] = {
    profile.name: profile
    for profile in (
        NEUTRAL_PERSONA_PROFILE,
        PersonaProfile(
            name="scalper",
            fast_window_bias=-4,
            slow_window_bias=-6,
            atr_window_bias=-3,
            take_profit_multiplier=0.75,
            stop_loss_multiplier=0.8,
            volume_multiplier=0.9,
            max_holding_multiplier=0.35,
            description="Ultra short-term persona favouring quick exits.",
        ),
        PersonaProfile(
            name="swing",
            fast_window_bias=2,
            slow_window_bias=6,
            atr_window_bias=2,
            take_profit_multiplier=1.2,
            stop_loss_multiplier=1.15,
            volume_multiplier=1.0,
            max_holding_multiplier=1.6,
            description="Captures multi-session moves with wider brackets.",
        ),
        PersonaProfile(
            name="position",
            fast_window_bias=4,
            slow_window_bias=12,
            atr_window_bias=4,
            take_profit_multiplier=1.5,
            stop_loss_multiplier=1.35,
            volume_multiplier=0.8,
            max_holding_multiplier=2.4,
            description="Long horizon persona letting trends mature.",
        ),
        PersonaProfile(
            name="mean_reverter",
            fast_window_bias=-2,
            slow_window_bias=-2,
            atr_window_bias=1,
            take_profit_multiplier=0.9,
            stop_loss_multiplier=0.95,
            volume_multiplier=0.95,
            max_holding_multiplier=0.8,
            description="Seeks fading extremes with tighter envelopes.",
        ),
    )
}

PERSONA_ALIASES: Mapping[str, str] = {
    "default": "neutral",
    "base": "neutral",
    "intraday": "scalper",
    "daytrader": "scalper",
    "swing": "swing",
    "swing_trader": "swing",
    "position": "position",
    "investor": "position",
    "mean": "mean_reverter",
    "mean_reversion": "mean_reverter",
}


def resolve_risk_profile(name: str | None) -> RiskProfile:
    """Return the configured risk profile, falling back to a neutral stance."""

    if not name:
        return NEUTRAL_RISK_PROFILE
    key = name.lower().strip()
    if key in DEFAULT_RISK_PROFILES:
        return DEFAULT_RISK_PROFILES[key]
    alias = RISK_ALIASES.get(key)
    if alias and alias in DEFAULT_RISK_PROFILES:
        return DEFAULT_RISK_PROFILES[alias]
    return RiskProfile(name=name)


def resolve_persona_profile(name: str | None) -> PersonaProfile:
    """Return persona tuning metadata, defaulting to a neutral profile."""

    if not name:
        return NEUTRAL_PERSONA_PROFILE
    key = name.lower().strip()
    if key in DEFAULT_PERSONA_PROFILES:
        return DEFAULT_PERSONA_PROFILES[key]
    alias = PERSONA_ALIASES.get(key)
    if alias and alias in DEFAULT_PERSONA_PROFILES:
        return DEFAULT_PERSONA_PROFILES[alias]
    return PersonaProfile(name=name)


def profile_to_dict(profile: RiskProfile | PersonaProfile) -> dict[str, object]:
    """Convert a profile dataclass into a JSON-friendly mapping."""

    return asdict(profile)
