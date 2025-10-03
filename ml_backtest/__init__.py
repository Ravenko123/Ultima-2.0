"""Next-generation machine-learning driven backtesting framework for Ultima."""

from .config import (
    DataCacheConfig,
    ExperimentTimeRange,
    ExperimentConfig,
    TelemetryOutputConfig,
)
from .data_loader import MarketDataCache, load_time_series
from .parameter_space import (
    BaseParameter,
    CategoricalParameter,
    ContinuousParameter,
    DiscreteParameter,
    ParameterSpace,
    ParameterSample,
)
from .experiment_runner import ExperimentRunner
from .optimizer import ExperimentOptimizer, OptimizationSettings, OptimizationSummary
from .results import ExperimentArtifact, ExperimentResult, TrialScorecard
from .risk import PersonaProfile, RiskProfile, resolve_persona_profile, resolve_risk_profile

__all__ = [
    "DataCacheConfig",
    "ExperimentTimeRange",
    "ExperimentConfig",
    "TelemetryOutputConfig",
    "MarketDataCache",
    "load_time_series",
    "BaseParameter",
    "CategoricalParameter",
    "ContinuousParameter",
    "DiscreteParameter",
    "ParameterSpace",
    "ParameterSample",
    "ExperimentRunner",
    "ExperimentOptimizer",
    "OptimizationSettings",
    "OptimizationSummary",
    "ExperimentArtifact",
    "ExperimentResult",
    "TrialScorecard",
    "RiskProfile",
    "PersonaProfile",
    "resolve_risk_profile",
    "resolve_persona_profile",
]
