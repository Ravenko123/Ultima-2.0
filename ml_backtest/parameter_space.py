"""Parameter search primitives used by the ML backtester."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

__all__ = [
    "ParameterSample",
    "BaseParameter",
    "ContinuousParameter",
    "DiscreteParameter",
    "CategoricalParameter",
    "ParameterSpace",
]


@dataclass(slots=True, frozen=True)
class ParameterSample:
    """A single sampled parameter assignment."""

    values: Mapping[str, object]

    def as_dict(self) -> Dict[str, object]:
        return dict(self.values)

    def __getitem__(self, item: str) -> object:
        return self.values[item]


@dataclass(slots=True)
class BaseParameter:
    name: str

    def sample(self, rng: random.Random) -> object:
        raise NotImplementedError


@dataclass(slots=True)
class ContinuousParameter(BaseParameter):
    lower: float
    upper: float

    def sample(self, rng: random.Random) -> float:
        if self.lower >= self.upper:
            raise ValueError(f"Continuous parameter '{self.name}' has invalid bounds")
        return rng.uniform(self.lower, self.upper)


@dataclass(slots=True)
class DiscreteParameter(BaseParameter):
    values: Sequence[float | int]

    def sample(self, rng: random.Random) -> float | int:
        if not self.values:
            raise ValueError(f"Discrete parameter '{self.name}' has no choices")
        return rng.choice(list(self.values))


@dataclass(slots=True)
class CategoricalParameter(BaseParameter):
    choices: Sequence[object]

    def sample(self, rng: random.Random) -> object:
        if not self.choices:
            raise ValueError(f"Categorical parameter '{self.name}' has no choices")
        return rng.choice(list(self.choices))


class ParameterSpace:
    """Container for parameter definitions that can emit random samples."""

    def __init__(self, parameters: Iterable[BaseParameter]) -> None:
        seen: MutableMapping[str, BaseParameter] = {}
        for param in parameters:
            if param.name in seen:
                raise ValueError(f"Duplicate parameter name detected: {param.name}")
            seen[param.name] = param
        self._parameters: Mapping[str, BaseParameter] = dict(seen)

    def sample(self, *, rng: random.Random | None = None) -> ParameterSample:
        rng = rng or random.Random()
        assignment = {name: parameter.sample(rng) for name, parameter in self._parameters.items()}
        return ParameterSample(values=assignment)

    def iter_samples(self, count: int, *, rng: random.Random | None = None) -> Iterator[ParameterSample]:
        rng = rng or random.Random()
        for _ in range(count):
            yield self.sample(rng=rng)

    @property
    def names(self) -> Sequence[str]:
        return tuple(self._parameters.keys())

    def __len__(self) -> int:
        return len(self._parameters)
