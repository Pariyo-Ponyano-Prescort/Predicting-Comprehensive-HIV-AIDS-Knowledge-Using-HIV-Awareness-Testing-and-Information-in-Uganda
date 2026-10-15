"""Reusable tools for the UBOS district population forecasting mini-project.

Population values are expressed in thousands of people, matching the assignment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DistrictPopulation:
    """Store and validate one district's annual population estimates.

    Args:
        name: District name.
        years: Annual years, in chronological order.
        populations: Population estimates in thousands of people.
    """
    name: str
    years: np.ndarray
    populations: np.ndarray

    def __post_init__(self) -> None:
        self.years = np.asarray(self.years, dtype=int)
        self.populations = np.asarray(self.populations, dtype=float)
        if self.years.ndim != 1 or self.populations.ndim != 1:
            raise ValueError("years and populations must be one-dimensional.")
        if len(self.years) != len(self.populations):
            raise ValueError("years and populations must have equal lengths.")
        if len(self.years) == 0:
            raise ValueError("a district must contain at least one observation.")
        if not np.all(np.isfinite(self.populations)):
            raise ValueError("populations must be finite numbers.")
        if np.any(self.populations < 0):
            raise ValueError("populations cannot be negative.")
        if len(np.unique(self.years)) != len(self.years):
            raise ValueError("years must be unique.")
        if np.any(np.diff(self.years) <= 0):
            raise ValueError("years must be in strictly increasing order.")

    def __repr__(self) -> str:
        return (f"DistrictPopulation(name={self.name!r}, "
                f"years={self.years[0]}–{self.years[-1]}, "
                f"observations={len(self)})")

    def __len__(self) -> int:
        return len(self.populations)


class Forecaster(ABC):
    """Abstract interface shared by all forecasting models."""

    def __init__(self) -> None:
        self.years_: np.ndarray | None = None
        self.values_: np.ndarray | None = None
        self.fitted_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None

    @abstractmethod
    def fit(self, years: Iterable[int], values: Iterable[float]) -> "Forecaster":
        """Fit the model to year/value observations."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Predict the next `horizon` annual observations."""
        raise NotImplementedError

    def _validate_and_store(self, years: Iterable[int], values: Iterable[float]) -> None:
        x = np.asarray(list(years), dtype=float)
        y = np.asarray(list(values), dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("years and values must be equal-length one-dimensional data.")
        if len(x) < 2:
            raise ValueError("at least two observations are required.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training data must contain only finite values.")
        if np.any(y < 0):
            raise ValueError("population values cannot be negative.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("years must be strictly increasing.")
        self.years_ = x
        self.values_ = y

    @staticmethod
    def _check_horizon(horizon: int) -> None:
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")


class LinearTrendForecaster(Forecaster):
    """Fit a straight-line trend using NumPy's least-squares polynomial fit."""

    def fit(self, years: Iterable[int], values: Iterable[float]) -> "LinearTrendForecaster":
        self._validate_and_store(years, values)
        assert self.years_ is not None and self.values_ is not None
        self.coefficients_ = np.polyfit(self.years_, self.values_, deg=1)
        self.fitted_ = np.polyval(self.coefficients_, self.years_)
        self.residuals_ = self.values_ - self.fitted_
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        if self.years_ is None:
            raise RuntimeError("Call fit() before predict().")
        future_years = np.arange(self.years_[-1] + 1, self.years_[-1] + horizon + 1)
        return np.maximum(0.0, np.polyval(self.coefficients_, future_years))


class CAGRForecaster(Forecaster):
    """Forecast using a constant compound annual growth rate."""

    def fit(self, years: Iterable[int], values: Iterable[float]) -> "CAGRForecaster":
        self._validate_and_store(years, values)
        assert self.years_ is not None and self.values_ is not None
        first, last = self.values_[0], self.values_[-1]
        elapsed = self.years_[-1] - self.years_[0]
        if first <= 0 or last <= 0:
            raise ValueError("CAGR requires strictly positive first and last populations.")
        self.rate_ = (last / first) ** (1.0 / elapsed) - 1.0
        self.fitted_ = first * (1.0 + self.rate_) ** (self.years_ - self.years_[0])
        self.residuals_ = self.values_ - self.fitted_
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        if self.values_ is None:
            raise RuntimeError("Call fit() before predict().")
        future_steps = np.arange(1, horizon + 1)
        return self.values_[-1] * (1.0 + self.rate_) ** future_steps


class FibonacciRatioForecaster(Forecaster):
    """Naive benchmark using successive Fibonacci-number ratios as multipliers.

    Starting from the last observed value, each future year is multiplied by
    F(n+1)/F(n), using ratios 3/2, 5/3, 8/5, ... . These approach the golden
    ratio, so this is a deliberately aggressive benchmark rather than a
    demographically grounded assumption.
    """

    def fit(self, years: Iterable[int], values: Iterable[float]) -> "FibonacciRatioForecaster":
        self._validate_and_store(years, values)
        assert self.years_ is not None and self.values_ is not None
        self.fitted_ = np.full_like(self.values_, np.nan, dtype=float)
        # For training-fit residuals, use one-step Fibonacci-ratio predictions.
        for i in range(1, len(self.values_)):
            a, b = self._fib_pair(i + 3)
            self.fitted_[i] = self.values_[i - 1] * (b / a)
        self.residuals_ = self.values_[1:] - self.fitted_[1:]
        return self

    @staticmethod
    def _fib_pair(n: int) -> tuple[int, int]:
        seq = [0, 1]
        while len(seq) <= n:
            seq.append(seq[-1] + seq[-2])
        return seq[n - 1], seq[n]

    def predict(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        if self.values_ is None:
            raise RuntimeError("Call fit() before predict().")
        current = float(self.values_[-1])
        output = []
        for step in range(1, horizon + 1):
            a, b = self._fib_pair(step + 3)
            current *= b / a
            output.append(current)
        return np.asarray(output)


def calculate_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE; MAPE excludes zero actual values."""
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("actual and predicted must be non-empty equal-length vectors.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("actual and predicted values must be finite.")
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    nonzero = y != 0
    mape = float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])) * 100) if np.any(nonzero) else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def bootstrap_prediction_interval(
    model: Forecaster,
    horizon: int,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create residual-bootstrap forecast paths and return point, lower, upper.

    Residuals are sampled with replacement and accumulated over future years.
    This is a simple empirical interval, not a full demographic uncertainty model.
    """
    if n_resamples < 1000:
        raise ValueError("Use at least 1,000 bootstrap resamples.")
    point = model.predict(horizon)
    residuals = np.asarray(model.residuals_, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 2:
        residuals = np.array([0.0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(residuals, size=(n_resamples, horizon), replace=True)
    paths = point[None, :] + np.cumsum(sampled, axis=1)
    lower = np.maximum(0.0, np.percentile(paths, 2.5, axis=0))
    upper = np.percentile(paths, 97.5, axis=0)
    return point, lower, upper
