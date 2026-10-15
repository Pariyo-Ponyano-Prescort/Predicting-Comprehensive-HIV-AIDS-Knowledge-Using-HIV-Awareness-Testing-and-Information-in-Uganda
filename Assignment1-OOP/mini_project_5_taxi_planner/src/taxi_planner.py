"""Simple tools for taxi route revenue, forecasting, and fleet planning."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve


class Route:
    """Passenger demand and fare for one taxi route."""

    def __init__(self, name: str, passengers, fare: float):
        """Create a route; passengers must be a non-empty 1-D array."""
        values = np.asarray(passengers, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("passengers must be a non-empty 1-D array")
        if np.any(values < 0) or fare < 0:
            raise ValueError("passengers and fare cannot be negative")
        self.name = name
        self.passengers = values
        self.fare = float(fare)

    def daily_revenue(self) -> np.ndarray:
        """Revenue for each observed day, in UGX."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        """Total revenue across all observed days, in UGX."""
        return float(np.sum(self.daily_revenue()))

    def statistics(self) -> dict:
        """Return mean, population variance, and population standard deviation."""
        return {
            "mean": float(np.mean(self.passengers)),
            "variance": float(np.var(self.passengers)),
            "std_dev": float(np.std(self.passengers)),
        }


class Forecaster(ABC):
    """Base class for a small forecasting method."""

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast the next `horizon` values from the supplied history."""


class MovingAverage(Forecaster):
    """Forecast using the mean of the latest three observations."""

    def predict(self, history: np.ndarray, horizon: int = 1) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if len(history) == 0 or horizon < 1:
            raise ValueError("history must be non-empty and horizon >= 1")
        value = float(np.mean(history[-3:]))
        return np.repeat(value, horizon)


class ExponentialSmoothing(Forecaster):
    """Simple exponential smoothing with a user-selected alpha."""

    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def predict(self, history: np.ndarray, horizon: int = 1) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if len(history) == 0 or horizon < 1:
            raise ValueError("history must be non-empty and horizon >= 1")
        level = float(history[0])
        for value in history[1:]:
            level = self.alpha * float(value) + (1 - self.alpha) * level
        return np.repeat(level, horizon)


class LinearTrend(Forecaster):
    """Fit a straight line to observed values and extend it forward."""

    def predict(self, history: np.ndarray, horizon: int = 1) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if len(history) == 0 or horizon < 1:
            raise ValueError("history must be non-empty and horizon >= 1")
        if len(history) == 1:
            return np.repeat(float(history[0]), horizon)
        x = np.arange(len(history), dtype=float)
        slope, intercept = np.polyfit(x, history, 1)
        future_x = np.arange(len(history), len(history) + horizon, dtype=float)
        return slope * future_x + intercept


class SeasonalNaive(Forecaster):
    """Repeat the value from one seasonal cycle ago (weekly period = 7)."""

    def __init__(self, period: int = 7):
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period

    def predict(self, history: np.ndarray, horizon: int = 1) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if len(history) < self.period or horizon < 1:
            raise ValueError("history must contain at least one full season")
        return np.array([history[-self.period + (i % self.period)] for i in range(horizon)])


def equilibrium() -> tuple[float, float]:
    """Solve Qd = 120 - 0.02P and Qs = 10 + 0.03P."""
    # 0.02P + Q = 120; -0.03P + Q = 10
    matrix = np.array([[0.02, 1.0], [-0.03, 1.0]])
    rhs = np.array([120.0, 10.0])
    price, quantity = solve(matrix, rhs)
    return float(price), float(quantity)


def mae(actual, predicted) -> float:
    """Mean absolute error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal non-zero lengths")
    return float(np.mean(np.abs(actual - predicted)))


def backtest(passengers, forecaster: Forecaster, first_test_index: int = 3) -> float:
    """Walk-forward MAE: predict each test point using only earlier observations."""
    values = np.asarray(passengers, dtype=float)
    errors = []
    for i in range(first_test_index, len(values)):
        pred = forecaster.predict(values[:i])[0]
        errors.append(abs(values[i] - pred))
    if not errors:
        raise ValueError("not enough data for backtesting")
    return float(np.mean(errors))


def tune_alpha(passengers, alphas=None) -> tuple[float, float]:
    """Pick alpha with the smallest walk-forward MAE."""
    if alphas is None:
        alphas = np.arange(0.1, 1.0, 0.1)
    scores = [(float(a), backtest(passengers, ExponentialSmoothing(float(a)))) for a in alphas]
    return min(scores, key=lambda item: item[1])


def fleet_needed(predicted_passengers: float, capacity: int = 14,
                 trips_per_vehicle: int = 8, buffer: float = 0.15) -> int:
    """Vehicles needed for forecast passengers, allowing a demand buffer."""
    if predicted_passengers < 0 or capacity <= 0 or trips_per_vehicle <= 0 or buffer < 0:
        raise ValueError("invalid fleet planning input")
    total_capacity = capacity * trips_per_vehicle
    return int(np.ceil(predicted_passengers * (1 + buffer) / total_capacity))
