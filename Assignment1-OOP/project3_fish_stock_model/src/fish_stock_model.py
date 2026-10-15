"""Fish stock, price and revenue-risk models."""

from dataclasses import dataclass
import statistics
from typing import Iterable

import numpy as np


@dataclass
class FishStock:
    """Model weekly fish biomass using logistic growth and harvesting."""

    initial_stock: float = 4000.0
    growth_rate: float = 0.4
    carrying_capacity: float = 10000.0
    harvest_rate: float = 0.10

    def __post_init__(self) -> None:
        if self.initial_stock < 0:
            raise ValueError("Initial stock cannot be negative.")
        if self.growth_rate < 0:
            raise ValueError("Growth rate cannot be negative.")
        if self.carrying_capacity <= 0:
            raise ValueError("Carrying capacity must be positive.")
        if not 0 <= self.harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1.")

    def simulate(
        self,
        weeks: int = 52,
        closed_season_weeks: Iterable[int] = (),
    ) -> np.ndarray:
        """Return weekly stock, including the initial stock at week zero."""
        if weeks < 0:
            raise ValueError("Weeks cannot be negative.")

        closed = set(closed_season_weeks)
        stock = np.zeros(weeks + 1, dtype=float)
        stock[0] = self.initial_stock

        for t in range(weeks):
            n = stock[t]
            harvest = 0.0 if t in closed else self.harvest_rate * n

            stock[t + 1] = max(
                0.0,
                n + self.growth_rate * n
                * (1 - n / self.carrying_capacity) - harvest,
            )

        return stock

    def weekly_harvest_tonnes(
        self,
        stock: np.ndarray,
        closed_season_weeks: Iterable[int] = (),
    ) -> np.ndarray:
        """Calculate the catch in tonnes for each simulated week."""
        closed = set(closed_season_weeks)

        return np.array([
            0.0 if t in closed else self.harvest_rate * stock[t]
            for t in range(len(stock) - 1)
        ])


@dataclass
class PriceModel:
    """Generate reproducible weekly prices using a bounded random walk."""

    start_price: float = 12000.0
    lower_bound: float = 9000.0
    upper_bound: float = 16000.0
    weekly_shock_sd: float = 450.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.lower_bound <= 0:
            raise ValueError("Lower price bound must be positive.")
        if self.upper_bound < self.lower_bound:
            raise ValueError("Upper bound must exceed lower bound.")
        if not self.lower_bound <= self.start_price <= self.upper_bound:
            raise ValueError("Starting price must be within the bounds.")
        if self.weekly_shock_sd < 0:
            raise ValueError("Price volatility cannot be negative.")

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Return weekly prices in UGX per kilogram."""
        if weeks < 0:
            raise ValueError("Weeks cannot be negative.")

        rng = np.random.default_rng(self.seed)
        prices = np.empty(weeks, dtype=float)

        if weeks == 0:
            return prices

        prices[0] = self.start_price

        for t in range(1, weeks):
            change = rng.normal(0, self.weekly_shock_sd)
            prices[t] = np.clip(
                prices[t - 1] + change,
                self.lower_bound,
                self.upper_bound,
            )

        return prices


@dataclass
class RiskAssessor:
    """Classify revenue risk using its coefficient of variation."""

    low_max: float = 0.10
    moderate_max: float = 0.20

    def __post_init__(self) -> None:
        if not 0 <= self.low_max < self.moderate_max:
            raise ValueError("Risk thresholds are invalid.")

    def assess(self, revenues: list[float]) -> tuple[dict, str]:
        """Calculate descriptive statistics and assign a risk class."""
        values = [float(x) for x in revenues]

        if not values:
            raise ValueError("Revenue data cannot be empty.")
        if any(not np.isfinite(x) or x < 0 for x in values):
            raise ValueError("Revenue values must be finite and non-negative.")

        mean = statistics.mean(values)
        median = statistics.median(values)
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = sd / mean if mean > 0 else 0.0

        summary = {
            "mean": mean,
            "median": median,
            "variance": variance,
            "standard_deviation": sd,
            "cv": cv,
        }

        if cv < self.low_max:
            risk = "Low"
        elif cv <= self.moderate_max:
            risk = "Moderate"
        else:
            risk = "High"

        return summary, risk


def fibonacci_baseline(count: int = 15) -> list[int]:
    """Generate Fibonacci numbers for the unrealistic baseline."""
    if count < 0:
        raise ValueError("Count cannot be negative.")

    result = []
    a, b = 0, 1

    for _ in range(count):
        result.append(a)
        a, b = b, a + b

    return result