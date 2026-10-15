import numpy as np
import pytest
from src.taxi_planner import (
    Route, MovingAverage, ExponentialSmoothing, LinearTrend,
    SeasonalNaive, equilibrium, mae, fleet_needed, backtest
)

def test_route_revenue_and_stats():
    route = Route("test", [10, 20], 2000)
    assert route.total_revenue() == 60000
    assert route.daily_revenue().tolist() == [20000, 40000]
    assert route.statistics()["mean"] == 15

def test_route_rejects_empty_passengers():
    with pytest.raises(ValueError):
        Route("empty", [], 1000)

def test_forecasters():
    history = np.array([1, 2, 3, 4], dtype=float)
    assert MovingAverage().predict(history)[0] == 3
    assert ExponentialSmoothing(1).predict(history)[0] == 4
    assert LinearTrend().predict(history)[0] == pytest.approx(5)

def test_bad_alpha_and_fleet_inputs():
    with pytest.raises(ValueError):
        ExponentialSmoothing(0)
    with pytest.raises(ValueError):
        fleet_needed(100, capacity=0)

def test_equilibrium_and_mae():
    price, quantity = equilibrium()
    assert price == pytest.approx(2200)
    assert quantity == pytest.approx(76)
    assert mae([1, 2], [2, 2]) == pytest.approx(0.5)

def test_seasonal_naive():
    assert SeasonalNaive(7).predict(np.arange(1, 8), 2).tolist() == [1.0, 2.0]
