import numpy as np
import pytest

from src.population_forecaster import (
    DistrictPopulation,
    LinearTrendForecaster,
    CAGRForecaster,
    FibonacciRatioForecaster,
    calculate_metrics,
)


def test_district_population_length_and_repr():
    district = DistrictPopulation("Test", np.array([2020, 2021]), np.array([10, 12]))
    assert len(district) == 2
    assert "Test" in repr(district)


def test_district_rejects_negative_population():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", np.array([2020, 2021]), np.array([10, -1]))


def test_district_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", np.array([2020, 2021]), np.array([10]))


def test_linear_trend_predicts_positive_horizon():
    model = LinearTrendForecaster().fit([2020, 2021, 2022], [10, 12, 14])
    assert np.allclose(model.predict(2), [16, 18])


def test_forecaster_rejects_zero_horizon():
    model = CAGRForecaster().fit([2020, 2021], [10, 11])
    with pytest.raises(ValueError):
        model.predict(0)


def test_metrics_perfect_predictions():
    result = calculate_metrics([10, 20], [10, 20])
    assert result["MAE"] == 0
    assert result["RMSE"] == 0
    assert result["MAPE (%)"] == 0


def test_fibonacci_model_returns_requested_horizon():
    model = FibonacciRatioForecaster().fit([2020, 2021, 2022], [10, 11, 12])
    assert len(model.predict(4)) == 4
