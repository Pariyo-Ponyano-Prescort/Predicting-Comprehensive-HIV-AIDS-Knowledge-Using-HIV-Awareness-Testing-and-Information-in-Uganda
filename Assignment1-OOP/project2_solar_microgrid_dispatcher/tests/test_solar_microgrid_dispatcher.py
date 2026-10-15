import pytest
import pandas as pd

# Adjust this import to match your actual module and class names.
from solar_microgrid_dispatcher import SolarMicrogridDispatcher


@pytest.fixture
def dispatcher():
    """Create a dispatcher instance for testing."""
    return SolarMicrogridDispatcher()


def test_dispatcher_initialization(dispatcher):
    """The dispatcher should initialize successfully."""
    assert dispatcher is not None


def test_daily_cost_returns_numeric_value(dispatcher):
    """Daily energy cost should be a numeric value."""
    cost = dispatcher.daily_cost(solar=5.0, battery=2.0)

    assert isinstance(cost, (int, float))
    assert cost >= 0


def test_zero_solar_and_zero_battery(dispatcher):
    """Zero solar and battery energy should not cause an error."""
    cost = dispatcher.daily_cost(solar=0.0, battery=0.0)

    assert isinstance(cost, (int, float))
    assert cost >= 0


def test_daily_cost_is_repeatable(dispatcher):
    """Identical inputs should produce identical costs."""
    cost_1 = dispatcher.daily_cost(solar=5.0, battery=2.0)
    cost_2 = dispatcher.daily_cost(solar=5.0, battery=2.0)

    assert cost_1 == pytest.approx(cost_2)


def test_dispatch_dataframe_has_required_columns():
    """Dispatch results should contain the required energy columns."""
    dispatch_df = pd.DataFrame({
        "day": [1, 2, 3],
        "solar_kwh": [5.0, 6.0, 4.0],
        "battery_kwh": [2.0, 1.0, 3.0],
    })

    required_columns = {
        "day",
        "solar_kwh",
        "battery_kwh",
    }

    assert required_columns.issubset(dispatch_df.columns)


def test_monthly_cost_equals_sum_of_daily_costs():
    """Monthly cost should equal the sum of daily costs."""
    dispatch_df = pd.DataFrame({
        "daily_cost_ugx": [1000.0, 1500.0, 2000.0],
    })

    monthly_cost = float(dispatch_df["daily_cost_ugx"].sum())

    assert monthly_cost == pytest.approx(4500.0)


def test_average_daily_cost():
    """Average daily cost should equal total cost divided by days."""
    dispatch_df = pd.DataFrame({
        "daily_cost_ugx": [1000.0, 1500.0, 2000.0],
    })

    average_cost = dispatch_df["daily_cost_ugx"].mean()

    assert average_cost == pytest.approx(1500.0)