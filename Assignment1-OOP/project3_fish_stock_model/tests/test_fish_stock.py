import sys
from pathlib import Path
import unittest
import numpy as np

# Locate the project root and src directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Add src to Python's module search path
sys.path.insert(0, str(SRC_DIR))

from fish_stock_model import FishStock



class TestFishStock(unittest.TestCase):

    def setUp(self):
        """Create a standard fish stock model for each test."""
        self.model = FishStock(
            initial_stock=4000,
            growth_rate=0.4,
            carrying_capacity=10000,
            harvest_rate=0.10,
        )

    def test_initial_stock(self):
        """Initial stock should match the configured value."""
        trajectory = self.model.simulate(52)

        self.assertAlmostEqual(trajectory[0], 4000)

    def test_simulation_length(self):
        """Simulation should return the expected number of stock values."""
        trajectory = self.model.simulate(52)

        self.assertEqual(len(trajectory), 53)

    def test_stock_values_are_non_negative(self):
        """Fish stock should never be negative."""
        trajectory = self.model.simulate(52)

        self.assertTrue(np.all(np.asarray(trajectory) >= 0))

    def test_stock_values_are_finite(self):
        """Simulation should not produce NaN or infinite values."""
        trajectory = self.model.simulate(52)

        self.assertTrue(np.all(np.isfinite(trajectory)))

    def test_harvest_is_non_negative(self):
        """Weekly harvest quantities should not be negative."""
        trajectory = self.model.simulate(52)
        harvest = self.model.weekly_harvest_tonnes(trajectory)

        self.assertTrue(np.all(np.asarray(harvest) >= 0))

    def test_zero_harvest_rate(self):
        """A zero harvest rate should produce zero harvest."""
        model = FishStock(
            initial_stock=4000,
            growth_rate=0.4,
            carrying_capacity=10000,
            harvest_rate=0.0,
        )

        trajectory = model.simulate(52)
        harvest = model.weekly_harvest_tonnes(trajectory)

        np.testing.assert_allclose(harvest, 0.0)

    def test_closed_season_produces_zero_harvest(self):
        """Harvest should be zero during closed weeks."""
        closed_weeks = {0, 1, 2, 3}

        trajectory = self.model.simulate(
            52,
            closed_season_weeks=closed_weeks,
        )

        harvest = self.model.weekly_harvest_tonnes(
            trajectory,
            closed_season_weeks=closed_weeks,
        )

        for week in closed_weeks:
            self.assertAlmostEqual(harvest[week], 0.0)

    def test_invalid_simulation_length(self):
        """Negative simulation length should be rejected."""
        with self.assertRaises((ValueError, TypeError)):
            self.model.simulate(-1)


if __name__ == "__main__":
    unittest.main()