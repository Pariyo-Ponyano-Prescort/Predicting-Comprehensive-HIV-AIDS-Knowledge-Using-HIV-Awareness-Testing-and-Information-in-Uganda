
import numpy as np
import pytest

from rainfall import (
    Region,
    CropRule,
    cosine_similarity,
    compare_regions,
)


# -------------------------
# Region tests
# -------------------------

def test_annual_total():
    region = Region("Test", [10.0] * 12)
    assert region.annual_total() == 120.0


def test_mean_rainfall():
    region = Region("Test", [10.0] * 12)
    assert region.mean() == 10.0


def test_wettest_month():
    rainfall = [10, 20, 30, 40, 50, 60,
                70, 80, 90, 100, 110, 120]
    region = Region("Test", rainfall)
    assert region.wettest_month() == "Dec"


def test_driest_month():
    rainfall = [10, 20, 30, 40, 50, 60,
                70, 80, 90, 100, 110, 120]
    region = Region("Test", rainfall)
    assert region.driest_month() == "Jan"


def test_zero_rainfall_cv():
    region = Region("Dry", np.zeros(12))
    assert np.isnan(region.coefficient_of_variation())


def test_coefficient_of_variation():
    rainfall = [10.0] * 12
    region = Region("Test", rainfall)
    assert region.coefficient_of_variation() == 0.0


def test_invalid_region_name():
    with pytest.raises(ValueError):
        Region("", [10.0] * 12)


def test_invalid_rainfall_length():
    with pytest.raises(ValueError):
        Region("Invalid", [1.0, 2.0])


def test_negative_rainfall():
    rainfall = [10.0] * 12
    rainfall[0] = -1.0

    with pytest.raises(ValueError):
        Region("Invalid", rainfall)


def test_non_finite_rainfall():
    rainfall = [10.0] * 12
    rainfall[0] = np.nan

    with pytest.raises(ValueError):
        Region("Invalid", rainfall)


def test_rainfall_is_copied():
    values = np.array([10.0] * 12)
    region = Region("Test", values)

    values[0] = 999.0

    assert region.rainfall[0] == 10.0


# -------------------------
# Rainy-season tests
# -------------------------

def test_detect_two_rainy_seasons():
    rainfall = [
        10, 20, 40, 100, 40, 20,
        10, 20, 40, 90, 40, 20
    ]

    result = Region.detect_rainy_seasons(
        rainfall, prominence=30
    )

    assert len(result["peak_indices"]) == 2
    assert list(result["peak_indices"]) == [3, 9]
    assert result["peak_months"] == ["Apr", "Oct"]


def test_rainy_season_invalid_length():
    with pytest.raises(ValueError):
        Region.detect_rainy_seasons([10, 20, 30])


def test_rainy_seasons_method():
    rainfall = [
        10, 20, 40, 100, 40, 20,
        10, 20, 40, 90, 40, 20
    ]

    region = Region("Test", rainfall)
    assert region.rainy_seasons() == ["Apr", "Oct"]


def test_bimodal_season_pattern():
    rainfall = [
        10, 20, 40, 100, 40, 20,
        10, 20, 40, 90, 40, 20
    ]

    region = Region("Test", rainfall)
    assert region.season_pattern() == "Bimodal"


# -------------------------
# CropRule tests
# -------------------------

def test_crop_rule_good_rainfall():
    maize = CropRule("maize", 80, 200)
    assert maize.classify(100) == "Good for maize"


def test_crop_rule_drought_risk():
    maize = CropRule("maize", 80, 200)
    assert maize.classify(20) == "Drought risk"


def test_crop_rule_waterlogging_risk():
    maize = CropRule("maize", 80, 200)
    assert maize.classify(220) == "Waterlogging risk"


def test_crop_rule_threshold_boundaries():
    maize = CropRule("maize", 80, 200)

    assert maize.classify(80) == "Good for maize"
    assert maize.classify(200) == "Good for maize"


def test_invalid_crop_thresholds():
    with pytest.raises(ValueError):
        CropRule("maize", 200, 80)


def test_invalid_crop_rainfall():
    maize = CropRule("maize", 80, 200)

    with pytest.raises(ValueError):
        maize.classify(-5)


# -------------------------
# Cosine similarity tests
# -------------------------

def test_cosine_similarity():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])

    assert np.isclose(cosine_similarity(a, b), 1.0)


def test_cosine_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(3), np.ones(3))


def test_cosine_incompatible_shapes():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_cosine_non_finite_values():
    with pytest.raises(ValueError):
        cosine_similarity([1, np.nan], [2, 3])


# -------------------------
# Region comparison tests
# -------------------------

def test_compare_regions_returns_all_matrices():
    regions = {
        "A": Region("A", [10.0] * 12),
        "B": Region("B", [20.0] * 12),
    }

    result = compare_regions(regions)

    assert set(result.keys()) == {
        "Cosine similarity",
        "Pearson correlation",
        "Euclidean distance",
    }


def test_compare_regions_matrix_shapes():
    regions = {
        "A": Region("A", [10.0] * 12),
        "B": Region("B", [20.0] * 12),
        "C": Region("C", list(range(12))),
    }

    result = compare_regions(regions)

    for matrix in result.values():
        assert matrix.shape == (3, 3)


def test_compare_identical_regions():
    values = [10, 20, 30, 40, 50, 60,
              70, 80, 90, 100, 110, 120]

    regions = {
        "A": Region("A", values),
        "B": Region("B", values),
    }

    result = compare_regions(regions)

    assert np.isclose(
        result["Cosine similarity"][0, 1], 1.0
    )
    assert np.isclose(
        result["Pearson correlation"][0, 1], 1.0
    )
    assert np.isclose(
        result["Euclidean distance"][0, 1], 0.0
    )


def test_compare_constant_regions_pearson_is_nan():
    regions = {
        "A": Region("A", [10.0] * 12),
        "B": Region("B", [20.0] * 12),
    }

    result = compare_regions(regions)

    assert np.isnan(
        result["Pearson correlation"][0, 1]
    )
