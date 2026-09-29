import pandas as pd

from src.data_processing import (
    create_age_group,
    clean_indicators,
    select_indicators
)


# Test 1: create_age_group()
def test_create_age_group():
    df = pd.DataFrame({
        "Age Group": [
            "15 - 19 Years",
            "25 - 29 Years",
            "40 - 49 Years",
            "50 - 54 Years"
        ]
    })

    result = create_age_group(df)

    assert result["Age.Group.New"].tolist() == [
        "15 - 29 Years",
        "15 - 29 Years",
        "35 - 49 Years",
        "50 - 64 Years"
    ]


# Test 2: clean_indicators()
def test_clean_indicators():
    df = pd.DataFrame({
        "Indicator": [
            "  Percentage of men with HIV knowledge  ",
            "Percentage of women tested\nfor AIDS"
        ]
    })

    result = clean_indicators(df)

    assert result["Indicator"].tolist() == [
        "Percentage of men with HIV knowledge",
        "Percentage of women tested for AIDS"
    ]


# Test 3: select_indicators()
def test_select_indicators():
    df = pd.DataFrame({
        "Indicator": [
            "Indicator A",
            "Indicator B",
            "Indicator C"
        ],
        "Age.Group.New": [
            "15 - 29 Years",
            "35 - 49 Years",
            "50 - 64 Years"
        ],
        "Value": [10, 20, 30],
        "Gender": [
            "Female",
            "Male",
            "Female"
        ]
    })

    indicators = [
        "Indicator A",
        "Indicator C"
    ]

    result = select_indicators(df, indicators)

    assert len(result) == 2
    assert result["Indicator"].tolist() == [
        "Indicator A",
        "Indicator C"
    ]

# TEST 4: Edge case: empty input
def test_select_indicators_empty():
    df = pd.DataFrame({
        "Indicator": [],
        "Age.Group.New": [],
        "Value": [],
        "Gender": []
    })

    result = select_indicators(
        df,
        ["Indicator A"]
    )

    assert result.empty