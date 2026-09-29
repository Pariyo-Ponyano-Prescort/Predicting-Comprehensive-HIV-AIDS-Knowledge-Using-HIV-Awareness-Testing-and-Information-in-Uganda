import numpy as np
import pandas as pd


def clean_indicators(df):
    """Clean the Indicator column."""
    df = df.copy()

    df["Indicator"] = (
        df["Indicator"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def create_age_group(df):
    """Create a new grouped age variable."""
    df = df.copy()

    df["Age.Group.New"] = np.select(
        [
            df["Age Group"].isin([
                "15 - 19 Years",
                "15 - 24 Years",
                "20 - 24 Years",
                "25 - 29 Years"
            ]),

            df["Age Group"].isin([
                "30 - 34 Years",
                "30 - 39 Years",
                "35 - 39 Years",
                "40 - 44 Years",
                "40 - 49 Years",
                "45 - 49 Years"
            ]),

            df["Age Group"].isin([
                "50 - 54 Years"
            ])
        ],

        [
            "15 - 29 Years",
            "35 - 49 Years",
            "50 - 64 Years"
        ],

        default=None
    )

    return df


def select_indicators(df, indicators):
    """Select specified indicators from the dataset."""

    return df[
        df["Indicator"].isin(indicators)
    ][
        ["Indicator", "Age.Group.New", "Value", "Gender"]
    ]