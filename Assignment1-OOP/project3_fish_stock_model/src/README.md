

# Fish Stock, Price and Revenue Risk Models

A simple Python project for modelling fish population growth, weekly fish prices, and revenue risk. It uses mathematical models and random simulations to explore how harvesting and price changes may affect fishery revenue.

## Features

- **Fish stock:** Simulates weekly fish biomass using logistic growth and harvesting.
- **Closed seasons:** Models periods when harvesting is suspended.
- **Price model:** Simulates weekly fish prices within set limits.
- **Revenue risk:** Calculates mean, median, variance, standard deviation, and coefficient of variation (CV) to classify risk as low, moderate, or high.
- **Fibonacci baseline:** Generates Fibonacci numbers for comparison with the fish stock model.

## Requirements

- Python 3.10+
- NumPy

Install the dependency:

```
pip install numpy
```

## How to Use

Save the code as `fish_stock_model.py` and import the required classes:

```
from fish_model import FishStock, PriceModel, RiskAssessor

stock = FishStock().simulate(weeks=52)
prices = PriceModel().simulate(weeks=52)

weekly_harvest = FishStock().weekly_harvest_tonnes(stock)
```

Use `RiskAssessor` to analyse a list of calculated revenues.

## Findings

- Fish stock depends on initial biomass, growth rate, carrying capacity, and harvesting rate.
- Closed seasons allow the stock to grow without harvesting during the specified weeks.
- Prices fluctuate randomly but remain within configured bounds.
- A higher revenue coefficient of variation indicates greater relative variability and is classified as higher risk.

## Limitations

- The models use assumed parameters rather than real fishery and market data.
- Price changes are random and do not account for actual market demand, seasonality, or inflation.
- The revenue risk classification uses fixed thresholds and does not predict future losses.
- The Fibonacci sequence is only a comparison baseline and is not a realistic fish population model.
- The model does not account for fish species, environmental changes, or illegal harvesting.
