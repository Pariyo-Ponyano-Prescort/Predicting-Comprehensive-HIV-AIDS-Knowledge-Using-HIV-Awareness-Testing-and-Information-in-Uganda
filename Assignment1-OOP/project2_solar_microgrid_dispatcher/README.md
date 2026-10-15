Writing

# Solar Micro-Grid Dispatch Planner

A Python project that models solar and battery energy requirements for a rural health centre in Kasese, Uganda. It uses linear algebra to estimate energy usage, calculate costs, and analyse demand changes over 30 days.

## Features

- Solves daily solar and battery energy equations.
- Checks the system's determinant and condition number.
- Compares loop-based and vectorized solutions.
- Detects negative energy values and applies non-negative least squares (NNLS).
- Generates and processes 30 days of sample demand data using CSV.
- Calculates daily and monthly energy costs.
- Performs statistical and Monte Carlo sensitivity analysis.
- Generates a plot of energy usage and daily costs.
- Demonstrates an extended solar-battery-diesel model.

## Requirements

- Python 3.10+
- NumPy
- Pandas
- SciPy
- Matplotlib

Install the dependencies:

```
pip install numpy pandas scipy matplotlib
```

## How to Run

Save the program as `microgrid.py`, then run:

```
python microgrid.py
```

Choose one of the available options to enter a single day's demand, generate sample data, or load an existing CSV file.

## Findings

- Vectorized solving can reduce processing time compared with solving each day in a loop, although the actual performance depends on the system and environment.
- The determinant checks whether the equations have a unique solution, while the condition number indicates sensitivity to input changes.
- Negative energy values indicate physically infeasible results under the model. NNLS provides non-negative alternatives but may leave some demand unmet.
- Solar and battery usage, estimated costs, and sensitivity to demand changes can be compared over the 30-day period.

## Limitations

- The demand data is randomly generated sample data, not actual measurements from the health centre.
- Solar and battery costs are assumed values and may not reflect current market prices.
- The model does not account for battery capacity, charging efficiency, weather, solar availability, or equipment losses.
- NNLS solutions may not satisfy both demand equations exactly.
- The hybrid model assumes an additional energy constraint that must be consistent with the original equations.
- The results are planning estimates, not a complete engineering design or operational forecast.

## Output Files

- `kasese_microgrid_demand.csv` — generated daily demand data.

**Note:** Results depend on the model assumptions and should be validated using real demand data and engineering requirements before practical implementation.