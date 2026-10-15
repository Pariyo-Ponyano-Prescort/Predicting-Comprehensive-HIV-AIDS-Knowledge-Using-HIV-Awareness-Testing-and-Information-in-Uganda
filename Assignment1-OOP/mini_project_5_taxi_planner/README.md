# Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner

The notebook is `project5_taxi.ipynb`; reusable code is in `src/taxi_planner.py`. The project uses NumPy, SciPy, Matplotlib, and pytest

## Setup

python -m venv .venv


Open `project5_taxi.ipynb` in Jupyter and choose **Restart Kernel and Run All**. Run tests from the project folder with:


## Data and assumptions

- The three supplied 10-day passenger series and fares are used as provided.
- Revenue assumes every listed passenger pays the full stated fare; it does not subtract fuel, driver, maintenance, or association costs.
- Passenger counts are treated as daily totals. Fleet planning converts them into vehicle-trip capacity using 14 seats × 8 one-way trips per vehicle per day.
- The fleet buffer is 15%; the final count is rounded up so capacity is not knowingly below buffered demand.
- Equilibrium uses the demand and supply equations exactly as stated. Price is in UGX and quantity is passengers per trip-hour.
- Backtesting predicts days 4–10 using only earlier observations. MAE is measured in passengers, not UGX.
- The 60-day extension is synthetic, uses a fixed NumPy random seed, and adds a weekly pattern: Friday is busier and Sunday quieter.

## Summary of findings

The equilibrium solution is UGX 2,200 and 76 passengers per trip-hour, so the current Ntinda fare of UGX 2,000 is below the modelled equilibrium price. Model performance differs by route and should be compared using the printed walk-forward MAE table. The notebook selects the best model based on the seven-day backtest and forecasts day-11 passenger counts and revenue. Fleet estimates include a 15% buffer but are only planning estimates because passenger totals do not reveal direction, departure-time peaks, vehicle turnaround time, or actual occupancy per trip.

## Findings & limitations

These are illustrative planning calculations, not audited association accounts. Ten observations per route are too few to represent holidays, school terms, weather, traffic, price changes, or long-term growth reliably. Walk-forward MAE gives a fair simple comparison, but seven test days can make model rankings unstable. The linear trend can extrapolate to unrealistic values if a trend continues too far; moving average and exponential smoothing do not explicitly model weekday seasonality. The equilibrium equations are assumptions rather than estimated Ugandan market curves, and the price response may not be linear. Daily revenue is gross fare revenue and ignores operating costs, discounts, empty return trips, fare evasion, and route-specific load factors. Fleet calculations treat passenger counts as passengers requiring service over the day and assume all 8 trips can be used at full 14-seat capacity. Real operations should use counts by trip and time of day, include vehicle availability and round-trip travel time, and validate the forecast with more data. The seasonal extension is deliberately synthetic and demonstrates method behaviour only; it is not evidence about real Kampala demand.


## An AI coding assistant was used to draft a simple implementation, notebook structure, tests, and explanatory text. 
