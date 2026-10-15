# UBOS District Population Forecaster

## Overview
An object-oriented Python mini-project comparing linear trend, constant CAGR, and a Fibonacci-ratio benchmark for illustrative district population estimates. Population values are stored in **thousands of people**, matching the assignment's supplied series.

## Setup
Python 3.10+ is recommended.

After creating a jupyter notebook and saving it, I ran these commands under the the terminal to create a python environment of windows powershell in vs code. 'uv venv .venv' creates a separate environment named .venv in my project. 'uv pip install --python .venv\Scripts\python.exe ipykernel' is run in the same terminal and wait for it to finish installing. The jupyter note book is opened, another kernel is selected then followed by python environments among the options. Then a python interpreter is selected inside the project.
uv venv .venv
uv pip install --python .venv\Scripts\python.exe ipykernel


## Run
Open `project1_population.ipynb` in Jupyter notebooks and select the `.venv` Python kernel. Use **Restart & Run All**.

Run unit tests from the repository root:

python -m pytest -q


## Data and assumptions
Kampala, Wakiso and Gulu are the illustrative series supplied in the assignment. Mbarara and Mbale are additional **synthetic illustrative** series created for demonstration; they are not official UBOS figures. No external population data is claimed. Classroom calculations interpret each population value as thousands of people, multiply by 1,000, assume 18% are primary-school age, and assume 53 pupils per classroom. “Additional classrooms” is the difference between estimated 2029 classroom demand and estimated 2024 demand, rounded up to whole classrooms.

## Findings
The notebook prints district-specific model comparison tables and selects the model with the lowest test-period RMSE (with MAE and MAPE also reported). The CAGR and linear models impose smooth trend assumptions. The Fibonacci-ratio model is intentionally included as a critical benchmark; successive Fibonacci ratios converge to the golden ratio, which implies implausibly rapid annual growth if interpreted literally as population multipliers.

## An AI coding assistant was used to help draft and debug the project structure, Python code, tests, and explanatory text. 
