

import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import solve
from scipy.optimize import nnlspup
from scipy.optimize import nnls

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SOLAR_COST = 150.0       # UGX per kWh
BATTERY_COST = 450.0     # UGX per kWh
DAYS_IN_MONTH = 30

CSV_FILE = "kasese_microgrid_demand.csv"
RANDOM_SEED = 42


# ---------------------------------------------------------
# MICROGRID CLASS
# ---------------------------------------------------------

class MicroGrid:
    """
    Solar-battery micro-grid.

    The demand equations are:

        3x + 2y = D1
        4x +  y = D2

    x = solar energy (kWh)
    y = battery energy (kWh)
    D1 = daytime demand
    D2 = critical-equipment demand
    """
def solve_nonnegative(self, d1, d2):
    """Find non-negative solar and battery energy values."""
    demand = np.array([d1, d2], dtype=float)

    solution, residual_norm = nnls(self.A, demand)

    return solution, residual_norm


    def __init__(self):
        self.A = np.array([
            [3.0, 2.0],
            [4.0, 1.0]
        ])

        self.determinant = np.linalg.det(self.A)
        self.condition_number = np.linalg.cond(self.A)

        self._check_system()

    def _check_system(self):
        """Check whether the coefficient matrix is suitable for solving."""

        print("\n--- SYSTEM WELL-POSEDNESS CHECK ---")
        print("Coefficient matrix:")
        print(self.A)

        print(f"Determinant: {self.determinant:.4f}")
        print(f"Condition number: {self.condition_number:.4f}")

        if np.isclose(self.determinant, 0.0):
            raise ValueError(
                "The coefficient matrix is singular. "
                "A unique solution cannot be guaranteed."
            )

        print(
            "Interpretation: The determinant is non-zero, so the "
            "system has a unique mathematical solution for each "
            "right-hand side."
        )

        print(
            "The condition number measures how sensitive the "
            "solution is to changes in demand. A modest condition "
            "number indicates that small relative input errors "
            "should not cause extreme relative solution errors."
        )

    def solve_day(self, d1, d2):
        """Solve one day's demand equations."""

        demand = np.array([d1, d2], dtype=float)
        return solve(self.A, demand)

    def solve_many_loop(self, demands):
        """Solve multiple days one at a time."""

        results = []

        for d1, d2 in demands:
            results.append(self.solve_day(d1, d2))

        return np.array(results)

    def solve_many_vectorized(self, demands):
        """
        Solve all days in one call.

        The right-hand side has shape (2, number_of_days).
        """

        rhs = np.asarray(demands, dtype=float).T
        solutions = solve(self.A, rhs)

        return solutions.T

    def solve_nonnegative(self, d1, d2):
        """
        Find a non-negative least-squares solution.

        This is used when the ordinary solution gives negative
        solar or battery usage.

        NNLS minimizes ||A @ solution - demand|| subject to
        solution >= 0.
        """

        demand = np.array([d1, d2], dtype=float)

        solution, residual_norm = nnls(self.A, demand)

        return solution, residual_norm

    def calculate_cost(self, solar, battery):
        """Calculate the cost of solar and battery energy."""

        return (
            solar * SOLAR_COST
            + battery * BATTERY_COST
        )


# ---------------------------------------------------------
# HYBRID MICROGRID EXTENSION
# ---------------------------------------------------------

class HybridMicroGrid(MicroGrid):
    """
    Hybrid solar-battery-diesel micro-grid.

    The third constraint is a planning assumption:

        x + y + z = D3

    D3 is a specified total energy target, and z is diesel
    generator energy in kWh.

    NOTE:
    The original two equations are retained, so D3 must be
    consistent with the resulting x and y if all three
    equations are to be satisfied exactly.
    """

    def __init__(self):
        self.A = np.array([
            [3.0, 2.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])

        self.determinant = np.linalg.det(self.A)
        self.condition_number = np.linalg.cond(self.A)

        print("\n--- HYBRID MICROGRID CHECK ---")
        print("Coefficient matrix:")
        print(self.A)
        print(f"Determinant: {self.determinant:.4f}")
        print(f"Condition number: {self.condition_number:.4f}")

        if np.isclose(self.determinant, 0.0):
            raise ValueError(
                "Hybrid system is singular. "
                "The equations are linearly dependent."
            )

        print(
            "A non-zero determinant indicates a unique solution "
            "for a given set of three demand values."
        )

    def solve_day(self, d1, d2, d3):
        """Solve for solar, battery and diesel energy."""

        demand = np.array([d1, d2, d3], dtype=float)
        return solve(self.A, demand)


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def get_nonnegative_number(prompt):
    """
    Read a valid non-negative number.

    Rejects empty, non-numeric, negative, NaN and infinite inputs.
    """

    while True:
        raw = input(prompt).strip()

        if not raw:
            print("Error: Input cannot be empty. Try again.")
            continue

        try:
            value = float(raw)
        except ValueError:
            print("Error: Enter a numeric value. Try again.")
            continue

        if not np.isfinite(value):
            print("Error: Enter a finite number. Try again.")
            continue

        if value < 0:
            print("Error: Negative demand is not allowed.")
            continue

        return value


def interactive_mode(grid):
    """Solve one day using interactive user input."""

    print("\n--- INTERACTIVE INPUT MODE ---")

    d1 = get_nonnegative_number(
        "Enter daytime demand D1 (kWh): "
    )

    d2 = get_nonnegative_number(
        "Enter critical-equipment demand D2 (kWh): "
    )

    solution = grid.solve_day(d1, d2)

    solar, battery = solution

    print("\nOrdinary mathematical solution:")
    print(f"Solar energy: {solar:.3f} kWh")
    print(f"Battery energy: {battery:.3f} kWh")

    if solar < 0 or battery < 0:
        print(
            "\nWARNING: This solution is physically infeasible "
            "because at least one energy source is negative."
        )

        adjusted, residual = grid.solve_nonnegative(d1, d2)

        print("\nNon-negative least-squares alternative:")
        print(f"Solar energy: {adjusted[0]:.3f} kWh")
        print(f"Battery energy: {adjusted[1]:.3f} kWh")
        print(f"Demand-equation residual norm: {residual:.3f}")

        print(
            "The adjusted solution respects non-negativity but "
            "may not satisfy both demand equations exactly."
        )
    else:
        cost = grid.calculate_cost(solar, battery)
        print(f"Estimated daily energy cost: UGX {cost:,.2f}")


# ---------------------------------------------------------
# GENERATE 30 DAYS OF CSV DATA
# ---------------------------------------------------------

def generate_demand_csv(filename=CSV_FILE):
    """
    Generate reproducible demand data with a weekly pattern
    and random noise.

    Demand levels are illustrative planning assumptions,
    not measured data from the health centre.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    day_numbers = np.arange(1, DAYS_IN_MONTH + 1)

    # Weekly demand pattern: weekdays have higher activity.
    weekly_pattern = np.array([
        1.00, 1.04, 1.02, 1.06, 1.08, 0.94, 0.88
    ])

    pattern = weekly_pattern[
        (day_numbers - 1) % 7
    ]

    # Illustrative base demands in the model's demand units.
    base_d1 = 120.0
    base_d2 = 100.0

    noise_d1 = rng.normal(0.0, 4.0, DAYS_IN_MONTH)
    noise_d2 = rng.normal(0.0, 3.0, DAYS_IN_MONTH)

    d1 = base_d1 * pattern + noise_d1
    d2 = base_d2 * pattern + noise_d2

    d1 = np.maximum(d1, 0.0)
    d2 = np.maximum(d2, 0.0)

    data = pd.DataFrame({
        "Day": day_numbers,
        "D1_kWh": d1,
        "D2_kWh": d2
    })

    data.to_csv(filename, index=False)

    print(f"\nGenerated demand CSV: {filename}")
    print(data.head())

    return data


def load_demand_csv(filename=CSV_FILE):
    """Load and validate the demand data from a CSV file."""

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {filename}"
        )

    data = pd.read_csv(path)

    required_columns = ["D1_kWh", "D2_kWh"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"Missing required CSV column: {column}"
            )

        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

        if not np.all(np.isfinite(data[column])):
            raise ValueError(
                f"{column} contains NaN or infinite values."
            )

        if (data[column] < 0).any():
            raise ValueError(
                f"{column} contains negative demand."
            )

    if len(data) != DAYS_IN_MONTH:
        raise ValueError(
            f"Expected exactly {DAYS_IN_MONTH} days, "
            f"but found {len(data)}."
        )

    return data


# ---------------------------------------------------------
# LOOP AND VECTORIZED SOLUTIONS
# ---------------------------------------------------------

def benchmark_solvers(grid, demands):
    """Compare loop-based and vectorized solving."""

    print("\n--- SOLVER PERFORMANCE ---")

    loop_time = timeit.timeit(
        lambda: grid.solve_many_loop(demands),
        number=1000
    )

    vector_time = timeit.timeit(
        lambda: grid.solve_many_vectorized(demands),
        number=1000
    )

    loop_results = grid.solve_many_loop(demands)
    vector_results = grid.solve_many_vectorized(demands)

    print(f"Loop time (1000 runs): {loop_time:.6f} seconds")
    print(
        f"Vectorized time (1000 runs): "
        f"{vector_time:.6f} seconds"
    )

    print(
        "Maximum absolute difference between solutions: "
        f"{np.max(np.abs(loop_results - vector_results)):.12f}"
    )

    if vector_time < loop_time:
        print(
            "Vectorization was faster in this run because it "
            "reduces Python-level looping and solves all right-hand "
            "sides in one operation."
        )
    else:
        print(
            "The loop was faster in this run. For a small 2-by-2 "
            "system and only 30 days, timing overhead and library "
            "implementation can affect the result."
        )

    return vector_results


# ---------------------------------------------------------
# INFEASIBILITY HANDLING
# ---------------------------------------------------------

def handle_infeasible_days(grid, demands, solutions):
    """
    Detect negative solutions and replace them with NNLS results.

    Returns:
        adjusted solutions,
        infeasibility flags,
        NNLS residuals.
    """

    adjusted = solutions.copy()
    flags = np.zeros(len(solutions), dtype=bool)
    residuals = np.zeros(len(solutions))

    print("\n--- PHYSICAL FEASIBILITY REPORT ---")

    for i, (d1, d2) in enumerate(demands):
        solar, battery = solutions[i]

        if solar < 0 or battery < 0:
            flags[i] = True

            adjusted[i], residuals[i] = (
                grid.solve_nonnegative(d1, d2)
            )

            print(
                f"Day {i + 1}: INFEASIBLE ordinary solution. "
                f"Solar={solar:.3f}, Battery={battery:.3f}. "
                f"NNLS replacement: Solar={adjusted[i, 0]:.3f}, "
                f"Battery={adjusted[i, 1]:.3f}, "
                f"Residual={residuals[i]:.3f}"
            )

    if not flags.any():
        print(
            "All 30 ordinary solutions are non-negative "
            "and physically feasible under this model."
        )

    print(f"Total infeasible days: {flags.sum()}")

    return adjusted, flags, residuals


# ---------------------------------------------------------
# STATISTICS AND COST
# ---------------------------------------------------------

def report_statistics(data, solutions, flags):
    """Report usage statistics and daily/monthly costs."""

    solar = solutions[:, 0]
    battery = solutions[:, 1]

    data = data.copy()

    data["Solar_kWh"] = solar
    data["Battery_kWh"] = battery
    data["Infeasible"] = flags

    data["Daily_Cost_UGX"] = (
        data["Solar_kWh"] * SOLAR_COST
        + data["Battery_kWh"] * BATTERY_COST
    )

    print("\n--- ENERGY USAGE STATISTICS ---")

    print("\nSolar energy:")
    print(f"Mean: {np.mean(solar):.3f} kWh")
    print(f"Variance: {np.var(solar):.3f} kWh^2")
    print(f"Standard deviation: {np.std(solar):.3f} kWh")

    print("\nBattery energy:")
    print(f"Mean: {np.mean(battery):.3f} kWh")
    print(f"Variance: {np.var(battery):.3f} kWh^2")
    print(
        f"Standard deviation: {np.std(battery):.3f} kWh"
    )

    if np.std(solar) > np.std(battery):
        print("Solar usage has greater absolute volatility.")
    elif np.std(battery) > np.std(solar):
        print("Battery usage has greater absolute volatility.")
    else:
        print("Both sources have equal absolute volatility.")

    print(
        "Note: Standard deviation compares absolute variation. "
        "The coefficient of variation (standard deviation divided "
        "by the mean) can be used to compare relative variation."
    )

    solar_cv = (
        np.std(solar) / abs(np.mean(solar))
        if np.mean(solar) != 0 else np.nan
    )

    battery_cv = (
        np.std(battery) / abs(np.mean(battery))
        if np.mean(battery) != 0 else np.nan
    )

    print(f"Solar coefficient of variation: {solar_cv:.4f}")
    print(
        f"Battery coefficient of variation: "
        f"{battery_cv:.4f}"
    )

    print("\n--- ENERGY COST ---")

    print(f"Solar rate: UGX {SOLAR_COST:.2f}/kWh")
    print(f"Battery rate: UGX {BATTERY_COST:.2f}/kWh")

    print("\nDaily costs:")
    print(
        data[
            ["Day", "Solar_kWh", "Battery_kWh",
             "Daily_Cost_UGX", "Infeasible"]
        ].to_string(index=False)
    )

    total_cost = data["Daily_Cost_UGX"].sum()
    average_daily_cost = data["Daily_Cost_UGX"].mean()

    print(
        f"\nTotal 30-day energy cost: UGX {total_cost:,.2f}"
    )

    print(
        f"Average daily energy cost: "
        f"UGX {average_daily_cost:,.2f}"
    )

    print(
        f"Monthly energy cost for this 30-day period: "
        f"UGX {total_cost:,.2f}"
    )

    return data


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def plot_usage_and_cost(data):
    """Plot stacked energy usage and daily cost."""

    days = data["Day"].to_numpy()

    solar = data["Solar_kWh"].to_numpy()
    battery = data["Battery_kWh"].to_numpy()
    costs = data["Daily_Cost_UGX"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(13, 7))

    ax1.bar(
        days,
        solar,
        label="Solar usage",
        color="gold"
    )

    ax1.bar(
        days,
        battery,
        bottom=solar,
        label="Battery usage",
        color="steelblue"
    )

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Energy usage (kWh)")
    ax1.set_title(
        "Kasese Health Centre: Solar Micro-Grid Dispatch Planner"
    )

    ax1.set_xticks(days)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        days,
        costs,
        color="darkred",
        marker="o",
        linewidth=2,
        label="Daily energy cost"
    )

    ax2.set_ylabel("Daily energy cost (UGX)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig("microgrid_usage_and_cost.png", dpi=150)
    plt.show()


# ---------------------------------------------------------
# SENSITIVITY ANALYSIS
# ---------------------------------------------------------

def sensitivity_analysis(grid, demands, draws=1000):
    """
    Perturb each day's D1 and D2 independently by up to +/-5%.

    A seeded Monte Carlo simulation is used for reproducibility.
    The results describe sensitivity of the unconstrained linear
    model; negative solutions are not automatically clipped here.
    """

    print("\n--- MONTE CARLO SENSITIVITY ANALYSIS ---")

    rng = np.random.default_rng(RANDOM_SEED + 1)

    original = grid.solve_many_vectorized(demands)

    perturbed_demands = (
        demands[None, :, :]
        * (
            1.0
            + rng.uniform(
                -0.05, 0.05,
                size=(draws, len(demands), 2)
            )
        )
    )

    # Shape: (draws, days, 2)
    # Solve all days and draws in one matrix operation.
    rhs = perturbed_demands.reshape(-1, 2).T

    perturbed_solutions = solve(
        grid.A, rhs
    ).T.reshape(draws, len(demands), 2)

    changes = perturbed_solutions - original[None, :, :]

    solar_changes = changes[:, :, 0]
    battery_changes = changes[:, :, 1]

    solar_abs = np.abs(solar_changes)
    battery_abs = np.abs(battery_changes)

    print(f"Monte Carlo draws: {draws}")
    print("Demand perturbation range: +/-5%")

    print("\nSolar solution movement:")
    print(
        f"Mean absolute change: "
        f"{np.mean(solar_abs):.4f} kWh"
    )
    print(
        f"95th percentile absolute change: "
        f"{np.percentile(solar_abs, 95):.4f} kWh"
    )
    print(
        f"Maximum absolute change: "
        f"{np.max(solar_abs):.4f} kWh"
    )

    print("\nBattery solution movement:")
    print(
        f"Mean absolute change: "
        f"{np.mean(battery_abs):.4f} kWh"
    )
    print(
        f"95th percentile absolute change: "
        f"{np.percentile(battery_abs, 95):.4f} kWh"
    )
    print(
        f"Maximum absolute change: "
        f"{np.max(battery_abs):.4f} kWh"
    )

    print(
        "\nInterpretation: The condition number describes worst-case "
        "relative sensitivity to input perturbations in the "
        "linear system. Monte Carlo results show the actual "
        "solution movement for the sampled +/-5% demand changes."
    )

    return changes


# ---------------------------------------------------------
# HYBRID MICROGRID DEMONSTRATION
# ---------------------------------------------------------

def demonstrate_hybrid_grid():
    """Demonstrate the 3-source extension."""

    print("\n--- HYBRID MICROGRID DEMONSTRATION ---")

    hybrid = HybridMicroGrid()

    d1 = 120.0
    d2 = 100.0

    # Select D3 to make the third equation consistent with
    # the first two equations for this demonstration.
    solar, battery = solve(
        hybrid.A[:2, :2],
        np.array([d1, d2])
    )

    d3 = solar + battery + 10.0

    x, y, z = hybrid.solve_day(d1, d2, d3)

    print(f"D1: {d1:.2f}")
    print(f"D2: {d2:.2f}")
    print(f"D3: {d3:.2f}")
    print(f"Solar energy x: {x:.3f} kWh")
    print(f"Battery energy y: {y:.3f} kWh")
    print(f"Diesel energy z: {z:.3f} kWh")

    if x < 0 or y < 0 or z < 0:
        print(
            "WARNING: The hybrid solution contains a negative "
            "energy value and is physically infeasible."
        )

    print(
        "\nIf the third equation were linearly dependent on the "
        "first two, the determinant would be zero. The system "
        "would then either have infinitely many solutions or "
        "no exact solution, depending on whether the demand "
        "values were consistent."
    )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():
    """Run the complete solar micro-grid dispatch planner."""

    print("=" * 65)
    print("SOLAR MICRO-GRID DISPATCH PLANNER")
    print("Rural Health Centre, Kasese, Uganda")
    print("=" * 65)

    grid = MicroGrid()

    print("\nChoose an input mode:")
    print("1. Interactive input for one day")
    print("2. Generate and process 30 days of CSV demand data")
    print("3. Load an existing 30-day CSV file")

    while True:
        choice = input("Enter 1, 2 or 3: ").strip()

        if choice in {"1", "2", "3"}:
            break

        print("Invalid choice. Enter 1, 2 or 3.")

    if choice == "1":
        interactive_mode(grid)
        return

    if choice == "2":
        generate_demand_csv(CSV_FILE)

    try:
        data = load_demand_csv(CSV_FILE)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error loading demand data: {exc}")
        return

    demands = data[
        ["D1_kWh", "D2_kWh"]
    ].to_numpy(dtype=float)

    # Solve all 30 days using both approaches.
    benchmark_solvers(grid, demands)

    # Obtain the ordinary mathematical solutions.
    ordinary_solutions = grid.solve_many_vectorized(demands)

    # Flag infeasible results and replace them with NNLS estimates.
    adjusted_solutions, flags, residuals = (
        handle_infeasible_days(
            grid, demands, ordinary_solutions
        )
    )

    # Report statistics and energy costs using adjusted results.
    results = report_statistics(
        data, adjusted_solutions, flags
    )

    # Plot daily energy usage and cost.
    plot_usage_and_cost(results)

    # Perform 1,000 Monte Carlo sensitivity draws.
    sensitivity_analysis(grid, demands, draws=1000)

    # Demonstrate the hybrid extension.
    demonstrate_hybrid_grid()

    print("\n--- COMPLETED ---")
    print("CSV file: kasese_microgrid_demand.csv")
    print("Plot file: microgrid_usage_and_cost.png")


if __name__ == "__main__":
    main()