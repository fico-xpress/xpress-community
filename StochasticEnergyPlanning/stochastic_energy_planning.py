"""
Bayesian Demand Forecasting and Robust Optimization for Electricity Generation
===============================================================================
Combines PyMC probabilistic forecasting with FICO Xpress robust optimization
for power system planning under demand uncertainty.

Forecasting (PyMC):
- Bayesian time-series model on Ontario historical demand data (2024-2025)
- Fourier seasonality (bi-yearly + weekly) with linear trend
- LogNormal likelihood for positive-valued demand
- Posterior predictive forecast for Jan-Jun 2026 (4000 scenarios)

Optimization (Xpress):
- Multi-source generation dispatch (nuclear, hydro, gas, wind, solar)
- Capacity factors for realistic renewable intermittency modeling
- Source-specific capacity limits (MW) and marginal costs ($/MWh)
- Clean energy policy constraint (configurable minimum from non-emitting sources)
- Demand uncertainty from PyMC Bayesian forecasts

Approaches Compared:
1. Chance-Constrained: Meet 95th percentile demand (5% shortage tolerance)
2. CVaR: Minimize expected cost in worst 5% of scenarios (tail risk)

NOTE: Approach 2 requires a full Xpress licence - the CVaR model exceeds
the community edition limits. 
Approach 1 runs without restriction on the community edition. 
Free 30-day trial: https://www.fico.com/en/fico-xpress-trial-and-licensing-options

Pareto Analysis:
- Traces cost-reliability frontiers for different risk profiles and clean energy policies
- Quantifies trade-offs to support policy decision-making

Outputs:
- Optimal generation mix for both approaches
- Cost vs reliability trade-off comparison
- Pareto frontiers by clean energy policy

(c) 2026 PyMC and Fair Isaac Corporation

"""

import xpress as xp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

# -- Paths and files ----------------------------------------------------------

HERE = Path(__file__).parent
NC_FILE = HERE / "ontario-demand-forecast-2026-jan-july.nc"
CSV_FILE = HERE / "ontario-demand-data-2024-2025-daily.csv"

# -- Configuration ------------------------------------------------------------

CONFIDENCE_LEVEL = 0.95  # For chance constraints
ALPHA_CVAR = 0.05  # CVaR tail: penalize the worst 5% of scenarios
N_SCENARIOS_CVAR = 1000  # Sample size for CVaR (due to computational tractability)
CVAR_WEIGHT = 500  # Weight on CVaR term in objective (tuned to explore trade-offs)
CLEAN_ENERGY_MIN = 0.80  # Policy: at least 80% from non-emitting sources
SHORTAGE_PENALTY = 500  # $/MWh
HDI_PROB = 0.94  # HDI probability for posterior predictive intervals
FORECAST_START = "2026-01-01"  # Start of out-of-sample forecast period
FORECAST_END = "2026-06-30"  # End of out-of-sample forecast period

# Generation sources - capacity (MW), costs ($/MWh), and capacity factors
# Capacity factor reflects average availability (weather-dependent for renewables)
# Costs include O&M even for renewables (no fuel cost, but maintenance)
SOURCES = {
    "nuclear": {
        "capacity": 13000,
        "cost": 30,
        "clean": True,
        "capacity_factor": 0.90,
    },
    "hydro": {
        "capacity": 9000,
        "cost": 20,
        "clean": True,
        "capacity_factor": 0.50,
    },
    "gas": {
        "capacity": 10000,
        "cost": 80,
        "clean": False,
        "capacity_factor": 0.85,
    },
    "wind": {
        "capacity": 5000,
        "cost": 5,
        "clean": True,
        "capacity_factor": 0.35,
    },
    "solar": {
        "capacity": 1000,
        "cost": 8,
        "clean": True,
        "capacity_factor": 0.20,
    },
}


# =============================================================================
# Utility functions
# =============================================================================


def _solver_controls(prob: xp.problem) -> None:
    """Apply standard solver controls for barrier method."""
    prob.controls.outputlog = 0  # Disable all solver output
    prob.controls.lpflags = 4  # Bit 2 set -> use barrier method
    prob.controls.crossover = 0  # No crossover: return interior-point solution directly
    prob.controls.bargaptarget = (
        1e-5  # Barrier convergence: target relative duality gap
    )
    prob.controls.baralg = (
        2  # Homogeneous self-dual barrier (robust for degenerate LPs)
    )


def calculate_shortage_rate(
    generation_schedule: np.ndarray, scenarios: np.ndarray
) -> float:
    """Calculate percentage of (scenario, day) pairs where demand exceeds generation."""
    # Broadcast generation_schedule (n_days,) against scenarios (n_scenarios, n_days)
    # so each day's generation is compared against every scenario row at once.
    return float(np.mean(scenarios > generation_schedule[np.newaxis, :]) * 100)


# =============================================================================
# 1. Bayesian Demand Forecast (PyMC)
# =============================================================================
# Fits a time-series model on Ontario historical daily demand (2024-2025) and
# produces an out-of-sample posterior predictive forecast for Jan-Jun 2026.
#
# Model structure:
#   mu(t) = intercept + trend(t) + fourier_biyearly(t) + fourier_weekly(t)
#   y(t)  ~ LogNormal(mu(t), sigma)          # positive-valued demand
#
# Fourier seasonality uses sin/cos pairs at bi-yearly and weekly frequencies.
# Posterior is sampled with NUTS (4 chains x 1000 draws = 4,000 scenarios).
# Result is cached to NetCDF to avoid re-sampling on subsequent runs.
# =============================================================================


def forecast() -> Path:
    """Run PyMC Bayesian sampling or load from cache.

    If NC_FILE already exists, sampling is skipped (cache checkpoint).
    Returns the Path to the NetCDF forecast file.
    """
    if NC_FILE.exists():
        print(f"[PyMC] Cached forecast found at {NC_FILE} -- skipping sampling.")
        return NC_FILE

    print("[PyMC] No cached forecast found -- running Bayesian sampling...")

    import arviz as az
    import pymc as pm

    # -- Load data ------------------------------------------------------------

    df = pd.read_csv(CSV_FILE)
    df["Time period (UTC)"] = pd.to_datetime(df["Time period (UTC)"])

    dates = df["Time period (UTC)"].to_numpy()
    y_data = df["Observation value"].to_numpy()

    # -- Feature engineering --------------------------------------------------

    harmonics = ["sin", "cos"]
    coords = {"date": dates, "harmonic": harmonics}
    trend_index = np.arange(0, len(y_data))
    full_series = np.linspace(-np.pi, np.pi, len(y_data))
    x_biyearly = 4 * full_series
    x_weekly = 104 * full_series

    def build_fourier_design_matrix(x):
        return np.column_stack([np.sin(x), np.cos(x)])

    X_yearly = build_fourier_design_matrix(x_biyearly)
    X_weekly = build_fourier_design_matrix(x_weekly)

    # -- Model definition -----------------------------------------------------

    with pm.Model(coords=coords) as model:
        data_yearly = pm.Data("X_yearly", X_yearly, dims=("date", "harmonic"))
        data_weekly = pm.Data("X_weekly", X_weekly, dims=("date", "harmonic"))
        trend_data = pm.Data("trend_data", trend_index, dims="date")
        y_data_pm = pm.Data("y_data", y_data, dims="date")

        fourier_betas_yearly = pm.Normal(
            "fourier_betas_yearly", mu=0, sigma=0.5, dims="harmonic"
        )
        fourier_betas_weekly = pm.Normal(
            "fourier_betas_weekly", mu=0, sigma=0.25, dims="harmonic"
        )
        fourier_component_yearly = pm.Deterministic(
            "fourier_component_yearly",
            fourier_betas_yearly @ data_yearly.T,
            dims="date",
        )
        fourier_component_weekly = pm.Deterministic(
            "fourier_component_weekly",
            fourier_betas_weekly @ data_weekly.T,
            dims="date",
        )
        fourier_component = pm.Deterministic(
            "fourier_component",
            fourier_component_yearly + fourier_component_weekly,
            dims="date",
        )

        intercept = pm.Normal("intercept", mu=13, sigma=0.5)
        trend_effect = pm.Normal("trend_effect", mu=0, sigma=0.1)
        trend = pm.Deterministic("trend", trend_data * trend_effect, dims="date")
        combined = intercept + fourier_component + trend
        noise = pm.Exponential("noise", scale=0.25)

        _ = pm.LogNormal("y", mu=combined, sigma=noise, observed=y_data_pm, dims="date")

    # -- Sampling -------------------------------------------------------------

    with model:
        trace = pm.sample(tune=1000, draws=1000, chains=4, cores=1)

    # -- Posterior predictive check -------------------------------------------

    with model:
        trace.extend(pm.sample_posterior_predictive(trace, var_names=["y"]))

    y_pred = trace.posterior_predictive.y
    y_pred_mean = y_pred.mean(dim=["chain", "draw"])
    y_pred_hdi = az.hdi(
        trace.posterior_predictive, var_names=["y"], hdi_prob=HDI_PROB
    ).y

    # -- Out-of-sample forecast: Jan-Jun 2026 ---------------------------------

    future_dates = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq="D")
    X_yearly_future = X_yearly[: len(future_dates), :]
    X_weekly_future = X_weekly[: len(future_dates), :]
    blank_y = np.zeros(len(future_dates), dtype=np.int32)
    trend_index_future = np.arange(len(y_data), len(y_data) + len(future_dates))

    with model:
        pm.set_data(
            new_data={
                "X_yearly": X_yearly_future,
                "X_weekly": X_weekly_future,
                "trend_data": trend_index_future,
                "y_data": blank_y,
            },
            coords={"date": future_dates},
        )

    with model:
        forecast_trace = pm.sample_posterior_predictive(trace)

    # -- Save NC --------------------------------------------------------------

    forecast_trace.posterior_predictive.to_netcdf(NC_FILE)
    print(f"[PyMC] Forecast saved to {NC_FILE}")

    # -- Diagnostic plots (2x2) -----------------------------------------------

    fc_mean = forecast_trace.posterior_predictive.y.mean(dim=["chain", "draw"])
    fc_hdi = az.hdi(
        forecast_trace.posterior_predictive, var_names=["y"], hdi_prob=HDI_PROB
    ).y

    from scipy.stats import skew

    skewnesses = [
        skew(forecast_trace.posterior_predictive.y.isel(date=i).values.flatten())
        for i in range(len(future_dates))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Raw historical demand
    ax = axes[0, 0]
    ax.plot(dates, y_data, "o-", markersize=2)
    ax.set_ylabel("Demand (MWh)")
    ax.set_title("Daily electricity demand in Ontario")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Top-right: Posterior predictive check
    ax = axes[0, 1]
    ax.plot(
        dates,
        y_data,
        "-o",
        markersize=2,
        label="Observed data",
        color="black",
        alpha=0.6,
    )
    ax.plot(dates, y_pred_mean, label="Posterior mean", color="tab:blue", linewidth=1)
    ax.fill_between(
        dates,
        y_pred_hdi.sel(hdi="lower"),
        y_pred_hdi.sel(hdi="higher"),
        alpha=0.3,
        color="tab:blue",
        label="94% HDI",
    )
    ax.set_ylabel("Electricity Demand (MWh)")
    ax.set_title("Posterior Predictive Check: Model vs Observed Data")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Forecast Jan-Jun 2026
    ax = axes[1, 0]
    ax.plot(
        future_dates, fc_mean, label="Posterior mean", color="tab:blue", linewidth=1
    )
    ax.fill_between(
        future_dates,
        fc_hdi.sel(hdi="lower"),
        fc_hdi.sel(hdi="higher"),
        alpha=0.3,
        color="tab:blue",
        label="94% HDI",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity Demand (MWh)")
    ax.set_title("PyMC Forecast: Jan-Jun 2026")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Skewness
    ax = axes[1, 1]
    ax.plot(future_dates, skewnesses, "o-", markersize=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Skewness")
    ax.set_title("Skewness of posterior predictive distribution per date")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HERE / "pymc_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[PyMC] Forecast complete.")
    return NC_FILE


# =============================================================================
# 2. Approach 1 - Chance-Constrained Optimization
# =============================================================================
# Deterministic approximation: replace the stochastic demand with its
# (1 - alpha)-th percentile, guaranteeing feasibility in at least that fraction
# of scenarios by construction.
#
# Formulation per day t:
#   min  Sum_s  cost_s * gen_s(t)
#   s.t. Sum_s  gen_s(t)  >= demand_pct(t)      # meet percentile demand
#        gen_s(t)         <= cap_s               # capacity limit
#        Sum_{clean s}    >= clean_min * total   # clean energy policy
# =============================================================================


def solve_chance_constrained(
    demand_scenarios: np.ndarray,
    n_days: int,
    sources: dict = SOURCES,
    clean_min: float = CLEAN_ENERGY_MIN,
    confidence: float = CONFIDENCE_LEVEL,
) -> dict:
    """Deterministic approximation: meet the (confidence)-th percentile demand.

    Returns dict with keys: gen_by_source, total_gen, cost, clean_pct.
    """
    source_names = list(sources.keys())
    clean_sources = [s for s in source_names if sources[s]["clean"]]

    print(
        f"\n[2/7] Solving Approach 1: Chance Constraints "
        f"({confidence * 100}% reliability)..."
    )
    print(f"       Sources: {', '.join(source_names)}")
    print(f"       Clean energy minimum: {clean_min * 100}%")

    # Deterministic demand target: (confidence)-th percentile per day
    demand_pct = np.percentile(demand_scenarios, confidence * 100, axis=0)

    prob = xp.problem(name="Chance_Constrained_MultiSource")
    _solver_controls(prob)

    # Decision variables - generation by source and day
    gen = {
        src: prob.addVariables(n_days, lb=0, name=f"gen_{src}") for src in source_names
    }
    total_gen = [xp.Sum([gen[src][t] for src in source_names]) for t in range(n_days)]

    # Meet demand at the chosen percentile
    prob.addConstraint(total_gen[t] >= demand_pct[t] for t in range(n_days))

    # Capacity constraints (capacity x capacity_factor x 24h = available MWh/day)
    for src in source_names:
        cap_mwh = sources[src]["capacity"] * sources[src]["capacity_factor"] * 24
        prob.addConstraint(gen[src][t] <= cap_mwh for t in range(n_days))

    # Clean energy constraint
    for t in range(n_days):
        clean_gen = xp.Sum([gen[src][t] for src in clean_sources])
        prob.addConstraint(clean_gen >= clean_min * total_gen[t])

    # Objective: minimize generation cost
    total_cost = xp.Sum(
        [
            sources[src]["cost"] * xp.Sum([gen[src][t] for t in range(n_days)])
            for src in source_names
        ]
    )
    prob.setObjective(total_cost, sense=xp.minimize)

    prob.optimize()

    if prob.attributes.solstatus not in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
        max_clean = sum(
            sources[s]["capacity"] * sources[s]["capacity_factor"] * 24
            for s in clean_sources
        )
        raise RuntimeError(
            f"ERROR: Could not find a feasible solution to the chance-constrained model"
            f"(solstatus={prob.attributes.solstatus}). "
            f"Max clean capacity: {max_clean:,.0f} MWh/day, "
            f"max demand ({confidence * 100:.0f}th): {demand_pct.max():,.0f} MWh"
        )

    # Extract solution
    gen_by_source = {src: np.array(prob.getSolution(gen[src])) for src in source_names}
    total_gen_vals = np.sum(list(gen_by_source.values()), axis=0)
    cost = prob.attributes.objval
    clean_gen_vals = np.sum([gen_by_source[s] for s in clean_sources], axis=0)
    clean_pct = clean_gen_vals.sum() / total_gen_vals.sum() * 100

    shortage_rate = calculate_shortage_rate(total_gen_vals, demand_scenarios)

    result = {
        "gen_by_source": gen_by_source,
        "total_gen": total_gen_vals,
        "cost": cost,
        "clean_pct": clean_pct,
        "shortage_rate": shortage_rate,
    }

    print(f"   Total generation cost: ${cost:,.0f}")
    print(f"   Average daily generation: {total_gen_vals.mean():,.0f} MWh")
    print(f"   Clean energy: {clean_pct:.1f}%")
    print(f"   {'Shortage rate (Monte Carlo):':<30} {shortage_rate:.2f}%")
    print("   Generation mix (avg MWh/day):")
    for src in source_names:
        avg = gen_by_source[src].mean()
        pct = avg / total_gen_vals.mean() * 100
        print(f"      {src:8s}: {avg:>10,.0f} ({pct:5.1f}%)")

    return result


# =============================================================================
# 3. Approach 2 - CVaR (Conditional Value-at-Risk)
# =============================================================================
# CVaR answers: "What is the expected cost in the worst alpha% of scenarios?"
#
# Rockafellar-Uryasev LP reformulation:
#   CVaR_alpha = min_z { z + (1/alpha) x E[max(0, Loss - z)] }
#
# For discrete scenarios:
#   CVaR ~ z + (1 / (n_scenarios x alpha)) x Sum aux[s]
#   aux[s] >= 0,  aux[s] >= shortage_cost[s] - z
# =============================================================================


def solve_cvar(
    demand_scenarios: np.ndarray,
    n_days: int,
    sources: dict = SOURCES,
    clean_min: float = CLEAN_ENERGY_MIN,
    alpha: float = ALPHA_CVAR,
    n_scenarios: int = N_SCENARIOS_CVAR,
    cvar_weight: float = CVAR_WEIGHT,
    shortage_pen: float = SHORTAGE_PENALTY,
    scenario_subset: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Minimize generation cost + weighted CVaR of shortage cost.

    If scenario_subset is provided it is used directly (caller controls sampling);
    otherwise n_scenarios are drawn at random from demand_scenarios.
    Returns dict with keys: gen_by_source, total_gen, cost, clean_pct.
    """
    source_names = list(sources.keys())
    clean_sources = [s for s in source_names if sources[s]["clean"]]

    if scenario_subset is not None:
        cvar_scenarios = scenario_subset
        n_scenarios = cvar_scenarios.shape[0]
    else:
        selected = np.random.choice(
            demand_scenarios.shape[0], n_scenarios, replace=False
        )
        cvar_scenarios = demand_scenarios[selected, :]

    if verbose:
        print(
            f"\n[3/7] Solving Approach 2: CVaR (alpha={alpha}, {n_scenarios} scenarios)..."
        )
        print(f"       Sources: {', '.join(source_names)}")
        print(f"       Clean energy minimum: {clean_min * 100}%")

    prob = xp.problem(name="CVaR_MultiSource")
    _solver_controls(prob)

    # Decision variables - generation by source and day
    gen = {
        src: prob.addVariables(n_days, lb=0, name=f"gen_{src}") for src in source_names
    }
    total_gen = [xp.Sum([gen[src][t] for src in source_names]) for t in range(n_days)]

    # CVaR variables
    z = prob.addVariable(name="var", lb=-xp.infinity)  # Value at Risk
    aux = prob.addVariables(n_scenarios, lb=0, name="aux")
    shortage = prob.addVariables(n_scenarios, n_days, lb=0, name="shortage")

    # Capacity constraints (capacity x capacity_factor x 24h)
    for src in source_names:
        cap_mwh = sources[src]["capacity"] * sources[src]["capacity_factor"] * 24
        prob.addConstraint(gen[src][t] <= cap_mwh for t in range(n_days))

    # Clean energy constraint
    for t in range(n_days):
        clean_gen = xp.Sum([gen[src][t] for src in clean_sources])
        prob.addConstraint(clean_gen >= clean_min * total_gen[t])

    # Shortage definition: shortage[s,t] >= demand[s,t] - total_generation[t]
    prob.addConstraint(
        shortage[s, t] >= cvar_scenarios[s, t] - total_gen[t]
        for s in range(n_scenarios)
        for t in range(n_days)
    )

    # CVaR constraints: aux[s] >= shortage_cost[s] - z
    pen_coef = xp.array([shortage_pen] * n_days)
    for s in range(n_scenarios):
        shortage_cost_s = xp.Dot(pen_coef, shortage[s, :])
        prob.addConstraint(aux[s] >= shortage_cost_s - z)

    # Objective: generation costs + weighted CVaR of shortage costs
    gen_cost = xp.Sum(
        [
            sources[src]["cost"] * xp.Sum([gen[src][t] for t in range(n_days)])
            for src in source_names
        ]
    )
    cvar_term = z + (1 / (n_scenarios * alpha)) * xp.Sum(aux)
    prob.setObjective(gen_cost + cvar_weight * cvar_term, sense=xp.minimize)

    prob.optimize()

    if prob.attributes.solstatus not in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
        raise RuntimeError(
            f"CVaR model infeasible (solstatus={prob.attributes.solstatus})."
        )

    # Extract solution
    gen_by_source = {src: np.array(prob.getSolution(gen[src])) for src in source_names}
    total_gen_vals = np.sum(list(gen_by_source.values()), axis=0)
    cost = sum(sources[src]["cost"] * gen_by_source[src].sum() for src in source_names)
    clean_gen_vals = np.sum([gen_by_source[s] for s in clean_sources], axis=0)
    clean_pct = clean_gen_vals.sum() / total_gen_vals.sum() * 100

    result = {
        "gen_by_source": gen_by_source,
        "total_gen": total_gen_vals,
        "cost": cost,
        "clean_pct": clean_pct,
    }

    if verbose:
        print(f"   Generation cost: ${cost:,.0f}")
        print(f"   Average daily generation: {total_gen_vals.mean():,.0f} MWh")
        print(f"   Clean energy: {clean_pct:.1f}%")
        print("   Generation mix (avg MWh/day):")
        for src in source_names:
            avg = gen_by_source[src].mean()
            pct = avg / total_gen_vals.mean() * 100
            print(f"      {src:8s}: {avg:>10,.0f} ({pct:5.1f}%)")

    return result


# =============================================================================
# 4. Pareto Analysis - Cost vs Shortage Trade-off
# =============================================================================


def pareto_analysis(
    demand_scenarios: np.ndarray,
    n_days: int,
    cc_cost_b: float,
    cc_shortage: float,
    sources: dict = SOURCES,
    clean_levels: list | None = None,
    cvar_weights: list | None = None,
    n_scenarios: int = N_SCENARIOS_CVAR,
    shortage_pen: float = SHORTAGE_PENALTY,
    alpha: float = ALPHA_CVAR,
) -> dict:
    """Trace Pareto frontiers across CVaR weights and clean energy levels.

    Generates and saves the Pareto front plot.
    Returns pareto_data dict: {clean_level: [(cost_B, shortage_pct), ...]}.
    """
    if clean_levels is None:
        clean_levels = [0.80, 0.90]
    if cvar_weights is None:
        cvar_weights = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    print("\n[7/7] Generating Pareto front (Cost vs Shortage)...")

    # Sample scenarios once - all weights for a given clean level use the same
    # subset so results are directly comparable.
    selected = np.random.choice(demand_scenarios.shape[0], n_scenarios, replace=False)
    cvar_scenarios = demand_scenarios[selected, :]

    # -- Sweep weights across clean energy levels -----------------------------

    print(
        f"   Testing {len(cvar_weights)} CVAR_WEIGHT values "
        f"across {len(clean_levels)} clean energy levels..."
    )

    pareto_data: dict = {}

    for clean_min in clean_levels:
        pareto_data[clean_min] = []
        print(f"\n   Clean energy >= {clean_min * 100:.0f}%:")
        for w in cvar_weights:
            result = solve_cvar(
                demand_scenarios,
                n_days,
                sources=sources,
                clean_min=clean_min,
                alpha=alpha,
                n_scenarios=n_scenarios,
                cvar_weight=w,
                shortage_pen=shortage_pen,
                scenario_subset=cvar_scenarios,
                verbose=False,
            )
            cost, gen_schedule = result["cost"], result["total_gen"]
            shortage_rate = calculate_shortage_rate(gen_schedule, demand_scenarios)
            pareto_data[clean_min].append((cost / 1e9, shortage_rate))
            print(
                f"      w={w:>4}: Cost=${cost / 1e9:.3f}B, Shortage={shortage_rate:.2f}%"
            )

    # -- Plot Pareto fronts ---------------------------------------------------

    _, ax = plt.subplots(figsize=(11, 7))

    colors = {0.80: "#e74c3c", 0.90: "#3498db"}
    labels = {0.80: "80% Clean", 0.90: "90% Clean"}
    markers_map = {0.80: "o", 0.90: "s"}
    zorders = {0.80: 3, 0.90: 4}

    for clean_min in sorted(clean_levels):
        costs = [p[0] for p in pareto_data[clean_min]]
        shortages = [p[1] for p in pareto_data[clean_min]]
        sorted_pairs = sorted(zip(shortages, costs))
        shortages_sorted = [p[0] for p in sorted_pairs]
        costs_sorted = [p[1] for p in sorted_pairs]

        linestyle = "-" if clean_min == 0.90 else "--"
        ax.plot(
            shortages_sorted,
            costs_sorted,
            color=colors[clean_min],
            linewidth=3,
            linestyle=linestyle,
            marker=markers_map[clean_min],
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=2,
            label=f"{labels[clean_min]} Policy",
            zorder=zorders[clean_min],
        )

    # Chance-constrained reference point
    ax.scatter(
        [cc_shortage],
        [cc_cost_b],
        s=200,
        c="gold",
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label="Chance-Constrained\n(80% Clean)",
    )

    # Annotate best-reliability point per frontier
    for clean_min in clean_levels:
        costs = [p[0] for p in pareto_data[clean_min]]
        shortages = [p[1] for p in pareto_data[clean_min]]
        min_idx = shortages.index(min(shortages))
        ax.annotate(
            f"{shortages[min_idx]:.2f}%\n${costs[min_idx]:.2f}B",
            (shortages[min_idx], costs[min_idx]),
            textcoords="offset points",
            xytext=(-40, 10),
            fontsize=8,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    ax.annotate(
        f"CC: {cc_shortage:.1f}%\n${cc_cost_b:.2f}B",
        (cc_shortage, cc_cost_b),
        textcoords="offset points",
        xytext=(15, -25),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    ax.set_xlabel("Shortage Rate (%)", fontsize=13)
    ax.set_ylabel("Generation Cost ($B)", fontsize=13)
    ax.set_title(
        "Pareto Frontiers: Cost vs Reliability by Clean Energy Policy",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    all_shortages = [p[1] for cl in clean_levels for p in pareto_data[cl]]
    all_costs = [p[0] for cl in clean_levels for p in pareto_data[cl]]
    ax.set_xlim(-0.5, max(all_shortages) + 1)
    ax.set_ylim(min(all_costs) - 0.2, max(all_costs) + 0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1)

    plt.tight_layout()
    pareto_file = HERE / "pareto_cost_shortage.png"
    plt.savefig(pareto_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n   Saved Pareto front: {pareto_file}")

    return pareto_data


# =============================================================================
# Visualization helpers
# =============================================================================


def plot_forecast_uncertainty(
    demand_scenarios: np.ndarray,
    n_days: int,
    daily_training: pd.DataFrame | None = None,
) -> None:
    """Two-panel plot: historical + forecast uncertainty, and seasonal pattern."""
    demand_mean = np.mean(demand_scenarios, axis=0)
    demand_05 = np.percentile(demand_scenarios, 5, axis=0)
    demand_25 = np.percentile(demand_scenarios, 25, axis=0)
    demand_75 = np.percentile(demand_scenarios, 75, axis=0)
    demand_95 = np.percentile(demand_scenarios, 95, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Historical + Forecast with uncertainty bands
    ax1 = axes[0]
    if daily_training is not None:
        training_days = np.arange(-len(daily_training), 0)
        ax1.plot(
            training_days,
            daily_training["daily_MWh"].values,
            color="darkgray",
            linewidth=0.8,
            alpha=0.7,
            label="Historical (2024-2025)",
        )
        ax1.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
        y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        ax1.text(
            5,
            ax1.get_ylim()[0] + 0.03 * y_range,
            "Forecast Start",
            fontsize=9,
            alpha=0.7,
        )

    forecast_days = np.arange(n_days)
    ax1.fill_between(
        forecast_days,
        demand_05,
        demand_95,
        alpha=0.2,
        color="steelblue",
        label="5th-95th percentile",
    )
    ax1.fill_between(
        forecast_days,
        demand_25,
        demand_75,
        alpha=0.4,
        color="steelblue",
        label="25th-75th percentile",
    )
    ax1.plot(forecast_days, demand_mean, "b-", linewidth=2, label="Mean forecast")
    ax1.plot(
        forecast_days,
        demand_95,
        "r-",
        linewidth=1.5,
        alpha=0.8,
        label="95th percentile (CC threshold)",
    )
    ax1.set_xlabel("Days (relative to Jan 1, 2026)", fontsize=11)
    ax1.set_ylabel("Demand (MWh/day)", fontsize=11)
    ax1.set_title(
        "Ontario Electricity Demand: Historical + PyMC Forecast",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Seasonal pattern or sample scenarios
    ax2 = axes[1]
    if daily_training is not None:
        daily_training["month"] = daily_training["date"].dt.month
        monthly_avg = daily_training.groupby("month")["daily_MWh"].mean()
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        x_months = np.arange(12)
        bars = ax2.bar(
            x_months,
            [monthly_avg.get(m + 1, 0) for m in range(12)],
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
            label="Historical monthly avg",
        )
        for i in range(6):
            bars[i].set_color("coral")
            bars[i].set_alpha(0.8)
        ax2.axhline(
            y=demand_mean.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Forecast mean: {demand_mean.mean():,.0f}",
        )
        ax2.set_xticks(x_months)
        ax2.set_xticklabels(month_names, fontsize=9)
        ax2.set_xlabel("Month", fontsize=11)
        ax2.set_ylabel("Avg Daily Demand (MWh)", fontsize=11)
        ax2.set_title(
            "Seasonal Pattern: Historical vs Forecast", fontsize=12, fontweight="bold"
        )
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.annotate(
            "Forecast\nperiod",
            xy=(2.5, monthly_avg.mean() * 0.9),
            fontsize=10,
            ha="center",
            color="coral",
            fontweight="bold",
        )
    else:
        n_total = demand_scenarios.shape[0]
        sample_indices = np.random.choice(n_total, 50, replace=False)
        for idx in sample_indices:
            ax2.plot(
                forecast_days,
                demand_scenarios[idx, :],
                "gray",
                alpha=0.15,
                linewidth=0.5,
            )
        ax2.plot(forecast_days, demand_mean, "b-", linewidth=2, label="Mean")
        ax2.plot(forecast_days, demand_95, "r-", linewidth=2, label="95th percentile")
        ax2.fill_between(
            forecast_days, demand_05, demand_95, alpha=0.1, color="steelblue"
        )
        ax2.set_xlabel("Day (Jan-Jun 2026)", fontsize=11)
        ax2.set_ylabel("Demand (MWh)", fontsize=11)
        ax2.set_title(
            "50 Sample Scenarios from PyMC Posterior", fontsize=12, fontweight="bold"
        )
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = HERE / "pymc_forecast_uncertainty.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out}")


def plot_comparison(
    cc: dict,
    cvar: dict,
    demand_scenarios: np.ndarray,
    n_days: int,
    shortage_rate_cc: float,
    shortage_rate_cvar: float,
    sources: dict = SOURCES,
) -> None:
    """Four-panel comparison: plans vs demand, mix, 30-day zoom, metrics."""
    from matplotlib.patches import Patch

    demand_mean = np.mean(demand_scenarios, axis=0)
    demand_05 = np.percentile(demand_scenarios, 5, axis=0)
    demand_95 = np.percentile(demand_scenarios, 95, axis=0)

    source_names = list(sources.keys())
    source_colors = {
        "nuclear": "#1f77b4",
        "hydro": "#2ca02c",
        "gas": "#d62728",
        "wind": "#9467bd",
        "solar": "#ff7f0e",
    }

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Demand Uncertainty and Optimal Plans
    ax1 = plt.subplot(2, 2, 1)
    ax1.fill_between(
        range(n_days),
        demand_05,
        demand_95,
        alpha=0.2,
        color="gray",
        label="5th-95th percentile",
    )
    ax1.plot(demand_mean, "k--", label="Mean forecast", linewidth=2, alpha=0.7)
    ax1.plot(demand_95, "r-", label="95th percentile", linewidth=2, alpha=0.7)
    ax1.plot(cc["total_gen"], "b-", label="Chance-Constrained plan", linewidth=2)
    ax1.plot(cvar["total_gen"], "g-", label="CVaR plan", linewidth=2)
    ax1.set_xlabel("Day (Jan-Jun 2026)", fontsize=11)
    ax1.set_ylabel("Demand / Generation (MWh)", fontsize=11)
    ax1.set_title(
        "Demand Uncertainty and Optimal Generation Plans",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Generation Mix (Stacked Bar)
    ax2 = plt.subplot(2, 2, 2)
    approaches = ["Chance-\nConstrained", "CVaR"]
    x_pos = np.arange(len(approaches))
    width = 0.6
    bottom_cc = 0
    bottom_cvar = 0
    for src in source_names:
        cc_val = cc["gen_by_source"][src].mean()
        cvar_val = cvar["gen_by_source"][src].mean()
        ax2.bar(
            x_pos[0],
            cc_val,
            width,
            bottom=bottom_cc,
            color=source_colors[src],
            edgecolor="white",
        )
        ax2.bar(
            x_pos[1],
            cvar_val,
            width,
            bottom=bottom_cvar,
            color=source_colors[src],
            edgecolor="white",
        )
        bottom_cc += cc_val
        bottom_cvar += cvar_val

    legend_elements = [
        Patch(
            facecolor=source_colors[s],
            label=f"{s.capitalize()} (${sources[s]['cost']}/MWh)",
        )
        for s in source_names
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(approaches)
    ax2.set_ylabel("Average Generation (MWh/day)", fontsize=11)
    ax2.set_title("Generation Mix by Source", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: First 30 Days Comparison
    ax3 = plt.subplot(2, 2, 3)
    days_to_show = 30
    x_days = range(days_to_show)
    ax3.fill_between(
        x_days,
        demand_mean[:days_to_show],
        demand_95[:days_to_show],
        alpha=0.15,
        color="gray",
        label="Demand range (mean-95th)",
    )
    ax3.plot(
        x_days,
        cc["total_gen"][:days_to_show],
        "b-",
        linewidth=2.5,
        label="Chance-Constrained",
        marker="o",
        markersize=4,
    )
    ax3.plot(
        x_days,
        cvar["total_gen"][:days_to_show],
        "g-",
        linewidth=2.5,
        label="CVaR",
        marker="s",
        markersize=4,
    )
    ax3.fill_between(
        x_days,
        cc["total_gen"][:days_to_show],
        cvar["total_gen"][:days_to_show],
        alpha=0.3,
        color="yellow",
        label="Generation difference",
    )
    ax3.set_xlabel("Day (Jan-Feb 2026)", fontsize=11)
    ax3.set_ylabel("Generation (MWh)", fontsize=11)
    ax3.set_title(
        "Generation Plan Comparison (First 30 Days)", fontsize=12, fontweight="bold"
    )
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance Metrics
    ax4 = plt.subplot(2, 2, 4)
    metrics = [
        "Total Cost\n($M)",
        "Shortage\nRate (%)",
        "Avg Daily\nGen (GWh)",
        "Clean\nEnergy (%)",
    ]
    cc_values = [
        cc["cost"] / 1e6,
        shortage_rate_cc,
        cc["total_gen"].mean() / 1000,
        cc["clean_pct"],
    ]
    cvar_values = [
        cvar["cost"] / 1e6,
        shortage_rate_cvar,
        cvar["total_gen"].mean() / 1000,
        cvar["clean_pct"],
    ]

    cc_norm = []
    cvar_norm = []
    for i in range(len(metrics)):
        mx = max(cc_values[i], cvar_values[i])
        if mx > 0:
            cc_norm.append(cc_values[i] / mx * 100)
            cvar_norm.append(cvar_values[i] / mx * 100)
        else:
            cc_norm.append(0)
            cvar_norm.append(0)

    x_pos = np.arange(len(metrics))
    width = 0.35
    bars1 = ax4.bar(
        x_pos - width / 2,
        cc_norm,
        width,
        label="Chance-Constrained",
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )
    bars2 = ax4.bar(
        x_pos + width / 2,
        cvar_norm,
        width,
        label="CVaR",
        color="forestgreen",
        alpha=0.8,
        edgecolor="black",
    )

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if "Cost" in metrics[i]:
            t1, t2 = f"${cc_values[i]:.0f}M", f"${cvar_values[i]:.0f}M"
        elif "Rate" in metrics[i]:
            t1, t2 = f"{cc_values[i]:.2f}%", f"{cvar_values[i]:.2f}%"
        else:
            t1, t2 = f"{cc_values[i]:.0f}", f"{cvar_values[i]:.0f}"
        ax4.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height() + 2,
            t1,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        ax4.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + 2,
            t2,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax4.set_ylabel("Relative Scale (% of maximum)", fontsize=11)
    ax4.set_title("Performance Metrics Comparison", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.legend(loc="upper right", fontsize=10)
    ax4.set_ylim(0, 120)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = HERE / "robust_comparison_results.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out}")


def plot_cc_schedule(cc: dict, n_days: int, sources: dict = SOURCES) -> None:
    """Stacked bar chart of the chance-constrained generation schedule."""
    source_names = list(sources.keys())
    colors = ["steelblue", "mediumseagreen", "tomato", "skyblue", "gold"]
    future_dates = pd.date_range(start=FORECAST_START, periods=n_days, freq="D")

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(n_days)
    for src, color in zip(source_names, colors):
        ax.bar(
            future_dates,
            cc["gen_by_source"][src],
            bottom=bottom,
            label=src.capitalize(),
            color=color,
            width=1,
        )
        bottom += cc["gen_by_source"][src]
    ax.set_xlabel("Date")
    ax.set_ylabel("Generation (MWh/day)")
    ax.set_title("Approach 1: Chance-Constrained Generation Schedule (Jan–Jun 2026)")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = HERE / "cc_generation_schedule.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out}")


# =============================================================================
# Data loading and comparison
# =============================================================================


def load_forecast_data(nc_file: Path) -> tuple[np.ndarray, int]:
    """Load PyMC forecast from NetCDF, load training data, plot uncertainty.

    Returns (demand_scenarios, n_days).
    """
    print("\n[1/7] Loading PyMC forecast data...")

    with xr.open_dataset(nc_file, engine="netcdf4") as ds:
        y_samples = ds["y"].values  # (4, 1000, 181)

    demand_scenarios = y_samples.reshape(-1, y_samples.shape[-1])
    n_scenarios_total = demand_scenarios.shape[0]
    n_days = demand_scenarios.shape[1]

    print(f"   Loaded {n_scenarios_total} scenarios for {n_days} days")
    print(f"   Mean demand: {demand_scenarios.mean():.0f} MWh")
    print(f"   Std demand: {demand_scenarios.std():.0f} MWh")
    print(f"   5th percentile: {np.percentile(demand_scenarios, 5):.0f} MWh")
    print(f"   95th percentile: {np.percentile(demand_scenarios, 95):.0f} MWh")

    # Load training data (historical)
    print("\n   Loading real Ontario demand data (training period)...")
    training_data_file = HERE / "HFED_ Ontario ONTARIO_DEMAND data.csv"
    daily_training = None

    if training_data_file.exists():
        df_hourly = pd.read_csv(training_data_file)
        df_hourly["MW"] = (
            df_hourly["Observation value"]
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        df_hourly["datetime_utc"] = pd.to_datetime(df_hourly["Time period (UTC)"])
        df_hourly["date"] = df_hourly["datetime_utc"].dt.date

        daily_training = (
            df_hourly.groupby("date")["MW"].agg(["sum", "count"]).reset_index()
        )
        daily_training.columns = ["date", "daily_MWh", "hours"]
        daily_training = daily_training[daily_training["hours"] == 24].copy()
        daily_training["date"] = pd.to_datetime(daily_training["date"])

        print(
            f"   Training data: {len(daily_training)} days "
            f"({daily_training['date'].min().date()} to "
            f"{daily_training['date'].max().date()})"
        )
        print(f"   Training mean: {daily_training['daily_MWh'].mean():,.0f} MWh/day")
        print(
            f"   Training 95th: "
            f"{np.percentile(daily_training['daily_MWh'], 95):,.0f} MWh/day"
        )
    else:
        print("   Training data file not found - skipping historical visualization")

    # Forecast uncertainty visualization
    print("\n   Generating PyMC forecast visualization...")
    plot_forecast_uncertainty(demand_scenarios, n_days, daily_training)

    return demand_scenarios, n_days


def compare_results(
    cc: dict,
    cvar: dict,
    demand_scenarios: np.ndarray,
    n_days: int,
    sources: dict = SOURCES,
) -> tuple[float, float]:
    """Monte Carlo validation, results comparison table, and comparison plot.

    Returns (shortage_rate_cc, shortage_rate_cvar).
    """
    source_names = list(sources.keys())

    # Monte Carlo validation
    print("\n[4/7] Validating solutions with Monte Carlo simulation...")
    shortage_rate_cc = cc["shortage_rate"]
    shortage_rate_cvar = calculate_shortage_rate(cvar["total_gen"], demand_scenarios)
    print(f"   Chance-Constrained shortage rate: {shortage_rate_cc:.2f}%")
    print(f"   CVaR shortage rate: {shortage_rate_cvar:.2f}%")

    # Results comparison
    print("\n[5/7] Comparing results...")
    cost_diff = cvar["cost"] - cc["cost"]
    cost_diff_pct = (cost_diff / cc["cost"]) * 100
    shortage_impr = (
        (shortage_rate_cc - shortage_rate_cvar) / shortage_rate_cc * 100
        if shortage_rate_cc > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Chance-Constrained':<25} {'CVaR':<25}")
    print("-" * 80)
    print(
        f"{'Avg Generation (MWh/day)':<30} {cc['total_gen'].mean():>20,.0f}   "
        f"{cvar['total_gen'].mean():>20,.0f}"
    )
    print(f"{'Total Cost ($)':<30} {cc['cost']:>20,.0f}   {cvar['cost']:>20,.0f}")
    print(
        f"{'Clean Energy (%)':<30} {cc['clean_pct']:>20.1f}   {cvar['clean_pct']:>20.1f}"
    )
    print(
        f"{'Shortage Rate (%)':<30} {shortage_rate_cc:>20.2f}   "
        f"{shortage_rate_cvar:>20.2f}"
    )
    print("-" * 80)

    print("\nGeneration Mix Comparison (avg MWh/day):")
    print(f"{'Source':<12} {'Chance-Constrained':>20} {'CVaR':>20}")
    print("-" * 54)
    for src in source_names:
        cc_val = cc["gen_by_source"][src].mean()
        cvar_val = cvar["gen_by_source"][src].mean()
        print(f"{src:<12} {cc_val:>20,.0f} {cvar_val:>20,.0f}")
    print("-" * 54)

    print(f"\nCost Difference: ${cost_diff:,.0f} ({cost_diff_pct:+.1f}%)")
    if shortage_rate_cc > 0:
        print(f"Shortage Rate Improvement: {shortage_impr:.1f}% reduction")

    # Comparison visualizations
    print("\n[6/7] Generating visualizations...")
    plot_comparison(
        cc,
        cvar,
        demand_scenarios,
        n_days,
        shortage_rate_cc,
        shortage_rate_cvar,
        sources,
    )

    return shortage_rate_cc, shortage_rate_cvar


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("PyMC + Xpress: Robust Optimization Comparison")
    print("=" * 80)

    # Step 1: Run PyMC Bayesian sampling (or load cached NetCDF if already done)
    nc_file = forecast()
    # Step 2: Load forecast scenarios and plot demand uncertainty
    demand_scenarios, n_days = load_forecast_data(nc_file)

    # Step 3: Solve Approach 1 - chance-constrained dispatch
    cc = solve_chance_constrained(demand_scenarios, n_days)

    # Step 4: Solve Approach 2 - CVaR dispatch (requires full licence)
    cvar = None
    try:
        cvar = solve_cvar(demand_scenarios, n_days)
    except xp.SolverError as e:
        if "too many rows and columns" in str(e):
            print("\n[!] Approach 2 (CVaR) requires a full Xpress licence.")
            print(
                "    The community edition is limited to 5,000 rows and columns combined."
            )
            print("    Skipping CVaR, comparison, and Pareto analysis.\n")
        else:
            raise

    if cvar is not None:
        # Step 5: Compare both approaches and plot results
        shortage_cc, _ = compare_results(cc, cvar, demand_scenarios, n_days)

        # Step 6: Trace Pareto frontiers across CVaR weights and clean energy levels
        pareto_analysis(
            demand_scenarios,
            n_days,
            cc_cost_b=cc["cost"] / 1e9,
            cc_shortage=shortage_cc,
        )
    else:
        # Community edition: show Approach 1 results and generation schedule plot.
        plot_cc_schedule(cc, n_days)
        print("\n" + "=" * 80)
        print("EXPLORE FURTHER - THIS IS A PARTIAL REPORT")
        print("=" * 80)
        print("The results above cover Approach 1 (Chance-Constrained) only.")
        print("With a full licence you also get:")
        print("  - Approach 2: CVaR minimisation with explicit tail-risk control")
        print("  - Side-by-side cost vs. shortage comparison")
        print("  - Pareto frontier across risk profiles (cost vs. shortage trade-off)")
        print("\nRequest a free trial or academic licence:")
        print("  https://www.fico.com/en/fico-xpress-trial-and-licensing-options")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
