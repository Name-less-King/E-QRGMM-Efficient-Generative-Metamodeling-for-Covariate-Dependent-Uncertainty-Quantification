import numpy as np
from inventory_core import simulate_inventory_core


def estimate_cost(
    s,
    S,
    mu,
    initial=1000,
    periods=1000,
    theta=6.0,
    h=0.5,
    f=36.0,
    c=1.0,
    seed=None,
):
    """
    Call inventory_core once and return the average cost per period.
    Note: we no longer use n_replications here; we only run the simulation once.
    """
    rng = np.random.default_rng(seed)

    # Generate a single sequence of demands and lead times
    demands = rng.exponential(mu, size=periods)
    lead_times = rng.poisson(theta, size=periods)

    # Directly use simulate_inventory_core to obtain cost (total_cost / periods)
    cost = simulate_inventory_core(
        s,
        S,
        initial,
        periods,
        h,
        f,
        c,
        demands,
        lead_times,
    )
    return float(cost)


if __name__ == "__main__":
    # Specific (s, S, mu)
    s_example = 320
    S_example = 420
    mu_example = 330

    n_samples = 100000  # Generate 100,000 costs
    costs = np.empty(n_samples)

    # Use a global RNG to generate different seeds so that each simulation is independent
    rng = np.random.default_rng(0)

    for i in range(n_samples):
        seed = int(rng.integers(1, 1e9))
        costs[i] = estimate_cost(s_example, S_example, mu_example, seed=seed)

    # Save to a CSV file without header, one cost per line
    np.savetxt(
        "costs_100000.csv",
        costs,
        delimiter=",",
        comments=""  # No header is written, so there will be no header row
    )

    print(f"Generated {n_samples} costs and saved them to costs_100000.csv")
