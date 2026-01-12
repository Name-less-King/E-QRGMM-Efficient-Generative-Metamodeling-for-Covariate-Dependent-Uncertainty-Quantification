import numpy as np
import matplotlib.pyplot as plt
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
    Note: there is no n_replications here; we run only once.
    """
    rng = np.random.default_rng(seed)

    # Generate one sequence of demands and lead times
    demands = rng.exponential(mu, size=periods)
    lead_times = rng.poisson(theta, size=periods)

    # Directly use simulate_inventory_core to get the cost (total_cost / periods)
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
    # Example: input (s, S, mu)
    s_example = 320
    S_example = 420
    mu_example = 330

    n_samples = 100000  # generate 100000 cost samples
    costs = np.empty(n_samples)

    # Use a global RNG to generate different seeds so that each simulation is independent
    rng = np.random.default_rng(0)

    for i in range(n_samples):
        seed = int(rng.integers(1, 1e9))
        costs[i] = estimate_cost(s_example, S_example, mu_example, seed=seed)

    # Basic statistics
    print(f"Mean of {n_samples} costs: {costs.mean():.4f}")
    print(f"Standard deviation of {n_samples} costs: {costs.std(ddof=1):.4f}")
    print(f"80% quantile of {n_samples} costs: {np.quantile(costs, 0.8):.4f}")

    # Plot the empirical distribution (histogram)
    plt.hist(costs, bins=15, edgecolor="black")
    plt.xlabel("Average cost per period")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of {n_samples} costs\n"
        f"(s={s_example}, S={S_example}, mu={mu_example})"
    )
    plt.tight_layout()
    plt.show()
