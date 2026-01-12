import os
import random
import numpy as np
from inventory_core import simulate_inventory_core



def estimate_cost(
    s,
    S,
    mu,
    initial=1000,
    periods=1000,
    theta=6.0,   # Poisson lead time parameter
    h=0.5,
    f=36.0,
    c=1.0,
    seed=None,
):
    """
    Call simulate_inventory_core once and return one cost (average cost per period).
    """
    rng = np.random.default_rng(seed)

    demands = rng.exponential(mu, size=periods)
    lead_times = rng.poisson(theta, size=periods)

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



def build_param_space():
    """Construct all feasible (s, S, mu) combinations."""
    s_range = range(270, 341)   # 270 <= s <= 340
    S_range = range(380, 451)   # 380 <= S <= 450
    mu_range = range(310, 341)  # 310 <= mu <= 340

    all_combos = []
    for s in s_range:
        for S in S_range:
            for mu in mu_range:
                all_combos.append((s, S, mu))
    return all_combos


def sample_params(n_samples, seed=None):
    """
    Uniformly sample (with replacement) from all feasible (s, S, mu) combinations.
    """
    if seed is not None:
        random.seed(seed)

    param_space = build_param_space()
    if not param_space:
        raise ValueError("Parameter space is empty; please check ranges and constraints.")

    samples = random.choices(param_space, k=n_samples)
    return samples



def generate_dataset(n_samples=10000, seed=0):
    """
    Generate n_samples (s, S, mu, cost) samples and return a numpy array of shape (n_samples, 4).
    """
    # Uniformly sample (s, S, mu) from the parameter space
    param_samples = sample_params(n_samples, seed=seed)

    # For reproducibility, use numpy RNG to generate a simulation seed for each sample
    rng = np.random.default_rng(seed + 12345)

    data = np.empty((n_samples, 4), dtype=float)  # columns: s, S, mu, cost

    for i, (s, S, mu) in enumerate(param_samples):
        sim_seed = int(rng.integers(0, 1_000_000_000))
        cost = estimate_cost(s, S, mu, seed=sim_seed)

        data[i, 0] = s
        data[i, 1] = S
        data[i, 2] = mu
        data[i, 3] = cost

    return data



if __name__ == "__main__":
    n_datasets = 100      # number of datasets to generate
    n_samples = 10000     # number of samples per dataset
    output_dir = "IM_train_data"  # folder to store all datasets

    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, n_datasets + 1):
        print(f"Generating dataset {i}...")

        # Use a different seed for each file to make them different
        dataset_seed = i  
        dataset = generate_dataset(n_samples=n_samples, seed=dataset_seed)

        filename = f"IM_train_data_{i}.csv"
        filepath = os.path.join(output_dir, filename)

        np.savetxt(
            filepath,
            dataset,
            delimiter=",",
        )

    print(f"All done! Generated {n_datasets} files in folder: {output_dir}")
