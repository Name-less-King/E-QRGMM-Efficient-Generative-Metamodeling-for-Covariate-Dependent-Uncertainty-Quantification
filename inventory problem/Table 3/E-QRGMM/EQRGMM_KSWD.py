import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy import stats
from scipy.stats import wasserstein_distance
from tqdm import trange
# Due to model misspecification, we can apply the preprocessing step of Portnoy and Koenker (1997) to speed up computation.
from pyqreg import quantreg

import warnings
warnings.filterwarnings("ignore")

# ============= Key dimensions: use four features (mu, s+S, S-s, 1/(S-s)) =============
d = 4         # mu, sum_sS = s+S, gap = S-s, inv_gap = 1/(S-s)
n = 10000     # each training dataset has 10,000 rows

# ====== Test point (still using (s0, S0, mu0); features will be converted to mu, sum_sS, gap, inv_gap below) ======
s0  = 320
S0  = 420
mu0 = 330

sum0 = s0 + S0
gap0 = S0 - s0
inv_gap0 = 1.0 / gap0


def fit_model(q, mod):  # quantile regression
    res = mod.fit(q=q)
    # Parameter order: intercept + mu + sum_sS + gap + inv_gap
    return [
        q,
        res.params['Intercept'],
        res.params['mu'],
        res.params['sum_sS'],
        res.params['gap'],
        res.params['inv_gap'],
    ]


# Function for cubic Hermite interpolation
def cubic_hermite_interpolation(x_data, y_data, dy_data, x_fine, lb, ub):
    n_seg = len(x_data)
    y_fine = np.zeros_like(x_fine)
    x_fine_clipped = np.clip(x_fine, lb, ub)
    for i in range(n_seg - 1):
        x0, x1 = x_data[i], x_data[i+1]
        y0, y1 = y_data[i], y_data[i+1]
        dy0, dy1 = dy_data[i], dy_data[i+1]
        h = x1 - x0 + 1e-8
        mask = (x_fine_clipped > x0) & (x_fine_clipped < x1)
        t = (x_fine_clipped[mask] - x0) / h
        h00 = (2 * t**3 - 3 * t**2 + 1)
        h10 = (t**3 - 2 * t**2 + t)
        h01 = (-2 * t**3 + 3 * t**2)
        h11 = (t**3 - t**2)
        y_fine[mask] = h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1
    out_of_bounds_mask = (x_fine < lb) | (x_fine > ub)
    y_fine[out_of_bounds_mask] = np.interp(x_fine[out_of_bounds_mask], x_data, y_data)
    return y_fine


def create_custom_grid(m, taulb, tauub, c=2):
    base_points = np.linspace(1/m, 1 - 1/m, m - 1)
    mask = (base_points > taulb) & (base_points < tauub)
    spacing = (1/m) ** 0.4
    num_points = int((tauub - taulb) / spacing) * c
    if num_points > 0:
        new_inner_points = np.linspace(taulb, tauub, num_points + 2)[1:-1]
    else:
        new_inner_points = np.array([])
    outer_points = base_points[~mask]
    combined_points = np.concatenate([outer_points, new_inner_points, [taulb, tauub]])
    combined_points = np.unique(np.round(combined_points, decimals=6))
    return combined_points


# ====== Grid & QRGMM settings ======
m = 300  # number of quantile levels in QRGMM
taulb = 0.1
tauub = 0.9
quantiles = create_custom_grid(m, taulb, tauub, c=2)

k = 100000          # each time we generate k samples from the estimated distribution
runi = 100          # number of repetitions (using 100 training sets IM_train_data_i.csv)


def qrgmm_generate_at_x0(df):
    """
    Given one training dataset df (columns: mu, sum_sS, gap, inv_gap, cost),
    use QRGMM to estimate the conditional distribution, then at x0 corresponding
    to (s0, S0, mu0) generate k samples and return a 1D array y_pred (length k).
    """
    # Training data columns: mu, sum_sS, gap, inv_gap, cost
    # No bootstrap here; we directly use the full df
    bootstrap_sample = df

    # Model: cost ~ mu + sum_sS + gap + inv_gap
    mod = quantreg('cost ~ mu + sum_sS + gap + inv_gap', bootstrap_sample)
    models = [fit_model(x, mod) for x in quantiles]
    models = pd.DataFrame(
        models,
        columns=['q', 'b0', 'b1', 'b2', 'b3', 'b4']  # b0=intercept, b1=mu, b2=sum_sS, b3=gap, b4=inv_gap
    )
    nmodels = models.to_numpy()

    # X: [1, mu, sum_sS, gap, inv_gap], shape (N, d+1) = (N, 5)
    N = len(bootstrap_sample)
    ones_column = np.ones((N, 1))
    X = np.hstack((ones_column,
                   bootstrap_sample[['mu', 'sum_sS', 'gap', 'inv_gap']].values))
    Y = bootstrap_sample['cost'].values

    Beta = nmodels[:, 1:]  # shape: (num_q, d+1) = (num_q, 5)

    delta = 1e-1
    ex = np.mean(X, axis=0)  # (d+1,)

    derivative_list = []
    for j in range(Beta.shape[0]):
        q_val = nmodels[j, 0]
        if (q_val > (taulb - 1e-6)) and (q_val < (tauub + 1e-6)):
            beta = Beta[j].reshape(Beta[j].shape[0], 1)  # (5,1)
            Lambda_sum = np.zeros((d+1, d+1))  # (5,5)
            for i in range(N):
                x = X[i].reshape(-1, 1)
                y = Y[i]
                indicator = np.abs(y - x.T @ beta)
                if indicator < delta:
                    Lambda_sum += x @ x.T
            Lambda = Lambda_sum / N
            grad, _, _, _ = lstsq(Lambda, ex, cond=1e-5)
            derivative_list.append(2 * delta * grad)
        else:
            derivative_list.append(np.zeros(d+1))
    derivative_list = np.array(derivative_list)

    # Draw k random u in (0,1), representing quantile levels
    u = np.random.rand(k)
    Tau = nmodels[:, 0]  # = models['q'].values

    beta_curve_cubic = []
    for i in range(d+1):  # 0..4
        beta_inter = cubic_hermite_interpolation(
            Tau,
            Beta[:, i],
            derivative_list[:, i],
            u,
            taulb,
            tauub
        )
        beta_curve_cubic.append(beta_inter)
    beta_curve_cubic = np.array(beta_curve_cubic).reshape((d+1, k))  # (5, k)

    # x0 = [1, mu0, sum0, gap0, inv_gap0]
    x0 = np.array([1, mu0, sum0, gap0, inv_gap0]).reshape(d+1, 1)  # (5,1)

    gen_Y = beta_curve_cubic.T @ x0  # (k,1)

    return gen_Y.ravel()


if __name__ == "__main__":
    # ------- True distribution samples: from costs_100000.csv -------
    # This file is 100,000 costs simulated with fixed (s0, S0, mu0)
    true_costs = np.loadtxt("costs_100000.csv", delimiter=",")

    ks_list = []
    wd_list = []

    # Similar to E-QRGMM_normal: repeat runi times, each time using one IM_train_data_i.csv
    with trange(runi, dynamic_ncols=False) as pbar:
        for iter in pbar:
            csv_path = f'IM_train_data/IM_train_data_{iter+1}.csv'
            # File without header: read columns as mu, sum_sS, gap, inv_gap, cost
            df = pd.read_csv(
                csv_path,
                header=None,
                names=['mu', 'sum_sS', 'gap', 'inv_gap', 'cost']
            )

            # Use QRGMM to generate k predictive samples at x0
            y_pred = qrgmm_generate_at_x0(df)

            # ---- Use true_costs as the "true distribution" to compute KS and WD ----
            # KS: two-sample version, y_pred vs true_costs
            ks_stat, _ = stats.ks_2samp(y_pred, true_costs)

            # WD: one-dimensional Wasserstein distance
            wd_stat = wasserstein_distance(y_pred, true_costs)

            ks_list.append(ks_stat)
            wd_list.append(wd_stat)

            pbar.set_postfix({
                "KS": f"{ks_stat:.4f}",
                "WD": f"{wd_stat:.4f}",
            })

    # Same as in E-QRGMM_normal: report mean and standard error
    ks_mean = np.mean(ks_list)
    ks_se = np.std(ks_list, ddof=1) / np.sqrt(runi)

    wd_mean = np.mean(wd_list)
    wd_se = np.std(wd_list, ddof=1) / np.sqrt(runi)
    
    # To keep CPU utilization consistent/comparable, time computation is handled in Table 2 / E-QRGMM.
    print("KS:", ks_mean, ks_se)
    print("WD:", wd_mean, wd_se)
