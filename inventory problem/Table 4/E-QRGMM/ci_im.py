import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy import stats
from sklearn.utils import resample
from tqdm import trange

from pyqreg import quantreg
import ray
import os
import time

import warnings 
warnings.filterwarnings("ignore")

# ============= Key features: use (mu, s+S, S-s, 1/(S-s)) as four features =============
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


@ray.remote
def boot_once(df, k):
    warnings.filterwarnings('ignore')

    # Training data columns: mu, sum_sS, gap, inv_gap, cost
    bootstrap_sample = df.sample(n=len(df), replace=True)
    bootstrap_sample = pd.DataFrame(
        bootstrap_sample,
        columns=['mu', 'sum_sS', 'gap', 'inv_gap', 'cost']
    )

    # Model: cost ~ mu + sum_sS + gap + inv_gap
    mod = quantreg('cost ~ mu + sum_sS + gap + inv_gap', bootstrap_sample)
    models = [fit_model(x, mod) for x in quantiles]
    models = pd.DataFrame(
        models,
        columns=['q', 'b0', 'b1', 'b2', 'b3', 'b4']  # b0=intercept, b1=mu, b2=sum_sS, b3=gap, b4=inv_gap
    )
    nmodels = models.to_numpy()

    # X: [1, mu, sum_sS, gap, inv_gap], dimension (n, d+1) = (n, 5)
    ones_column = np.ones((n, 1))
    X = np.hstack((ones_column,
                   bootstrap_sample[['mu', 'sum_sS', 'gap', 'inv_gap']].values))
    Y = bootstrap_sample['cost'].values

    Beta = nmodels[:, 1:]  # shape: (num_q, d+1) = (num_q, 5)

    delta = 1e-1
    ex = np.mean(X, axis=0)  # (d+1,)

    derivative_list = []
    for j in range(Beta.shape[0]):
        if (nmodels[j, 0] > (taulb - 1e-6)) and (nmodels[j, 0] < (tauub + 1e-6)):
            beta = Beta[j].reshape(Beta[j].shape[0], 1)  # (5,1)
            Lambda_sum = np.zeros((d+1, d+1))  # (5,5)
            for i in range(n):
                x = X[i].reshape(-1, 1)
                y = Y[i]
                indicator = np.abs(y - x.T @ beta)
                if indicator < delta:
                    Lambda_sum += x @ x.T
            Lambda = Lambda_sum / n
            grad, _, _, _ = lstsq(Lambda, ex, cond=1e-5)
            derivative_list.append(2 * delta * grad)
        else:
            derivative_list.append(np.zeros(d+1))
    derivative_list = np.array(derivative_list)

    u = np.random.rand(k)
    Tau = nmodels.T[:1, :][0]  # = models['q'].values

    beta_curve_cubic = []
    for i in range(d+1):  # 0..4
        beta_inter = cubic_hermite_interpolation(
            Tau,
            Beta.T[i],
            derivative_list.T[i],
            u,
            taulb,
            tauub
        )
        beta_curve_cubic.append(beta_inter)
    beta_curve_cubic = np.array(beta_curve_cubic).reshape((d+1, k))  # (5, k)

    # x0 = [1, mu0, sum0, gap0, inv_gap0]
    x0 = np.array([1, mu0, sum0, gap0, inv_gap0]).reshape(d+1, 1)  # (5,1)

    gen_Y = beta_curve_cubic.T @ x0  # (k,1)

    mean = np.mean(gen_Y)
    quantile = np.percentile(gen_Y, 80)
    prob = np.sum(gen_Y > quantile_gt) / len(gen_Y)

    return [mean, quantile, prob]


# ====== Grid & confidence band settings ======
m = 300  # number of quantile levels in QRGMM
taulb = 0.1
tauub = 0.9
quantiles = create_custom_grid(m, taulb, tauub, c=2)

# These ground-truth values should be adjusted according to the true (s0, S0, mu0)
mean_gt = 357.86
quantile_gt = 366.72
prob_gt = 0.20

mean_cover_flag = 0
quantile_cover_flag = 0
prob_cover_flag = 0

# Number of repetitions and bootstrap samples
runi = 100
num_samples = 100
k = 100000

m_length = []
q_length = []
p_length = []
time_list = []

cpu = os.cpu_count() // 2
ray.init(num_cpus=cpu)

with trange(runi, dynamic_ncols=False) as pbar:
    for iter in pbar:
        # ================= Training data =================
        csv_path = f'IM_train_data/IM_train_data_{iter+1}.csv'
        # Read columns as mu, sum_sS, gap, inv_gap, cost
        df = pd.read_csv(
            csv_path,
            header=None,
            names=['mu', 'sum_sS', 'gap', 'inv_gap', 'cost']
        )

        start_time = time.time()
        futures = [boot_once.remote(df, k) for _ in range(num_samples)]
        boot_list = ray.get(futures)
        end_time = time.time()

        boot_list_array = np.array(boot_list)
        mean_list = boot_list_array[:, 0]
        quantile_list = boot_list_array[:, 1]
        prob_list = boot_list_array[:, 2]

        mean_q5 = np.percentile(mean_list, 5)
        mean_q95 = np.percentile(mean_list, 95)
        if mean_q5 < mean_gt < mean_q95:
            mean_cover_flag += 1
        mean_coverage = mean_cover_flag / (iter + 1)

        quantile_q5 = np.percentile(quantile_list, 5)
        quantile_q95 = np.percentile(quantile_list, 95)
        if quantile_q5 < quantile_gt < quantile_q95:
            quantile_cover_flag += 1
        quantile_coverage = quantile_cover_flag / (iter + 1)

        prob_q5 = np.percentile(prob_list, 5)
        prob_q95 = np.percentile(prob_list, 95)
        if prob_q5 < prob_gt < prob_q95:
            prob_cover_flag += 1
        prob_coverage = prob_cover_flag / (iter + 1)

        m_length.append(mean_q95 - mean_q5)
        q_length.append(quantile_q95 - quantile_q5)
        p_length.append(prob_q95 - prob_q5)
        time_list.append(end_time - start_time)

        pbar.set_postfix({
            "\n m coverage": mean_coverage, "m lb": "{:.4f}".format(mean_q5), "m ub": "{:.4f}".format(mean_q95),
            "\n q coverage": quantile_coverage, "q lb": "{:.4f}".format(quantile_q5), "q ub": "{:.4f}".format(quantile_q95),
            "\n p coverage": prob_coverage, "p lb": "{:.4f}".format(prob_q5), "p ub": "{:.4f}".format(prob_q95)
        })

ray.shutdown()

print("mean length: ", np.mean(m_length), np.std(m_length) / np.sqrt(runi))
print("quantile length: ", np.mean(q_length), np.std(q_length) / np.sqrt(runi))
print("prob length: ", np.mean(p_length), np.std(p_length) / np.sqrt(runi))
print("time: ", np.mean(time_list) / num_samples, np.std(time_list) / np.sqrt(runi) / num_samples)
