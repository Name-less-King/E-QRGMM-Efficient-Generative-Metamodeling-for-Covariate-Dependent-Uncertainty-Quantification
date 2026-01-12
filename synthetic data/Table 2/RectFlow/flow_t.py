from utils import *
from scipy import stats
import os
from tqdm import trange
import numpy as np
import torch

DEVICE = "cpu"


class Dataset:
    def __init__(self, args, x=None, y=None):
        """
        Dataset class to generate test instances.
        param args: Dictionary containing various arguments and settings.
        """
        # ------- Raw data (unstandardized) -------
        if x is not None and y is not None:
            self.x = np.asarray(x, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.float32)
        else:
            self.x, self.y = self.generate_data(args['data_type'], args['data_size'])
            self.x = np.asarray(self.x, dtype=np.float32)
            self.y = np.asarray(self.y, dtype=np.float32)

        # ------- Standardize X -------
        self.x_mean = self.x.mean(axis=0)
        self.x_std = self.x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (self.x - self.x_mean) / self.x_std

        # ------- Standardize Y -------
        self.y_mean = float(self.y.mean())
        self.y_std = float(self.y.std())
        if self.y_std == 0.0:
            self.y_std = 1.0
        y_norm = (self.y - self.y_mean) / self.y_std

        # ------- Feed standardized x / y into the model -------
        self.x_train = torch.as_tensor(x_norm, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_norm, dtype=torch.float32).reshape(-1, 1)
        self.device = DEVICE  # Automatically set device attribute

    def bootstrap(self):
        """Bootstrap resampling; returns a new Dataset instance built from resampled data in the original scale."""
        idx = np.random.choice(len(self.x), size=len(self.x), replace=True)
        x_boot = self.x[idx]
        y_boot = self.y[idx]
        # Reconstruct a Dataset using the bootstrapped data in the original scale (standardization will be redone inside)
        return Dataset(args, x=x_boot, y=y_boot)

    def generate_data(self, data_type, data_size):
        if data_type == 'normal':
            u1 = np.random.rand(data_size)
            x1 = x1lb + (x1ub - x1lb) * u1
            u2 = np.random.rand(data_size)
            x2 = x2lb + (x2ub - x2lb) * u2
            u3 = np.random.rand(data_size)
            x3 = x3lb + (x3ub - x3lb) * u3

            g1 = a0 + a1 * x1 + a2 * x2 + a3 * x3
            g2 = r0 + r1 * x1 + r2 * x2 + r3 * x3
            F = np.zeros((data_size, 4))
            for i in np.arange(0, data_size):
                F[i, 0] = x1[i]
                F[i, 1] = x2[i]
                F[i, 2] = x3[i]
                F[i, 3] = np.random.normal(g1[i], g2[i])
            x_train = F[:, 0:3]
            y_train = F[:, 3]

        elif data_type == 'halfnormal':
            u1 = np.random.rand(data_size)
            x1 = x1lb + (x1ub - x1lb) * u1
            u2 = np.random.rand(data_size)
            x2 = x2lb + (x2ub - x2lb) * u2
            u3 = np.random.rand(data_size)
            x3 = x3lb + (x3ub - x3lb) * u3

            g1 = a0 + a1 * x1 + a2 * x2 + a3 * x3
            g2 = r0 + r1 * x1 + r2 * x2 + r3 * x3
            F = np.zeros((data_size, 4))
            for i in np.arange(0, data_size):
                F[i, 0] = x1[i]
                F[i, 1] = x2[i]
                F[i, 2] = x3[i]
                F[i, 3] = stats.halfnorm.rvs(loc=g1[i], scale=g2[i])
            x_train = F[:, 0:3]
            y_train = F[:, 3]

        elif data_type == 'uniform':
            u1 = np.random.rand(data_size)
            x1 = x1lb + (x1ub - x1lb) * u1
            u2 = np.random.rand(data_size)
            x2 = x2lb + (x2ub - x2lb) * u2
            u3 = np.random.rand(data_size)
            x3 = x3lb + (x3ub - x3lb) * u3

            g1 = a0 + a1 * x1 + a2 * x2 + a3 * x3

            F = np.zeros((data_size, 4))
            for i in np.arange(0, data_size):
                F[i, 0] = x1[i]
                F[i, 1] = x2[i]
                F[i, 2] = x3[i]
                F[i, 3] = stats.uniform.rvs(g1[i], w)  # loc=g1[i], scale=w
            x_train = F[:, 0:3]
            y_train = F[:, 3]

        elif data_type == 't':
            u1 = np.random.rand(data_size)
            x1 = x1lb + (x1ub - x1lb) * u1
            u2 = np.random.rand(data_size)
            x2 = x2lb + (x2ub - x2lb) * u2
            u3 = np.random.rand(data_size)
            x3 = x3lb + (x3ub - x3lb) * u3

            g1 = a0 + a1 * x1 + a2 * x2 + a3 * x3
            g2 = r0 + r1 * x1 + r2 * x2 + r3 * x3
            F = np.zeros((data_size, 4))
            for i in np.arange(0, data_size):
                F[i, 0] = x1[i]
                F[i, 1] = x2[i]
                F[i, 2] = x3[i]
                F[i, 3] = stats.t.rvs(5, g1[i], g2[i])
            x_train = F[:, 0:3]
            y_train = F[:, 3]

        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        return x_train, y_train


def modify_args(args, data_set=None, data_type=None):
    if data_set is None and data_type is None:
        instance = str(args['data_type']) + '_' + str(args['data_set'])
    else:
        instance = str(data_type) + '_' + str(data_set)
        args['data_set'] = data_set
        args['data_type'] = data_type
    args['instance'] = instance
    if not os.path.exists(f'models/{instance}'):
        os.makedirs(f'models/{instance}')
    if not os.path.exists(f'results/{instance}'):
        os.makedirs(f'results/{instance}')
    return args


def boot_once(x, y, k, args):
    """
    Perform one round of bootstrap + training + prediction on (x, y):
    - bootstrap is done in the original scale;
    - data_boot standardizes X and Y internally;
    - x_test is standardized using data_boot.x_mean/x_std;
    - generated y is transformed back to the original scale using data_boot.y_mean/y_std.
    """
    # First construct a Dataset using (x, y), then bootstrap once based on it
    tmp = Dataset(args, x=x, y=y)
    data_boot = tmp.bootstrap()  # data_boot has its own x_mean/x_std and y_mean/y_std

    # Standardize x_test_raw using the bootstrap dataset's X mean/std
    x_test_norm = (x_test_raw - data_boot.x_mean) / data_boot.x_std
    x_test_tensor = torch.as_tensor(x_test_norm, dtype=torch.float32).to(DEVICE).view(1, -1)

    # Train rectified flow
    train_all(data_boot, args, 'rectified')

    # Generate in standardized x space, then transform y back to the original scale
    y_pred = generate(x_test_tensor, args, 'rectified', sample_num=k).cpu().numpy()
    y_pred = y_pred * data_boot.y_std + data_boot.y_mean
    y_pred_flat = y_pred.flatten()

    mean = np.mean(y_pred_flat)
    quantile = np.percentile(y_pred_flat, 80)
    prob = np.sum(y_pred_flat > quantile_gt) / len(y_pred_flat)

    return [mean, quantile, prob]


# ----------------- Global parameters -----------------

x1lb = 0
x1ub = 10
x2lb = -5
x2ub = 5
x3lb = 0
x3ub = 5
# range of covariates

a0 = 5
a1 = 1
a2 = 2
a3 = 0.5
a_array = [a0, a1, a2, a3]

r0 = 1
r1 = 0.1
r2 = 0.2
r3 = 0.05
r_array = [r0, r1, r2, r3]
w = 10

x0_1 = 4.0
x0_2 = -1.0
x0_3 = 3.0
# x_test in the original scale; the concrete standardization is done inside each data_boot
x_test_raw = np.array([x0_1, x0_2, x0_3], dtype=np.float32)

args = get_args('t')
args = modify_args(args)

dft = 5
loc = a0 + a1 * x0_1 + a2 * x0_2 + a3 * x0_3
scale = r0 + r1 * x0_1 + r2 * x0_2 + r3 * x0_3
mean_gt = loc
mean_cover_flag = 0

quantile_gt = stats.t.ppf(0.8, dft, loc, scale)
quantile_cover_flag = 0

prob_gt = 0.20
prob_cover_flag = 0

m_length = []
q_length = []
p_length = []

runi = 100
num_samples = 100
k = 100000

with trange(runi, dynamic_ncols=False) as pbar:
    for iter in pbar:
        data = Dataset(args)

        boot_list = []
        for _ in range(num_samples):
            result = boot_once(data.x, data.y, k, args)
            boot_list.append(result)

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

        pbar.set_postfix({
            "\n m coverage": mean_coverage, "m lb": "{:.4f}".format(mean_q5), "m ub": "{:.4f}".format(mean_q95),
            "\n q coverage": quantile_coverage, "q lb": "{:.4f}".format(quantile_q5), "q ub": "{:.4f}".format(quantile_q95),
            "\n p coverage": prob_coverage, "p lb": "{:.4f}".format(prob_q5), "p ub": "{:.4f}".format(prob_q95)
        })

print("mean length: ", np.mean(m_length), np.std(m_length)/np.sqrt(runi))
print("quantile length: ", np.mean(q_length), np.std(q_length)/np.sqrt(runi))
print("prob length: ", np.mean(p_length), np.std(p_length)/np.sqrt(runi))
