from utils import *
import time
from scipy import stats
import os
import numpy as np
import torch

DEVICE = "cpu"


class Dataset:
    def __init__(self, args):
        """
        Dataset class to generate test instances.
        param args: Dictionary containing various arguments and settings.
        """
        # Generate raw data (original scale)
        self.x, self.y = self.generate_data(args['data_type'], args['data_size'])

        # Convert to numpy float32 for convenient standardization
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

        # ===== Standardize X =====
        self.x_mean = self.x.mean(axis=0)
        self.x_std = self.x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (self.x - self.x_mean) / self.x_std

        # ===== Standardize Y =====
        self.y_mean = float(self.y.mean())
        self.y_std = float(self.y.std())
        if self.y_std == 0.0:
            self.y_std = 1.0

        y_norm = (self.y - self.y_mean) / self.y_std

        # Standardized x / y used for training
        self.x_train = torch.as_tensor(x_norm, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_norm, dtype=torch.float32).view(-1, 1)

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

x0_1 = 4.0
x0_2 = -1.0
x0_3 = 3.0
# x_test in the original scale, later standardized using each Dataset's x_mean/x_std
x_test_raw = np.array([x0_1, x0_2, x0_3], dtype=np.float32)

k = 100000

args = get_args('t')
args = modify_args(args)

gan_time_list = []
gan_ks_list = []
gan_wd_list = []

diffusion_time_list = []
diffusion_ks_list = []
diffusion_wd_list = []

rectified_time_list = []
rectified_ks_list = []
rectified_wd_list = []

runi = 100
for i in range(runi):
    data = Dataset(args)
    data.device = DEVICE

    # For the current dataset, standardize x_test using its x_mean/x_std
    x_test_norm = (x_test_raw - data.x_mean) / data.x_std
    x_test = torch.as_tensor(x_test_norm, dtype=torch.float32).to(DEVICE).view(1, -1)

    mean_gt = a0 + a1 * x0_1 + a2 * x0_2 + a3 * x0_3
    std_gt = r0 + r1 * x0_1 + r2 * x0_2 + r3 * x0_3

    # -------- GAN --------
    start_time = time.time()
    train_all(data, args, 'gan')
    y_pred = generate(x_test, args, 'gan', sample_num=k).cpu().numpy()
    # Transform back to the original scale
    y_pred = y_pred * data.y_std + data.y_mean
    end_time = time.time()
    y_pred_flat = y_pred.flatten()
    ks, _ = stats.kstest(
        y_pred_flat,
        't',
        args=(5, mean_gt, std_gt)
    )
    wd = stats.wasserstein_distance(
        y_pred_flat,
        stats.t.rvs(5, mean_gt, std_gt, k)
    )
    gan_time_list.append(end_time - start_time)
    gan_ks_list.append(ks)
    gan_wd_list.append(wd)

    # -------- Diffusion --------
    start_time = time.time()
    train_all(data, args, 'diffusion')
    y_pred = generate(x_test, args, 'diffusion', sample_num=k).cpu().numpy()
    y_pred = y_pred * data.y_std + data.y_mean
    end_time = time.time()
    y_pred_flat = y_pred.flatten()
    ks, _ = stats.kstest(
        y_pred_flat,
        't',
        args=(5, mean_gt, std_gt)
    )
    wd = stats.wasserstein_distance(
        y_pred_flat,
        stats.t.rvs(5, mean_gt, std_gt, k)
    )
    diffusion_time_list.append(end_time - start_time)
    diffusion_ks_list.append(ks)
    diffusion_wd_list.append(wd)

    # -------- Rectified Flow --------
    start_time = time.time()
    train_all(data, args, 'rectified')
    y_pred = generate(x_test, args, 'rectified', sample_num=k).cpu().numpy()
    y_pred = y_pred * data.y_std + data.y_mean
    end_time = time.time()
    y_pred_flat = y_pred.flatten()
    ks, _ = stats.kstest(
        y_pred_flat,
        't',
        args=(5, mean_gt, std_gt)
    )
    wd = stats.wasserstein_distance(
        y_pred_flat,
        stats.t.rvs(5, mean_gt, std_gt, k)
    )
    rectified_time_list.append(end_time - start_time)
    rectified_ks_list.append(ks)
    rectified_wd_list.append(wd)

print('gan time:', np.mean(gan_time_list), np.std(gan_time_list)/np.sqrt(runi))
print('gan ks:', np.mean(gan_ks_list), np.std(gan_ks_list)/np.sqrt(runi))
print('gan wd:', np.mean(gan_wd_list), np.std(gan_wd_list)/np.sqrt(runi))

print('diffusion time:', np.mean(diffusion_time_list), np.std(diffusion_time_list)/np.sqrt(runi))
print('diffusion ks:', np.mean(diffusion_ks_list), np.std(diffusion_ks_list)/np.sqrt(runi))
print('diffusion wd:', np.mean(diffusion_wd_list), np.std(diffusion_wd_list)/np.sqrt(runi))

print('rectified time:', np.mean(rectified_time_list), np.std(rectified_time_list)/np.sqrt(runi))
print('rectified ks:', np.mean(rectified_ks_list), np.std(rectified_ks_list)/np.sqrt(runi))
print('rectified wd:', np.mean(rectified_wd_list), np.std(rectified_wd_list)/np.sqrt(runi))
