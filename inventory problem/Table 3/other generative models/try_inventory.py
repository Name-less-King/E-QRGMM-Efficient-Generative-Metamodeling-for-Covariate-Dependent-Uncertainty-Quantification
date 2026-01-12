from utils import *  # get_args, train_all, generate, etc.
import os
import time
import numpy as np
import torch
from scipy import stats

DEVICE = "cpu"


class Dataset:
    def __init__(self, name=None, x=None, y=None):
        """
        Inventory Dataset:
        - If x and y are provided, use them directly;
        - Otherwise, load data from IM_train_data/IM_train_data_{name}.csv using name.
        Each row: [s, S, mu, cost], x = (s, S, mu), y = cost.
        """
        if x is not None and y is not None:
            self.x = np.asarray(x, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.float32).ravel()
        else:
            if name is None:
                raise ValueError("Either name or (x, y) must be provided.")
            self.x, self.y = self.read_data(name)

        # Ensure float32 numpy arrays
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32).ravel()

        # ====== Standardize X ======
        self.x_mean = self.x.mean(axis=0)
        self.x_std = self.x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (self.x - self.x_mean) / self.x_std

        # ====== Standardize Y ======
        self.y_mean = float(self.y.mean())
        self.y_std = float(self.y.std())
        if self.y_std == 0.0:
            self.y_std = 1.0
        y_norm = (self.y - self.y_mean) / self.y_std

        # The model is trained on standardized x/y
        self.x_train = torch.as_tensor(x_norm, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_norm, dtype=torch.float32).view(-1, 1)
        self.device = DEVICE  # Consistent with usage in utils.train_all

    def read_data(self, name: int):
        import pandas as pd

        csv_path = os.path.join("IM_train_data", f"IM_train_data_{name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None)
        x_train = df.iloc[:, 0:3].values  # s, S, mu
        y_train = df.iloc[:, 3].values    # cost
        return x_train, y_train


def modify_args(args, data_set=None, data_type=None):
    """
    Keep the same style as in flow_inventory.py / utils.py:
    create corresponding directories under models/ and results/.
    """
    if data_set is None and data_type is None:
        instance = str(args["data_type"]) + "_" + str(args["data_set"])
    else:
        instance = str(data_type) + "_" + str(data_set)
        args["data_set"] = data_set
        args["data_type"] = data_type

    args["instance"] = instance
    os.makedirs(f"models/{instance}", exist_ok=True)
    os.makedirs(f"results/{instance}", exist_ok=True)
    return args


if __name__ == "__main__":
    # -------- 1. Read ground-truth samples (costs_100000.csv) --------
    # File has a single column of 100000 costs, no header.
    true_costs = np.loadtxt("costs_100000.csv", delimiter=",")
    true_costs = np.asarray(true_costs).ravel()
    k = len(true_costs)  # Number of generated samples matches number of true samples

    # -------- 2. Fixed test point (s0, S0, mu0), consistent with costs_100000 generation --------
    s0 = 320
    S0 = 420
    mu0 = 330
    # Test point in original scale (later standardized using each dataset's mean/std)
    x_test_raw = np.array([s0, S0, mu0], dtype=np.float32)

    # -------- 3. Set args --------
    args = get_args("inventory")   # data_type = 'inventory'
    # Optionally adjust training hyperparameters (more iterations, smaller LR, etc.)
    # args['num_iteration'] = 5000
    # args['learning_rate'] = 1e-4
    args = modify_args(args)       # instance = 'inventory_None', etc.

    # -------- 4. Containers for statistics --------
    gan_time_list = []
    gan_ks_list = []
    gan_wd_list = []

    ddim_time_list = []
    ddim_ks_list = []
    ddim_wd_list = []

    rect_time_list = []
    rect_ks_list = []
    rect_wd_list = []

    # -------- 5. Number of runs and training data files --------
    runi = 100           # Corresponds to IM_train_data_1.csv ... IM_train_data_100.csv
    for i in range(runi):
        name = i + 1
        data = Dataset(name)
        data.device = DEVICE

        # For the current dataset, standardize x_test using its x_mean/x_std
        x_test_norm = (x_test_raw - data.x_mean) / data.x_std
        x_test = torch.as_tensor(x_test_norm, dtype=torch.float32).to(DEVICE).view(1, -1)

        # ====== GAN ======
        start_time = time.time()
        train_all(data, args, "gan")
        y_pred = generate(x_test, args, "gan", sample_num=k).cpu().numpy()
        # De-standardize back to the original cost scale
        y_pred = y_pred * data.y_std + data.y_mean
        end_time = time.time()

        y_pred_flat = y_pred.flatten()
        ks, _ = stats.ks_2samp(y_pred_flat, true_costs)
        wd = stats.wasserstein_distance(y_pred_flat, true_costs)

        gan_time_list.append(end_time - start_time)
        gan_ks_list.append(ks)
        gan_wd_list.append(wd)

        # ====== DDIM (using diffusion model, inf_step < time_step) ======
        start_time = time.time()
        train_all(data, args, "diffusion")
        y_pred = generate(x_test, args, "diffusion", sample_num=k).cpu().numpy()
        # De-standardize
        y_pred = y_pred * data.y_std + data.y_mean
        end_time = time.time()

        y_pred_flat = y_pred.flatten()
        ks, _ = stats.ks_2samp(y_pred_flat, true_costs)
        wd = stats.wasserstein_distance(y_pred_flat, true_costs)

        ddim_time_list.append(end_time - start_time)
        ddim_ks_list.append(ks)
        ddim_wd_list.append(wd)

        # ====== Rectified Flow ======
        start_time = time.time()
        train_all(data, args, "rectified")
        y_pred = generate(x_test, args, "rectified", sample_num=k).cpu().numpy()
        # De-standardize
        y_pred = y_pred * data.y_std + data.y_mean
        end_time = time.time()

        y_pred_flat = y_pred.flatten()
        ks, _ = stats.ks_2samp(y_pred_flat, true_costs)
        wd = stats.wasserstein_distance(y_pred_flat, true_costs)

        rect_time_list.append(end_time - start_time)
        rect_ks_list.append(ks)
        rect_wd_list.append(wd)

    # -------- 6. Print means and standard errors --------
    def mean_se(arr):
        arr = np.asarray(arr)
        return np.mean(arr), np.std(arr, ddof=1) / np.sqrt(len(arr))

    gan_time_mean, gan_time_se = mean_se(gan_time_list)
    gan_ks_mean, gan_ks_se = mean_se(gan_ks_list)
    gan_wd_mean, gan_wd_se = mean_se(gan_wd_list)

    ddim_time_mean, ddim_time_se = mean_se(ddim_time_list)
    ddim_ks_mean, ddim_ks_se = mean_se(ddim_ks_list)
    ddim_wd_mean, ddim_wd_se = mean_se(ddim_wd_list)

    rect_time_mean, rect_time_se = mean_se(rect_time_list)
    rect_ks_mean, rect_ks_se = mean_se(rect_ks_list)
    rect_wd_mean, rect_wd_se = mean_se(rect_wd_list)

    print("GAN time:", gan_time_mean, gan_time_se)
    print("GAN KS:", gan_ks_mean, gan_ks_se)
    print("GAN WD:", gan_wd_mean, gan_wd_se)

    print("DDIM time:", ddim_time_mean, ddim_time_se)
    print("DDIM KS:", ddim_ks_mean, ddim_ks_se)
    print("DDIM WD:", ddim_wd_mean, ddim_wd_se)

    print("Rectified Flow time:", rect_time_mean, rect_time_se)
    print("Rectified Flow KS:", rect_ks_mean, rect_ks_se)
    print("Rectified Flow WD:", rect_wd_mean, rect_wd_se)
