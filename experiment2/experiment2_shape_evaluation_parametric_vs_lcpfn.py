#!/usr/bin/env python
# coding: utf-8
import argparse

# # Evaluation Notebook - LC-PFN vs. Parametric Models
# ### Comparative experiment: Flat vs non-Flat, Monotone & Convex vs non-Monotone & Convex, Peaking vs non-Peaking, Dipping vs non-Dipping

# In[33]:


import h5py
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from math import pi
import matplotlib.patheffects as path_effects
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import lcpfn
from lcpfn import bar_distribution, encoders, train, utils

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMBA_NUM_THREADS'] = '16'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

### hyperparameter
OPENML_ID = {0: '3', 1: '6', 2: '11', 3: '12', 4: '13', 5: '14', 6: '15', 7: '16', 8: '18', 9: '21', 10: '22', 11: '23', 12: '24', 13: '26', 14: '28', 15: '29', 16: '30', 17: '31', 18: '32', 19: '36', 20: '37', 21: '38', 22: '44', 23: '46', 24: '50', 25: '54', 26: '55', 27: '57', 28: '60', 29: '61', 30: '151', 31: '179', 32: '180', 33: '181', 34: '182', 35: '184', 36: '185', 37: '188', 38: '201', 39: '273', 40: '293', 41: '299', 42: '300', 43: '307', 44: '336', 45: '346', 46: '351', 47: '354', 48: '357', 49: '380', 50: '389', 51: '390', 52: '391', 53: '392', 54: '393', 55: '395', 56: '396', 57: '398', 58: '399', 59: '401', 60: '446', 61: '458', 62: '469', 63: '554', 64: '679', 65: '715', 66: '718', 67: '720', 68: '722', 69: '723', 70: '727', 71: '728', 72: '734', 73: '735', 74: '737', 75: '740', 76: '741', 77: '743', 78: '751', 79: '752', 80: '761', 81: '772', 82: '797', 83: '799', 84: '803', 85: '806', 86: '807', 87: '813', 88: '816', 89: '819', 90: '821', 91: '822', 92: '823', 93: '833', 94: '837', 95: '843', 96: '845', 97: '846', 98: '847', 99: '849', 100: '866', 101: '871', 102: '881', 103: '897', 104: '901', 105: '903', 106: '904', 107: '910', 108: '912', 109: '913', 110: '914', 111: '917', 112: '923', 113: '930', 114: '934', 115: '953', 116: '958', 117: '959', 118: '962', 119: '966', 120: '971', 121: '976', 122: '977', 123: '978', 124: '979', 125: '980', 126: '991', 127: '993', 128: '995', 129: '1000', 130: '1002', 131: '1018', 132: '1019', 133: '1020', 134: '1021', 135: '1036', 136: '1040', 137: '1041', 138: '1042', 139: '1049', 140: '1050', 141: '1053', 142: '1056', 143: '1063', 144: '1067', 145: '1068', 146: '1069', 147: '1083', 148: '1084', 149: '1085', 150: '1086', 151: '1087', 152: '1088', 153: '1116', 154: '1119', 155: '1120', 156: '1128', 157: '1130', 158: '1134', 159: '1138', 160: '1139', 161: '1142', 162: '1146', 163: '1161', 164: '1166', 165: '1216', 166: '1233', 167: '1235', 168: '1236', 169: '1441', 170: '1448', 171: '1450', 172: '1457', 173: '1461', 174: '1462', 175: '1464', 176: '1465', 177: '1468', 178: '1475', 179: '1477', 180: '1478', 181: '1479', 182: '1480', 183: '1483', 184: '1485', 185: '1486', 186: '1487', 187: '1488', 188: '1489', 189: '1494', 190: '1497', 191: '1499', 192: '1501', 193: '1503', 194: '1509', 195: '1510', 196: '1515', 197: '1566', 198: '1567', 199: '1575', 200: '1590', 201: '1592', 202: '1597', 203: '4134', 204: '4135', 205: '4137', 206: '4534', 207: '4538', 208: '4541', 209: '6332', 210: '23381', 211: '23512', 212: '23517', 213: '40498', 214: '40499', 215: '40664', 216: '40668', 217: '40670', 218: '40672', 219: '40677', 220: '40685', 221: '40687', 222: '40701', 223: '40713', 224: '40900', 225: '40910', 226: '40923', 227: '40927', 228: '40966', 229: '40971', 230: '40975', 231: '40978', 232: '40979', 233: '40981', 234: '40982', 235: '40983', 236: '40984', 237: '40994', 238: '40996', 239: '41027', 240: '41142', 241: '41143', 242: '41144', 243: '41145', 244: '41146', 245: '41150', 246: '41156', 247: '41157', 248: '41158', 249: '41159', 250: '41161', 251: '41163', 252: '41164', 253: '41165', 254: '41166', 255: '41167', 256: '41168', 257: '41169', 258: '41228', 259: '41972', 260: '42734', 261: '42742', 262: '42769', 263: '42809', 264: '42810'}
LEARNER_ZOO = {0: 'SVC_linear', 1: 'SVC_poly', 2: 'SVC_rbf', 3: 'SVC_sigmoid', 4: 'Decision Tree', 5: 'ExtraTree', 6: 'LogisticRegression', 7: 'PassiveAggressive', 8: 'Perceptron', 9: 'RidgeClassifier', 10: 'SGDClassifier', 11: 'MLP', 12: 'LDA', 13: 'QDA', 14: 'BernoulliNB', 15: 'MultinomialNB', 16: 'ComplementNB', 17: 'GaussianNB', 18: 'KNN', 19: 'NearestCentroid', 20: 'ens.ExtraTrees', 21: 'ens.RandomForest', 22: 'ens.GradientBoosting', 23: 'DummyClassifier'}
ANCHOR_SIZE = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)

### load data: validation accuracy
lc_data = h5py.File(Path.cwd() / 'LCDB11_ACC_265_noFS_raw_compress.hdf5', 'r')['accuracy'][...][:,:,:,:,:,1]

mean_valid_lc_nofs =np.nanmean(lc_data, axis=(2, 3))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

n_datasets = len(OPENML_ID)
n_learners = len(LEARNER_ZOO) - 1  # Exclude DummyClassifier

all_pairs = []
for dataset_idx in range(n_datasets):
    for learner_idx in range(n_learners):
        all_pairs.append((dataset_idx, learner_idx))

all_pairs = np.array(all_pairs)
print(f"Total combinations: {len(all_pairs)}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

train_pair_indices, test_pair_indices = train_test_split(
    np.arange(len(all_pairs)),
    test_size=0.2,
    random_state=SEED,
    stratify=all_pairs[:, 1]
)

train_pairs = all_pairs[train_pair_indices]
test_pairs = all_pairs[test_pair_indices]

dipping_pairs = np.loadtxt("experiment2/dipping_pairs.csv", dtype=int, delimiter=",")
flat_pairs = np.loadtxt("experiment2/flat_pairs.csv", dtype=int, delimiter=",")
mono_conv_pairs = np.loadtxt("experiment2/mono_conv_pairs.csv", dtype=int, delimiter=",")
peaking_pairs = np.loadtxt("experiment2/peaking_pairs.csv", dtype=int, delimiter=",")

print(f"Train pairs: {len(train_pairs)}")
print(f"Test pairs: {len(test_pairs)}")

def extract_all_curves_for_pairs(pairs, lc_data):
    curves = []
    dataset_indices = []
    learner_indices = []

    for dataset_idx, learner_idx in pairs:
        dataset_learner_curves = lc_data[dataset_idx, learner_idx, :, :, :]

        for seed_idx in range(dataset_learner_curves.shape[0]):
            for split_idx in range(dataset_learner_curves.shape[1]):
                curve = dataset_learner_curves[seed_idx, split_idx, :]
                curve_length = np.count_nonzero(~np.isnan(curve))

                if curve_length > 0:
                    curves.append(curve)
                    dataset_indices.append(dataset_idx)
                    learner_indices.append(learner_idx)

    return curves, dataset_indices, learner_indices

train_curves, train_dataset_indices, train_learner_indices = extract_all_curves_for_pairs(train_pairs, lc_data)
test_curves, test_dataset_indices, test_learner_indices = extract_all_curves_for_pairs(test_pairs, lc_data)

print(f"Train curves: {len(train_curves)}")
print(f"Test curves: {len(test_curves)}")

test_dipping_curves, test_dipping_datasets, test_dipping_learners = extract_all_curves_for_pairs(dipping_pairs[:, [1, 0]], lc_data)
test_flat_curves, test_flat_datasets, test_flat_learners = extract_all_curves_for_pairs(flat_pairs[:, [1, 0]], lc_data)
test_mono_conv_curves, test_mono_conv_datasets, test_mono_conv_learners = extract_all_curves_for_pairs(mono_conv_pairs[:, [1, 0]], lc_data)
test_peaking_curves, test_peaking_datasets, test_peaking_learners = extract_all_curves_for_pairs(peaking_pairs[:, [1, 0]], lc_data)

print(f"Test dipping curves: {len(test_dipping_curves)}")
print(f"Test flat curves: {len(test_flat_curves)}")
print(f"Test mono_conv curves: {len(test_mono_conv_curves)}")
print(f"Test peaking curves: {len(test_peaking_curves)}")

model_name = 'lcpfn_model_exp2_140_512_12_1000_0.0001_100_1000.pth'
model = torch.load(f'trained_models/exp2_140_512_12_1000_0.0001_100_1000/{model_name}', weights_only=False)
model.eval()

# ## Extrapolating curves

# In[37]:


def extrapolate_lcpfn(curve, anchor_sizes, model, min_points=10, qs=[0.05, 0.5, 0.95],
                     random_cutoff=True, fixed_cutoff_idx=None, cutoff_percentage=None):
    """
    Extrapolate a learning curve using the LC-PFN model.

    Args:
        curve: Original learning curve (1D array)
        anchor_sizes: Training sizes corresponding to the curve points
        model: Trained LC-PFN model
        min_points: Minimum number of points to use for fitting
        qs: Quantiles for prediction intervals
        random_cutoff: If True, uses a random cutoff point
        fixed_cutoff_idx: If provided, uses this specific cutoff index instead of generating a random one
        cutoff_percentage: If provided, uses this percentage of the curve for training (e.g., 0.7 for 70%)

    Returns:
        x_train: Training x values (indices)
        y_train_norm: Normalised training y values
        x_test: Test x values (indices)
        y_test: Test y values (ground truth)
        pred_mean: Predicted mean values
        pred_lower: Lower prediction interval
        pred_upper: Upper prediction interval
    """

    valid_mask = np.isfinite(curve)
    valid_curve = curve[valid_mask]
    valid_anchors = anchor_sizes[:len(curve)][valid_mask]

    if len(valid_curve) <= min_points:
        return None, None, None, None, None, None, None

    y = torch.from_numpy(valid_curve).float().unsqueeze(-1)
    x = torch.arange(1, y.shape[0] + 1).unsqueeze(-1).float()

    if cutoff_percentage is not None:
        cutoff_idx = max(min_points, int(cutoff_percentage * len(valid_curve)))
        cutoff_idx = min(cutoff_idx, len(valid_curve) - 1)  # Ensure we have at least 1 test point
    elif fixed_cutoff_idx is not None:
        cutoff_idx = fixed_cutoff_idx
    elif random_cutoff:
        cutoff_idx = random.randint(min_points, len(valid_curve) - 1)
    else:
        cutoff_idx = len(valid_curve) - 1

    x_train = x[:cutoff_idx]
    y_train = y[:cutoff_idx]
    x_test = x[cutoff_idx:]
    y_test = y[cutoff_idx:]

    normalizer = lcpfn.utils.identity_normalizer()
    y_train_norm = normalizer[0](y_train)

    single_eval_pos = x_train.shape[0]
    x_combined = torch.cat([x_train, x_test], dim=0).unsqueeze(1)
    y_input = y_train.unsqueeze(1)

    logits = model((x_combined, y_input), single_eval_pos=single_eval_pos)

    predictions = normalizer[1](
        torch.cat([model.criterion.icdf(logits, q) for q in qs], dim=1)
    )

    pred_mean = predictions[:, 1].detach().cpu().numpy()
    pred_lower = predictions[:, 0].detach().cpu().numpy()
    pred_upper = predictions[:, 2].detach().cpu().numpy()

    x_test_np = x_test.detach().cpu().numpy().flatten()

    return (
        x_train.detach().cpu().numpy().flatten(),
        y_train_norm.detach().cpu().numpy().flatten(),
        x_test_np,
        y_test.detach().cpu().numpy().flatten(),
        pred_mean,
        pred_lower,
        pred_upper
    )


# In[38]:


def mmf4(n, a, b, c, d):
    return (a * b + c * n**d) / (b + n**d)

def wbl4(n, a, b, c, d):
    return c - b * np.exp(-a * n**d)

def pow4(n, a, b, c, d):
    return a - b * (d + n)**(-c)

def mmf4_jacobian(n, a, b, c, d):
    """
    Jacobian for MMF4: f(n) = (a * b + c * n^d) / (b + n^d)
    Returns partial derivatives w.r.t. [a, b, c, d]
    """
    n_d = n**d
    denominator = b + n_d

    # ∂f/∂a = b / (b + n^d)
    da = b / denominator

    # ∂f/∂b = (a * (b + n^d) - (a * b + c * n^d)) / (b + n^d)^2
    #        = (a * n^d - c * n^d) / (b + n^d)^2
    db = (a - c) * n_d / (denominator**2)

    # ∂f/∂c = n^d / (b + n^d)
    dc = n_d / denominator

    # ∂f/∂d = (c * n^d * ln(n) * (b + n^d) - (a * b + c * n^d) * n^d * ln(n)) / (b + n^d)^2
    #        = n^d * ln(n) * (c * b - a * b) / (b + n^d)^2
    log_n = np.log(n)
    log_n = np.where(n > 0, log_n, 0)  # Handle n=0 case
    dd = n_d * log_n * b * (c - a) / (denominator**2)

    return np.column_stack([da, db, dc, dd])

def wbl4_jacobian(n, a, b, c, d):
    """
    Jacobian for WBL4: f(n) = c - b * exp(-a * n^d)
    Returns partial derivatives w.r.t. [a, b, c, d]
    """
    n_d = n**d
    exp_term = np.exp(-a * n_d)

    # ∂f/∂a = b * n^d * exp(-a * n^d)
    da = b * n_d * exp_term

    # ∂f/∂b = -exp(-a * n^d)
    db = -exp_term

    # ∂f/∂c = 1
    dc = np.ones_like(n)

    # ∂f/∂d = b * a * n^d * ln(n) * exp(-a * n^d)
    log_n = np.log(n)
    log_n = np.where(n > 0, log_n, 0)
    dd = b * a * n_d * log_n * exp_term

    return np.column_stack([da, db, dc, dd])

def pow4_jacobian(n, a, b, c, d):
    """
    Jacobian for POW4: f(n) = a - b * (d + n)^(-c)
    Returns partial derivatives w.r.t. [a, b, c, d]
    """
    d_plus_n = d + n
    power_term = d_plus_n**(-c)

    # ∂f/∂a = 1
    da = np.ones_like(n)

    # ∂f/∂b = -(d + n)^(-c)
    db = -power_term

    # ∂f/∂c = b * (d + n)^(-c) * ln(d + n)
    log_term = np.log(d_plus_n)
    log_term = np.where(d_plus_n > 0, log_term, 0)
    dc = b * power_term * log_term

    # ∂f/∂d = b * c * (d + n)^(-c-1)
    dd = b * c * (d_plus_n**(-c-1))

    return np.column_stack([da, db, dc, dd])

# In[39]:

total_param_fits = 0
unsuccessful_param_fits = 0

def extrapolate_parametric(curve, anchor_sizes, model="MMF4", min_points=10,
                          random_cutoff=True, fixed_cutoff_idx=None, cutoff_percentage=None):
    """
    Fit and extrapolate a learning curve with an optional random OR pre-defined cutoff point, via a parametric model.

    Initial guesses are independent of the training data to avoid data leakage.

    Args:
        curve: Original/complete learning curve
        anchor_sizes: Training sizes corresponding to the curve points
        model: "MMF4", "WBL4", or "POW4"
        min_points: Minimum number of points to use for fitting
        random_cutoff: If True, uses a random cutoff point
        fixed_cutoff_idx: If provided, uses this specific cutoff index instead of generating a random one
        cutoff_percentage: If provided, uses this percentage of the curve for training (e.g., 0.7 for 70%)

    Returns:
        x_train: Training x values
        y_train: Training y values
        x_test: Test x values
        y_test: Test y values (ground truth)
        y_pred: Predicted values for full curve
    """
    # Remove NaN values
    valid_mask = np.isfinite(curve)
    valid_curve = curve[valid_mask]
    valid_anchors = anchor_sizes[:len(valid_curve)][valid_mask]

    if len(valid_curve) <= min_points:
        return None, None, None, None, None

    if cutoff_percentage is not None:
        cutoff_idx = max(min_points, int(cutoff_percentage * len(valid_curve)))
        cutoff_idx = min(cutoff_idx, len(valid_curve) - 1)
    elif fixed_cutoff_idx is not None:
        cutoff_idx = fixed_cutoff_idx
    elif random_cutoff:
        cutoff_idx = random.randint(min_points, len(valid_curve) - 1)
    else:
        cutoff_idx = len(valid_curve) - 1

    y_train = valid_curve[:cutoff_idx]
    y_test = valid_curve[cutoff_idx:]
    x_train = valid_anchors[:cutoff_idx]
    x_test = valid_anchors[cutoff_idx:]

    if model == "MMF4":
        model_func = mmf4
    elif model == "WBL4":
        model_func = wbl4
    else:
        model_func = pow4

    if model == "MMF4":
        # MMF4: (a * b + c * n^d) / (b + n^d)
        # Typical learning curves: start low, asymptote high
        p0 = [0.9, 1000.0, 0.1, 1.0]  # a, b, c, d
        bounds = ([0.01, 1e-6, 0.0, 0.01], [1.0, np.inf, 1.0, 10.0])
        model_func = mmf4
        jac_func = mmf4_jacobian
    elif model == "WBL4":
        # WBL4: c - b * exp(-a * n^d)
        # Typical: exponential approach to asymptote
        p0 = [0.001, 0.8, 0.9, 1.0]  # a, b, c, d
        bounds = ([1e-10, 0.01, 0.01, 0.01], [1.0, 2.0, 1.0, 5.0])
        model_func = wbl4
        jac_func = wbl4_jacobian
    else:
        # POW4: a - b * (d + n)^(-c)
        # Power law decay from initial value
        p0 = [0.9, 0.8, 1.0, 100.0]  # a, b, c, d
        bounds = ([0.01, 0.01, 0.001, 1.0], [1.0, 2.0, 5.0, 10000.0])
        model_func = pow4
        jac_func = pow4_jacobian

    fit_successful = False
    global total_param_fits
    ++total_param_fits

    # Strategy 1: Try with fixed initial guesses
    try:
        popt, _ = curve_fit(model_func, x_train, y_train, p0=p0, bounds=bounds, jac=jac_func, maxfev=25000)
        fit_successful = True
    except (RuntimeError, ValueError, TypeError):
        pass

    # Strategy 2: If no success, use unbounded fitting
    if not fit_successful:
        try:
            popt, _ = curve_fit(model_func, x_train, y_train, p0=p0, jac=jac_func, maxfev=25000)
            fit_successful = True
        except (RuntimeError, ValueError, TypeError):
            pass

    if not fit_successful:
        # print(f"Warning: Curve fitting failed for {model}, using initial parameter guess")
        popt = p0
        global unsuccessful_param_fits
        ++unsuccessful_param_fits

    x_full = valid_anchors
    y_pred = model_func(x_full, *popt)

    y_pred = np.clip(y_pred, 0.0, 1.0)

    return x_train, y_train, x_test, y_test, y_pred


# In[40]:


def compare_extrapolations(curve_idx, data, anchor_sizes, lcpfn_model, min_points=10,
                          random_cutoff=True, fixed_cutoff_idx=None, cutoff_percentage=None):
    """
    Compare extrapolations of a learning curve using LC-PFN and parametric models.

    Args:
        curve_idx: Index of the learning curve to use
        data: Array of learning curves
        anchor_sizes: Training sizes corresponding to the curve points
        lcpfn_model: Trained LC-PFN model
        min_points: Minimum number of points to use for fitting
        random_cutoff: If True, uses a random cutoff point
        fixed_cutoff_idx: If provided, uses this specific cutoff index instead of generating a random one
        cutoff_percentage: If provided, uses this percentage of the curve for training

    Returns:
        Figure with plotted extrapolations
    """

    curve = data[curve_idx].flatten()
    valid_mask = np.isfinite(curve)
    curve = curve[valid_mask]

    if len(curve) <= min_points:
        return None

    curve_anchor_sizes = anchor_sizes[:len(curve)]

    if cutoff_percentage is not None:
        cutoff_idx = max(min_points, int(cutoff_percentage * len(curve)))
        cutoff_idx = min(cutoff_idx, len(curve) - 1)
    elif fixed_cutoff_idx is not None:
        cutoff_idx = fixed_cutoff_idx
    elif random_cutoff:
        cutoff_idx = random.randint(min_points, len(curve) - 1)
    else:
        cutoff_idx = len(curve) - 1

    x_train_pfn, y_train_pfn, x_test_pfn, y_test_pfn, pred_mean, pred_lower, pred_upper = extrapolate_lcpfn(
        curve, curve_anchor_sizes, lcpfn_model, min_points=min_points,
        random_cutoff=False, fixed_cutoff_idx=cutoff_idx
    )

    results = {}
    for model_name in ["MMF4", "WBL4", "POW4"]:
        result = extrapolate_parametric(
            curve, curve_anchor_sizes, model_name, min_points=min_points,
            random_cutoff=False, fixed_cutoff_idx=cutoff_idx
        )
        if result[0] is not None:
            results[model_name] = result

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(curve_anchor_sizes, curve, 'k-', linewidth=2, label='Ground Truth')

    actual_cutoff_size = curve_anchor_sizes[cutoff_idx]
    ax.axvline(x=actual_cutoff_size, color='gray', linestyle='--', linewidth=1,
               label=f'Cutoff Point (Size: {actual_cutoff_size:.0f})')

    test_anchor_sizes = curve_anchor_sizes[cutoff_idx:]
    ax.plot(test_anchor_sizes, pred_mean, 'b-', linewidth=2, label='LC-PFN')
    ax.fill_between(test_anchor_sizes, pred_lower, pred_upper, color='blue', alpha=0.2, label='LC-PFN 90% CI')

    colors = {'MMF4': 'red', 'WBL4': 'green', 'POW4': 'purple'}
    for model_name, result in results.items():
        _, _, _, _, y_pred = result
        ax.plot(curve_anchor_sizes, y_pred, color=colors[model_name], linestyle='-', linewidth=2, label=model_name)

    if cutoff_percentage:
        target_size = curve_anchor_sizes[0] + cutoff_percentage * (curve_anchor_sizes[-1] - curve_anchor_sizes[0])
        cutoff_info = f" (Target: {cutoff_percentage*100:.0f}%, Actual Size: {actual_cutoff_size:.0f}, Target Size: {target_size:.0f})"
    else:
        cutoff_info = f" (Cutoff Index: {cutoff_idx})"

    ax.set_title(f'Learning Curve Extrapolation Comparison (Curve {curve_idx}){cutoff_info}', fontsize=14)
    ax.set_xlabel('Training Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.patch.set_facecolor('white')

    return fig


# In[41]:


fig = compare_extrapolations(curve_idx=320, data=test_curves, anchor_sizes=ANCHOR_SIZE, lcpfn_model=model, cutoff_percentage=0.8)
plt.show()


# ## Performance Evaluation Metrics

# In[42]:


def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(np.where(denominator == 0, 0, np.abs(y_pred - y_true) / denominator))

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# In[43]:


def evaluate_extrapolations_with_metadata(curve_idx, curve, dataset_idx, learner_idx, anchor_sizes,
                                        lcpfn_model, min_points=10, random_cutoff=True,
                                        fixed_cutoff_idx=None, cutoff_percentage=None):
    """
    Evaluate extrapolations of a learning curve using LC-PFN and parametric models,
    keeping track of dataset and learner metadata.

    Args:
        curve_idx: Index of the learning curve in the original data
        curve: The learning curve data
        dataset_idx: Index of the dataset this curve comes from
        learner_idx: Index of the learner this curve comes from
        anchor_sizes: Training sizes corresponding to the curve points
        lcpfn_model: Trained LC-PFN model
        min_points: Minimum number of points to use for fitting
        random_cutoff: If True, uses a random cutoff point
        fixed_cutoff_idx: If provided, uses this specific cutoff index instead of generating a random one
        cutoff_percentage: If provided, uses this percentage of the curve for training

    Returns:
        List of dictionaries with metrics and metadata for each model
    """

    curve_flat = curve.flatten()
    valid_mask = np.isfinite(curve_flat)
    curve_clean = curve_flat[valid_mask]

    if len(curve_clean) <= min_points:
        return []

    curve_anchor_sizes = anchor_sizes[:len(curve_clean)]

    if cutoff_percentage is not None:
        cutoff_idx = max(min_points, int(cutoff_percentage * len(curve_clean)))
        cutoff_idx = min(cutoff_idx, len(curve_clean) - 1)
    elif fixed_cutoff_idx is not None:
        cutoff_idx = fixed_cutoff_idx
    elif random_cutoff:
        cutoff_idx = random.randint(min_points, len(curve_clean) - 1)
    else:
        cutoff_idx = len(curve_clean) - 1

    # LC-PFN extrapolation
    x_train_pfn, y_train_pfn, x_test_pfn, y_test_pfn, pred_mean, _, _ = extrapolate_lcpfn(
        curve_clean, anchor_sizes, lcpfn_model, min_points=min_points,
        random_cutoff=False, fixed_cutoff_idx=cutoff_idx
    )

    results = []
    if x_train_pfn is not None:
        smape = calculate_smape(y_test_pfn, pred_mean)
        mae = calculate_mae(y_test_pfn, pred_mean)
        mse = calculate_mse(y_test_pfn, pred_mean)

        results.append({
            'Curve_idx': curve_idx,
            'Dataset_idx': dataset_idx,
            'Learner_idx': learner_idx,
            'Learner_name': LEARNER_ZOO[learner_idx],
            'Model': 'LC-PFN',
            'SMAPE': smape,
            'MAE': mae,
            'MSE': mse,
            'Cutoff_idx': cutoff_idx,
            'Cutoff_percentage': cutoff_percentage,
            'Train_points': len(x_train_pfn),
            'Test_points': len(y_test_pfn)
        })

    # Parametric model extrapolations
    for model_name in ["MMF4", "WBL4", "POW4"]:
        result = extrapolate_parametric(
            curve_clean, anchor_sizes, model_name, min_points=min_points,
            random_cutoff=False, fixed_cutoff_idx=cutoff_idx
        )

        if result[0] is not None:
            x_train, y_train, x_test, y_test, y_pred_full = result
            y_pred_test = y_pred_full[cutoff_idx:]

            smape = calculate_smape(y_test, y_pred_test)
            mae = calculate_mae(y_test, y_pred_test)
            mse = calculate_mse(y_test, y_pred_test)

            results.append({
                'Curve_idx': curve_idx,
                'Dataset_idx': dataset_idx,
                'Learner_idx': learner_idx,
                'Learner_name': LEARNER_ZOO[learner_idx],
                'Model': model_name,
                'SMAPE': smape,
                'MAE': mae,
                'MSE': mse,
                'Cutoff_idx': cutoff_idx,
                'Cutoff_percentage': cutoff_percentage,
                'Train_points': len(x_train),
                'Test_points': len(y_test)
            })

    return results


# In[44]:


def evaluate_single_curve_with_metadata(args):
    """
    Wrapper function for evaluating a single curve with metadata.
    Needed for parallel processing to unpack arguments properly.
    """
    (curve_idx, curve, dataset_idx, learner_idx, anchor_sizes, model,
     min_points, cutoff_percentage) = args

    try:
        results = evaluate_extrapolations_with_metadata(
            curve_idx=curve_idx,
            curve=curve,
            dataset_idx=dataset_idx,
            learner_idx=learner_idx,
            anchor_sizes=anchor_sizes,
            lcpfn_model=model,
            min_points=min_points,
            random_cutoff=False,
            cutoff_percentage=cutoff_percentage
        )
        return curve_idx, results
    except Exception as e:
        print(f"Error processing curve {curve_idx}: {e}")
        return curve_idx, []

def evaluate_all_curves_with_metadata_per_scenario(curves, dataset_indices, learner_indices, anchor_sizes, lcpfn_model,
                                                   min_points=10, cutoff_percentages=None,
                                                   sample_size_per_scenario=1000, n_workers=None):
    """
    Evaluate curves with separate sampling for each cutoff-scenario combination.

    Args:
        curves: List of learning curves
        dataset_indices: List of dataset indices for each curve
        learner_indices: List of learner indices for each curve
        anchor_sizes: Training sizes corresponding to the curve points
        lcpfn_model: Trained LC-PFN model
        min_points: Minimum number of points to use for fitting
        cutoff_percentages: List of cutoff percentages to evaluate
        sample_size_per_scenario: Sample size per cutoff-scenario combination
        n_workers: Number of parallel workers

    Returns:
        DataFrame with all evaluation results and metadata
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Using {n_workers} parallel workers")

    scenarios = {
        'Flat': set(map(tuple, flat_pairs)),
        'MonoConv': set(map(tuple, mono_conv_pairs)),
        'Peaking': set(map(tuple, peaking_pairs)),
        'Dipping': set(map(tuple, dipping_pairs))
    }

    scenario_curves = {scenario: {'curves': [], 'dataset_indices': [], 'learner_indices': [], 'orig_indices': []}
                       for scenario in scenarios}

    for i, (curve, dataset_idx, learner_idx) in enumerate(zip(curves, dataset_indices, learner_indices)):
        for scenario_name, pair_set in scenarios.items():
            if (learner_idx, dataset_idx) in pair_set:
                scenario_curves[scenario_name]['curves'].append(curve)
                scenario_curves[scenario_name]['dataset_indices'].append(dataset_idx)
                scenario_curves[scenario_name]['learner_indices'].append(learner_idx)
                scenario_curves[scenario_name]['orig_indices'].append(i)
                break

    all_results = []

    for scenario_name, scenario_data in scenario_curves.items():
        if len(scenario_data['curves']) == 0:
            print(f"No curves found for scenario {scenario_name}")
            continue

        print(f"Processing scenario: {scenario_name} ({len(scenario_data['curves'])} total curves)")

        for cutoff_pct in cutoff_percentages:
            print(f"  Processing {scenario_name} with {cutoff_pct * 100:.0f}% cutoff...")

            # Sample for this specific cutoff-scenario combination
            available_curves = len(scenario_data['curves'])
            sample_size = min(sample_size_per_scenario, available_curves)

            if sample_size < available_curves:
                sample_indices = random.sample(range(available_curves), sample_size)
                sampled_curves = [scenario_data['curves'][i] for i in sample_indices]
                sampled_dataset_indices = [scenario_data['dataset_indices'][i] for i in sample_indices]
                sampled_learner_indices = [scenario_data['learner_indices'][i] for i in sample_indices]
                sampled_orig_indices = [scenario_data['orig_indices'][i] for i in sample_indices]
            else:
                sampled_curves = scenario_data['curves']
                sampled_dataset_indices = scenario_data['dataset_indices']
                sampled_learner_indices = scenario_data['learner_indices']
                sampled_orig_indices = scenario_data['orig_indices']

            # Prepare arguments for parallel processing
            args_list = [
                (orig_idx, curve, dataset_idx, learner_idx,
                 anchor_sizes, lcpfn_model, min_points, cutoff_pct)
                for orig_idx, curve, dataset_idx, learner_idx in
                zip(sampled_orig_indices, sampled_curves, sampled_dataset_indices, sampled_learner_indices)
            ]

            # Process curves in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(evaluate_single_curve_with_metadata, args): args[0]
                    for args in args_list
                }

                cutoff_results = []
                completed = 0

                for future in as_completed(future_to_idx):
                    curve_idx, results = future.result()
                    completed += 1

                    # Add scenario information to results
                    for result in results:
                        result['Scenario'] = scenario_name

                    if completed % 50 == 0:
                        print(f"    Completed {completed}/{len(args_list)} curves")

                    cutoff_results.extend(results)

                all_results.extend(cutoff_results)
                print(f"    Completed {scenario_name} {cutoff_pct * 100:.0f}% - {len(cutoff_results)} results")

    return pd.DataFrame(all_results)

sample_size = 1000

model_name_no_ext = model_name.replace('.pth', '')
file_path = Path(f'experiment2/learning_curve_extrapolation_results_{sample_size}_samples_{model_name_no_ext}.csv')
if file_path.exists():
    results_df = pd.read_csv(file_path)
else:
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    results_df = evaluate_all_curves_with_metadata_per_scenario(
        curves=test_curves,
        dataset_indices=test_dataset_indices,
        learner_indices=test_learner_indices,
        anchor_sizes=ANCHOR_SIZE,
        lcpfn_model=model,
        min_points=15,
        cutoff_percentages=[0.1, 0.3, 0.5, 0.7, 0.9],
        sample_size_per_scenario=sample_size,
        n_workers=14  # Use 14 workers for 16 CPU allocation
    )

results_df.to_csv(file_path, index=False)
