#!/usr/bin/env python
# coding: utf-8

# In[34]:


import h5py
import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import lcpfn 
from lcpfn import bar_distribution, encoders
from lcpfn import train as lctrain

# create folder
import sys
import os
main_path = os.path.abspath(os.path.join(os.getcwd(), ''))
if main_path not in sys.path:
    sys.path.insert(0, main_path)

print(f"Project path: {main_path}")

TRAINING_NAME = 'training-experiment1'
experiments_path = f"{main_path}/{TRAINING_NAME}"
os.makedirs(experiments_path, exist_ok=True)


# ### Load LCDB 1.1 data

# In[35]:


### hyperparameter
OPENML_ID = {0: '3', 1: '6', 2: '11', 3: '12', 4: '13', 5: '14', 6: '15', 7: '16', 8: '18', 9: '21', 10: '22', 11: '23', 12: '24', 13: '26', 14: '28', 15: '29', 16: '30', 17: '31', 18: '32', 19: '36', 20: '37', 21: '38', 22: '44', 23: '46', 24: '50', 25: '54', 26: '55', 27: '57', 28: '60', 29: '61', 30: '151', 31: '179', 32: '180', 33: '181', 34: '182', 35: '184', 36: '185', 37: '188', 38: '201', 39: '273', 40: '293', 41: '299', 42: '300', 43: '307', 44: '336', 45: '346', 46: '351', 47: '354', 48: '357', 49: '380', 50: '389', 51: '390', 52: '391', 53: '392', 54: '393', 55: '395', 56: '396', 57: '398', 58: '399', 59: '401', 60: '446', 61: '458', 62: '469', 63: '554', 64: '679', 65: '715', 66: '718', 67: '720', 68: '722', 69: '723', 70: '727', 71: '728', 72: '734', 73: '735', 74: '737', 75: '740', 76: '741', 77: '743', 78: '751', 79: '752', 80: '761', 81: '772', 82: '797', 83: '799', 84: '803', 85: '806', 86: '807', 87: '813', 88: '816', 89: '819', 90: '821', 91: '822', 92: '823', 93: '833', 94: '837', 95: '843', 96: '845', 97: '846', 98: '847', 99: '849', 100: '866', 101: '871', 102: '881', 103: '897', 104: '901', 105: '903', 106: '904', 107: '910', 108: '912', 109: '913', 110: '914', 111: '917', 112: '923', 113: '930', 114: '934', 115: '953', 116: '958', 117: '959', 118: '962', 119: '966', 120: '971', 121: '976', 122: '977', 123: '978', 124: '979', 125: '980', 126: '991', 127: '993', 128: '995', 129: '1000', 130: '1002', 131: '1018', 132: '1019', 133: '1020', 134: '1021', 135: '1036', 136: '1040', 137: '1041', 138: '1042', 139: '1049', 140: '1050', 141: '1053', 142: '1056', 143: '1063', 144: '1067', 145: '1068', 146: '1069', 147: '1083', 148: '1084', 149: '1085', 150: '1086', 151: '1087', 152: '1088', 153: '1116', 154: '1119', 155: '1120', 156: '1128', 157: '1130', 158: '1134', 159: '1138', 160: '1139', 161: '1142', 162: '1146', 163: '1161', 164: '1166', 165: '1216', 166: '1233', 167: '1235', 168: '1236', 169: '1441', 170: '1448', 171: '1450', 172: '1457', 173: '1461', 174: '1462', 175: '1464', 176: '1465', 177: '1468', 178: '1475', 179: '1477', 180: '1478', 181: '1479', 182: '1480', 183: '1483', 184: '1485', 185: '1486', 186: '1487', 187: '1488', 188: '1489', 189: '1494', 190: '1497', 191: '1499', 192: '1501', 193: '1503', 194: '1509', 195: '1510', 196: '1515', 197: '1566', 198: '1567', 199: '1575', 200: '1590', 201: '1592', 202: '1597', 203: '4134', 204: '4135', 205: '4137', 206: '4534', 207: '4538', 208: '4541', 209: '6332', 210: '23381', 211: '23512', 212: '23517', 213: '40498', 214: '40499', 215: '40664', 216: '40668', 217: '40670', 218: '40672', 219: '40677', 220: '40685', 221: '40687', 222: '40701', 223: '40713', 224: '40900', 225: '40910', 226: '40923', 227: '40927', 228: '40966', 229: '40971', 230: '40975', 231: '40978', 232: '40979', 233: '40981', 234: '40982', 235: '40983', 236: '40984', 237: '40994', 238: '40996', 239: '41027', 240: '41142', 241: '41143', 242: '41144', 243: '41145', 244: '41146', 245: '41150', 246: '41156', 247: '41157', 248: '41158', 249: '41159', 250: '41161', 251: '41163', 252: '41164', 253: '41165', 254: '41166', 255: '41167', 256: '41168', 257: '41169', 258: '41228', 259: '41972', 260: '42734', 261: '42742', 262: '42769', 263: '42809', 264: '42810'}
LEARNER_ZOO = {0: 'SVC_linear', 1: 'SVC_poly', 2: 'SVC_rbf', 3: 'SVC_sigmoid', 4: 'Decision Trees', 5: 'ExtraTrees', 6: 'LogisticRegression', 7: 'PassiveAggressive', 8: 'Perceptron', 9: 'RidgeClassifier', 10: 'SGDClassifier', 11: 'MLP', 12: 'LDA', 13: 'QDA', 14: 'BernoulliNB', 15: 'MultinomialNB', 16: 'ComplementNB', 17: 'GaussianNB', 18: 'KNN', 19: 'NearestCentroid', 20: 'ens.ExtraTrees', 21: 'ens.RandomForest', 22: 'ens.GradientBoosting', 23: 'DummyClassifier'}
ANCHOR_SIZE = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)

### load data: validation accuracy
lc_data = h5py.File('LCDB11_ACC_265_noFS_raw_compress.hdf5', 'r')['accuracy'][...][:,:,:,:,:,1]

mean_valid_lc_nofs =np.nanmean(lc_data, axis=(2, 3))


# In[36]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

### dataset split
train_data_indices, test_data_indices = train_test_split(np.arange(len(OPENML_ID)), test_size=0.2, random_state=42)

### learner split
train_learner_indices, test_learner_indices = train_test_split(np.arange(len(LEARNER_ZOO)), test_size=0.2, random_state=42)

print(test_learner_indices)

### UD, UL, UDUL
train_data = lc_data[train_data_indices][:, train_learner_indices, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1, 137)
test_data_KDKL = lc_data[train_data_indices][:, train_learner_indices, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1, 137)
test_data_UD = lc_data[test_data_indices][:, train_learner_indices, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1, 137)
test_data_UL = lc_data[train_data_indices][:, test_learner_indices, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1, 137)
test_data_UDUL = lc_data[test_data_indices][:, test_learner_indices, :].transpose(0, 2, 3, 1, 4).reshape(-1, 1, 137)

print(f"Train data shape: {train_data.shape}")
print(f"Test data KDKL shape: {test_data_KDKL.shape}")
print(f"Test data UD shape: {test_data_UD.shape}")
print(f"Test data UL shape: {test_data_UL.shape}")
print(f"Test data UDUL shape: {test_data_UDUL.shape}")

test_KDKL_curves = []
test_KDKL_curve_lengths = []

for i in range(test_data_KDKL.shape[0]):
    curve = test_data_KDKL[i, 0, :]
    curve_length = np.count_nonzero(~np.isnan(curve))

    if curve_length > 0:
        test_KDKL_curves.append(curve)
        test_KDKL_curve_lengths.append(curve_length)

print(f"KDKL Testing set size: {len(test_KDKL_curves)} curves")

test_UL_curves = []
test_UL_curve_lengths = []

for i in range(test_data_UL.shape[0]):
    curve = test_data_UL[i, 0, :]
    curve_length = np.count_nonzero(~np.isnan(curve))

    if curve_length > 0:
        test_UL_curves.append(curve)
        test_UL_curve_lengths.append(curve_length)

print(f"UL Testing set size: {len(test_UL_curves)} curves")

test_UD_curves = []
test_UD_curve_lengths = []

for i in range(test_data_UD.shape[0]):
    curve = test_data_UD[i, 0, :]
    curve_length = np.count_nonzero(~np.isnan(curve))

    if curve_length > 0:
        test_UD_curves.append(curve)
        test_UD_curve_lengths.append(curve_length)

print(f"UD Testing set size: {len(test_UD_curves)} curves")

test_UDUL_curves = []
test_UDUL_curve_lengths = []

for i in range(test_data_UDUL.shape[0]):
    curve = test_data_UDUL[i, 0, :]
    curve_length = np.count_nonzero(~np.isnan(curve))

    if curve_length > 0:
        test_UDUL_curves.append(curve)
        test_UDUL_curve_lengths.append(curve_length)

print(f"UDUL Testing set size: {len(test_UDUL_curves)} curves")


# In[37]:


def augment(y, n_anchors):
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    if y_min >= y_max:
        return y
    anchors = np.random.uniform(0,1,(n_anchors,))
    anchors.sort()
    min_diff = 10**10
    y_final = None
    for i in range(n_anchors):
        a_i = anchors[i]
        for j in range(i+1, n_anchors):
            a_j = anchors[j]
            y_transformed = (a_j - a_i) / (y_max - y_min) * (y - y_min) + a_i
            error = np.nansum((y - y_transformed)**2)
            if error < min_diff:
                min_diff = error
                y_final = y_transformed
                #print(f"{y_transformed} ([{a_i},{a_j}], error: {error})")
    return y_final

y = np.array([0.2,0.3, 0.35, 0.3, 0.6, 0.7, 0.72])
for i in range(100):
    yy = augment(y, n_anchors=10)
    if i==0:
        plt.plot(yy,color="k",alpha=0.2, label="augmented curves")
    else:
        plt.plot(yy,color="k",alpha=0.2)
ax = plt.gca()

# Hide X and Y axes label marks
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("Training Dataset Size")
plt.ylabel("Performance (e.g., accuracy)")
plt.ylim([0,1])
plt.plot(y,color="blue", linewidth=2, label="original curve")
plt.legend()


# In[38]:


lcdb_curves = []
lcdb_curve_lengths = []

for i in range(train_data.shape[0]):  
    curve = train_data[i, 0, :]  
    curve_length = np.count_nonzero(~np.isnan(curve)) 

    if curve_length > 0:  
        lcdb_curves.append(curve)
        lcdb_curve_lengths.append(curve_length)

print(f"Training set size: {len(lcdb_curves)} curves")


# In[54]:


def save_training_test_data(train_data, test_data_KDKL, test_data_UD, test_data_UL, test_data_UDUL,
                           train_data_indices, test_data_indices,
                           train_learner_indices, test_learner_indices):

    # Create directory if it doesn't exist
    save_dir = f'../trained_models/exp1_{SEQ_LEN}_{EMSIZE}_{NLAYERS}_{NUM_BORDERS}_{LR}_{BATCH_SIZE}_{EPOCH}'
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []

    # Save training data
    train_reshaped = train_data.reshape(train_data.shape[0], -1)
    train_df = pd.DataFrame(train_reshaped, dtype=float)
    train_path = os.path.join(save_dir, 'train_data.csv')
    train_df.to_csv(train_path, index=False, na_rep='NaN')
    saved_files.append(train_path)

    # Save test data - KDKL (Known Dataset, Known Learner)
    test_kdkl_reshaped = test_data_KDKL.reshape(test_data_KDKL.shape[0], -1)
    test_kdkl_df = pd.DataFrame(test_kdkl_reshaped, dtype=float)
    test_kdkl_path = os.path.join(save_dir, 'test_data_KDKL.csv')
    test_kdkl_df.to_csv(test_kdkl_path, index=False, na_rep='NaN')
    saved_files.append(test_kdkl_path)

    # Save test data - UD (Unknown Dataset)
    test_ud_reshaped = test_data_UD.reshape(test_data_UD.shape[0], -1)
    test_ud_df = pd.DataFrame(test_ud_reshaped, dtype=float)
    test_ud_path = os.path.join(save_dir, 'test_data_UD.csv')
    test_ud_df.to_csv(test_ud_path, index=False, na_rep='NaN')
    saved_files.append(test_ud_path)

    # Save test data - UL (Unknown Learner)
    test_ul_reshaped = test_data_UL.reshape(test_data_UL.shape[0], -1)
    test_ul_df = pd.DataFrame(test_ul_reshaped, dtype=float)
    test_ul_path = os.path.join(save_dir, 'test_data_UL.csv')
    test_ul_df.to_csv(test_ul_path, index=False, na_rep='NaN')
    saved_files.append(test_ul_path)

    # Save test data - UDUL (Unknown Dataset, Unknown Learner)
    test_udul_reshaped = test_data_UDUL.reshape(test_data_UDUL.shape[0], -1)
    test_udul_df = pd.DataFrame(test_udul_reshaped, dtype=float)
    test_udul_path = os.path.join(save_dir, 'test_data_UDUL.csv')
    test_udul_df.to_csv(test_udul_path, index=False, na_rep='NaN')
    saved_files.append(test_udul_path)

    # Save indices
    indices_data = {
        'train_data_indices': train_data_indices,
        'test_data_indices': test_data_indices,
        'train_learner_indices': train_learner_indices,
        'test_learner_indices': test_learner_indices
    }

    # Create a DataFrame for indices (pad shorter arrays with -1, then convert to float)
    max_len = max(len(arr) for arr in indices_data.values())
    indices_dict = {}
    for key, arr in indices_data.items():
        # Convert to float first, then pad with NaN
        arr_float = arr.astype(float)
        padded_arr = np.pad(arr_float, (0, max_len - len(arr_float)), constant_values=np.nan)
        indices_dict[key] = padded_arr

    indices_df = pd.DataFrame(indices_dict)
    indices_path = os.path.join(save_dir, 'data_split_indices.csv')
    indices_df.to_csv(indices_path, index=False, na_rep='NaN')
    saved_files.append(indices_path)

    print(f"Saved {len(saved_files)} files:")
    for file_path in saved_files:
        print(f"  - {file_path}")

    return saved_files


# ### Extrapolation using LC-PFN

# In[55]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

# create a single eval pos, seq len samplers
class LCDB_samplers():
    # pick a training task, note that if multiple training tasks have the same length curves, the eventual batch will contain a mixture

    def __init__(self, rng=np.random, verbose=False):
        self.rng = rng
        self.curve_length = self.rng.choice(lcdb_curve_lengths)
        self.verbose = verbose

    def seq_len(self):
        self.curve_length = self.rng.choice(lcdb_curve_lengths)
        if self.verbose:
            print(f"curve length: {self.curve_length}")
        return self.curve_length

    def single_eval_pos(self):
        sep = self.rng.randint(self.curve_length)
        if self.verbose:
            print(f"single eval pos: {sep}")
        return sep

lcdb_sampler = LCDB_samplers(verbose=False)

for _ in range(3):
    print((lcdb_sampler.seq_len(), lcdb_sampler.single_eval_pos()))


# In[56]:


AUGMENT = True


# In[57]:


ddevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {ddevice}")

# generating batches from the LCDB dataset
def get_batch_lcdb_real( batch_size, seq_len, num_features, device=ddevice, noisy_target=True, **_ ):
    assert num_features == 1
    assert noisy_target

    ###### sample the curve data
    x = np.empty((batch_size, seq_len), dtype=float)
    y_target = np.empty((batch_size, seq_len), dtype=float)
    y_noisy = np.empty((batch_size, seq_len), dtype=float)

    # determine a list of cids having the desired curve length (TODO: precompute)
    candidates = [cid for (cid, clen) in enumerate(lcdb_curve_lengths) if clen == seq_len]
    cids = np.random.choice(candidates, batch_size, replace=True)
    # replace = TRUE, cause sometimes the number of existing curves is smaller than batch size

    for i, cid in enumerate(cids):
        curve = lcdb_curves[cid]
        if AUGMENT:
            curve = augment(curve,n_anchors=10)
        token_idx = 0
        for j, y in enumerate(curve):
            if np.isnan(y):
                continue
            else:
                x[i, token_idx] = j+1
                y_target[i, token_idx] = y
                y_noisy[i, token_idx] = y
                token_idx += 1
        # assert(token_idx == seq_len)

    # turn numpy arrays into correctly shaped torch tensors & move them to device
    x = (torch.arange(1, seq_len + 1).repeat((num_features, batch_size, 1)).transpose(2, 0).to(device)).float()
    y_target = torch.from_numpy(y_target).transpose(1, 0).to(device).float()
    y_noisy = torch.from_numpy(y_noisy).transpose(1, 0).to(device).float()

    return x, y_noisy, y_target

def train_lcdbpfn(get_batch_func, seq_len, emsize, nlayers, num_borders, lr, batch_size, epochs, checkpoint_path):

    # training hyperparameters
    hps = {}

    # PFN training hyperparameters
    dataloader = lcpfn.priors.utils.get_batch_to_dataloader(get_batch_func)  # type: ignore

    num_features = 1

    ys = []
    for curve in lcdb_curves:
        ys.extend(curve)

    ys = torch.FloatTensor(ys)
    ys = torch.normal(ys,1e-5)

    bucket_limits = bar_distribution.get_bucket_limits(num_borders, ys=ys)
    #print(bucket_limits)

    # Discretization of the predictive distributions
    criterions = {
        num_features: {
            num_borders: bar_distribution.FullSupportBarDistribution(bucket_limits)
        }
    }


    config = dict(
        nlayers=nlayers,
        priordataloader_class=dataloader,
        criterion=criterions[num_features][num_borders],
        encoder_generator=lambda in_dim, out_dim: torch.nn.Sequential(
            encoders.Normalize(0.0, 101.0),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            encoders.Linear(in_dim, out_dim),
        ),
        emsize=emsize,
        nhead=(emsize // 128),
        warmup_epochs=(epochs // 4),
        y_encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        batch_size=batch_size,
        scheduler=lcpfn.utils.get_cosine_schedule_with_warmup,
        extra_prior_kwargs_dict={
            # "num_workers": 10,
            "num_features": num_features,
            "hyperparameters": {
                **hps,
            },
        },
        epochs=epochs,
        lr=lr,
        bptt=seq_len,
        seq_len_gen=lcdb_sampler.seq_len,
        single_eval_pos_gen=lambda: LCDB_samplers().single_eval_pos(),
        aggregate_k_gradients=1,
        nhid=(emsize * 2),
        steps_per_epoch=100,
        train_mixed_precision=False,
        checkpoint_path = checkpoint_path
    )

    return lctrain.train(**config)


# In[52]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

# local training
SEQ_LEN = 140
EMSIZE = 512
NLAYERS = 12
NUM_BORDERS = 1000
LR = 0.0001
BATCH_SIZE = 100
EPOCH = 1000

# batch size 100 x step size 100 x 1000 epochs = 10 million augmented curves will be generated during the batch creation in training

result = train_lcdbpfn(
    get_batch_func = get_batch_lcdb_real,
    seq_len = SEQ_LEN,
    emsize = EMSIZE,
    nlayers = NLAYERS,
    num_borders = NUM_BORDERS,
    lr = LR,
    batch_size = BATCH_SIZE,
    epochs = EPOCH,
    checkpoint_path=experiments_path
)


# In[58]:


model = result[2]
os.makedirs(f'../trained_models/exp1_{SEQ_LEN}_{EMSIZE}_{NLAYERS}_{NUM_BORDERS}_{LR}_{BATCH_SIZE}_{EPOCH}', exist_ok=True)
torch.save(model, f'../trained_models/exp1_{SEQ_LEN}_{EMSIZE}_{NLAYERS}_{NUM_BORDERS}_{LR}_{BATCH_SIZE}_{EPOCH}/lcpfn_model_exp1_{SEQ_LEN}_{EMSIZE}_{NLAYERS}_{NUM_BORDERS}_{LR}_{BATCH_SIZE}_{EPOCH}.pth')
print("Model saved successfully!")

saved_files = save_training_test_data(
    train_data=train_data,
    test_data_KDKL=test_data_KDKL,
    test_data_UD=test_data_UD,
    test_data_UL=test_data_UL,
    test_data_UDUL=test_data_UDUL,
    train_data_indices=train_data_indices,
    test_data_indices=test_data_indices,
    train_learner_indices=train_learner_indices,
    test_learner_indices=test_learner_indices
)
print("Data saved successfully!")
print(model)


# ### Extrapolate the learning curve with a cutoff

# In[25]:


model = torch.load(f'../trained_models/exp1_{SEQ_LEN}_{EMSIZE}_{NLAYERS}_{NUM_BORDERS}_{LR}_{BATCH_SIZE}_{EPOCH}/lcpfn_model_exp1_{SEQ_LEN}_{EMSIZE}_{NLAYERS}_{NUM_BORDERS}_{LR}_{BATCH_SIZE}_{EPOCH}.pth', weights_only=False)
model.eval()


# In[29]:


random_idx = random.randint(0, train_data.shape[0] - 1)
curve = train_data[250]  # (1, 137)
curve = curve[~np.isnan(curve)]

y = torch.from_numpy(curve).float().unsqueeze(-1)
x = torch.arange(1, y.shape[0] + 1).unsqueeze(-1).float()  

# construct 
num_last_anchor = 20
cutoff = len(curve) - num_last_anchor

x_train = x[:cutoff]
y_train = y[:cutoff]
x_test = x[cutoff:]
qs = [0.05, 0.5, 0.95]

normalizer = lcpfn.utils.identity_normalizer()

y_train_norm = normalizer[0](y_train)

# forward
single_eval_pos = x_train.shape[0]
x = torch.cat([x_train, x_test], dim=0).unsqueeze(1)
y = y_train.unsqueeze(1)

logits = model((x, y), single_eval_pos=single_eval_pos)

predictions = normalizer[1](
    torch.cat([model.criterion.icdf(logits, q) for q in qs], dim=1)
)

x_test_np = x[cutoff:].detach().cpu().numpy().flatten()
pred_mean = predictions[:, 1].detach().cpu().numpy()
pred_lower = predictions[:, 0].detach().cpu().numpy()
pred_upper = predictions[:, 2].detach().cpu().numpy()

# plot
plt.plot(curve, "black", label="target")
plt.plot(x_test_np, pred_mean, "blue", label="Extrapolation by PFN")
plt.fill_between(x_test_np, pred_lower, pred_upper, color="blue", alpha=0.2, label="CI of 90%")
plt.vlines(cutoff, 0, 1, linewidth=0.5, color="k", label="cutoff")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.show()


# In[ ]:




