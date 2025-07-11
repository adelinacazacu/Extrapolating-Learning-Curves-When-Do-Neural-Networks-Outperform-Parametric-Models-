import h5py
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from scipy import stats
import itertools
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import lcpfn
from lcpfn import bar_distribution, encoders, train, utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

SEED = args.seed

print(SEED)