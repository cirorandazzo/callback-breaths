# %%
# umap-03.00-train_multi_dataset.py
# 
# Training one embedding with data from multiple datasets (callback & spont data in ctrl, hvc_lesion conditions)

import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from utils.breath.preprocess import load_datasets
from utils.umap import run_umap_gridsearch

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# parameters

# path to pickled traces dataframe
traces_path = r"M:\randazzo\breathing\processed\insp_traces.pickle"

# path to save embeddings
embedding_path = r"M:\randazzo\breathing\embeddings"

# save precompuated distance matrices
# distance_matrices = Path(traces_path).parent

# path to metadata dataframes
file_format = r"M:\randazzo\breathing\processed\datasets\{dataset}\{dataset}-{file}.pickle"

datasets = [
    "callback",
    "spontaneous",
    "hvc_lesion_callback",
    "hvc_lesion_spontaneous",
]

fs_dataset = {
    "callback": 44100,
    "spontaneous": 32000,
    "hvc_lesion_callback": 44100,
    "hvc_lesion_spontaneous": 32000,
}

# breath type
breath_type = "insp"


# %%
# load metadata dfs

all_files, all_breaths = load_datasets(datasets, file_format, fs_dataset=fs_dataset)

# cut to only this breath type
all_breaths = all_breaths.loc[all_breaths.type == breath_type]

# sort indices
all_breaths.sort_index(inplace=True)
all_files.sort_index(inplace=True)

all_breaths

# %%
# load traces df
#
# takes a while, big.

with open(traces_path, "rb") as f:
    df_traces = pickle.load(f)

df_traces.sort_index(inplace=True)

df_traces

# %%
# assert traces match metadata

assert len(all_breaths) == len(df_traces), f"Different lengths! all_breaths: {len(all_breaths)} | df_traces: {len(df_traces)}"

assert all(all_breaths.index == df_traces.index), "Indices don't match!"

# %%
# check for errors
#
# these files will have df_traces.data = np.nan
#
# these are unlikely if interpolated w/o padding; all that data should exist
# (and my code never has bugs)

ii_error = df_traces.error.notna()
errored = df_traces.loc[ii_error]

print(f"{len(errored)} breaths had errors.")

assert sum(ii_error) == 0, "Deal with these errors before constructing UMAP train data below."

# %%
# subsample
#
# Subsample data, stratified by dataset

# take min dataset length from each dataset
n_samples = min(all_breaths.dataset.value_counts())

idx_subsampled = []

for d in datasets:
    # Get indices for the current dataset
    ii_dataset = all_breaths.loc[all_breaths.dataset == d].index
    
    # Randomly sample indices from the current dataset

    assert False, "seed np.random.choice!" # TODO: seed np.random.choice

    ii_sampled = np.random.choice(ii_dataset, n_samples, replace=False)  
    
    # Append sampled indices
    idx_subsampled.extend(ii_sampled)

print(idx_subsampled)

# %%
# create train data

data = np.stack(df_traces.loc[idx_subsampled, "data"])

print(f"Train data shape: {data.shape}")
# %%

conditions = [
    dict(
        n_neighbors=5,
        min_dist=0.01,
        metric="euclidean",
    ),
    dict(
        n_neighbors=5,
        min_dist=0.01,
        metric="cosine",
    ),
]

# %%
# run umap gridsearch
# 70 min for 2

errors = run_umap_gridsearch(data, conditions, embedding_path, "insp", True, False)

print(f"{len(errors)} errors.")

# %%
# make entire dataset

idx = df_traces.index
data = np.stack(df_traces["data"])

del df_traces  # save some memory

print(f"Train data shape: {data.shape}")
