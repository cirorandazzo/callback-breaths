# %%
# umap-03.00-train_multi_dataset.py
#
# Training one embedding with data from multiple datasets (callback & spont data in ctrl, hvc_lesion conditions)
#
# all insps, 25 conditions. n_neighbors [5, 1400], min_dist [1e-3, .7], euclidean dist
# >> 13983s manakin

from itertools import product
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality
from utils.umap import run_umap_gridsearch

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap


# %%

if __name__ == "__main__":
    # %%
    # parameters

    program_start = time.time()

    print("Setting parameters...")

    # path to pickled traces dataframe
    traces_path = r"M:\randazzo\breathing\processed\insp_traces.pickle"

    # path to save embeddings
    embedding_path = r"M:\randazzo\breathing\embeddings"

    # save precompuated distance matrices
    # distance_matrices = Path(traces_path).parent

    # path to metadata dataframes
    file_format = r"M:\randazzo\breathing\processed\{dataset}\{dataset}-{file}.pickle"

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
    call_only = True

    # expiratory amplitude threshold for putative call
    threshold = 6

    os.makedirs(embedding_path, exist_ok=False)

    # %%
    # load metadata dfs

    print("Loading metadata datasets...")

    all_files, all_breaths = load_datasets(
        datasets, file_format, fs_dataset=fs_dataset, assertions=False
    )
    all_files, all_breaths = TEMP_assert_file_quality(all_files, all_breaths)

    # add next_amplitude and next_type
    for col in ["type", "amplitude"]:
        all_breaths[f"next_{col}"] = (
            all_breaths[col].groupby(by="audio_filename").shift(-1)
        )

    # cut to only this breath type
    all_breaths = all_breaths.loc[all_breaths.type == breath_type]

    if breath_type == "insp":
        all_breaths["putative_call"] = all_breaths["next_amplitude"] > threshold
    else:
        all_breaths["putative_call"] = all_breaths["amplitude"] > threshold

    if call_only:
        all_breaths = all_breaths.loc[all_breaths.putative_call]

    # sort indices
    all_breaths.sort_index(inplace=True)
    all_files.sort_index(inplace=True)

    # %%
    # load traces df
    #
    # takes a while, big.

    print("Loading traces...")

    with open(traces_path, "rb") as f:
        df_traces = pickle.load(f)

    df_traces.sort_index(inplace=True)

    print(f"Loaded! (Total time elapsed: {time.time() - program_start}s)\n")

    df_traces

    # %%
    # assert traces match metadata

    with_traces = all_breaths.index.isin(df_traces.index)
    if not with_traces.all():
        print(
            f"Not all trials in `all_breaths` have traces in `df_traces`! Offending rows ({sum(with_traces)}/{len(with_traces)}):"
        )
        for i, name in enumerate(df_traces.loc[with_traces].index):
            print(f"{i:02}: {name}")

        raise KeyError("Not all trials in `all_breaths` have traces in `df_traces`!")
    else:
        print(f"Found traces for all {len(with_traces)} rows!")

    # %%
    # remove traces with inf
    # TODO: figure out exactly why this happens.
    # file: M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_030323_134724.107.cbin

    ii_finite = df_traces.data.apply(lambda x: np.all(np.isfinite(x)))

    df_traces = df_traces.loc[ii_finite]
    all_breaths = all_breaths.loc[ii_finite, :]

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

    assert (
        sum(ii_error) == 0
    ), "Deal with these errors before constructing UMAP train data below."

    # %%
    # subsample
    #
    # Subsample data, stratified by dataset

    # take min dataset length from each dataset
    n_samples = min(all_breaths.dataset.value_counts())

    print(f"Subsampling to {n_samples} per dataset...")

    idx_subsampled = []

    seed = 141787999960910448216396417811445764406
    rng = np.random.default_rng(seed=seed)

    for d in datasets:
        # Get indices for the current dataset
        ii_dataset = all_breaths.loc[all_breaths.dataset == d].index

        # Randomly sample indices from the current dataset
        ii_sampled = rng.choice(ii_dataset, n_samples, replace=False)

        # Append sampled indices
        idx_subsampled.extend(ii_sampled)

    # save subsampled data indices
    train_idx_path = Path(embedding_path) / "train_idx.pickle"
    with open(train_idx_path, "wb") as f:
        pickle.dump(idx_subsampled, f)

    print(idx_subsampled)

    # %%
    # create train data

    data = np.stack(df_traces.loc[idx_subsampled, "data"])

    print(f"Train data shape: {data.shape}")
    print(f"Subsampled! (Total time elapsed: {time.time() - program_start}s)\n")

    # %%

    print("Making kwargs for gridsearch...")

    # particular parameter dicts to exclude
    exclude = []

    umap_params = dict(
        n_neighbors=[5, 10, 100, 700, 1400],
        min_dist=[0.001, 0.01, 0.1, 0.5, 0.7],
        metric=["euclidean", "cosine"],
    )

    # make parameter combinations
    conditions = []
    for condition in product(*umap_params.values()):
        these_params = {k: v for k, v in zip(umap_params.keys(), condition)}

        if these_params in exclude:
            print(f"Skipping this parameter set: {these_params}")
        else:
            conditions.append(these_params)

    print(f"{len(conditions)} models to be trained:")

    print(f"Kwargs prepared! (Total time elapsed: {time.time() - program_start}s)\n")
    conditions

    # %%
    # run umap gridsearch
    # 70 min for 2

    print("Training UMAP embeddings...")

    errors = run_umap_gridsearch(data, conditions, embedding_path, "insp", True, False)

    print(f"Finished training UMAPs! {len(errors)} errors.")
    print(f"Total time elapsed: {time.time() - program_start}s)\n")

    # %%
    # make entire dataset

    # idx = df_traces.index
    # data = np.stack(df_traces["data"])

    # del df_traces  # save some memory

    # print(f"Train data shape: {data.shape}")
