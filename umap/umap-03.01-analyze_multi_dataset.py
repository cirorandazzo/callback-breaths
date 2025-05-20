# %%
# umap-03.01-analyze_multi_dataset.py
#
# Training one embedding with data from multiple datasets (callback & spont data in ctrl, hvc_lesion conditions)

from itertools import product
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN

from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality
from utils.umap import (
    plot_cluster_traces,
    plot_embedding_data,
    prepare_clusters_axs_dict,
)
from utils.math import is_point_in_range

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# parameters

program_start = time.time()

print("Setting parameters...")

# path to pickled traces dataframe
traces_path = r"M:\randazzo\breathing\processed\insp_traces.pickle"

# path to save embeddings
embedding_folder = r"M:\randazzo\breathing\embeddings"

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

# %%
# load metadata dfs

print("Loading metadata datasets...")

all_files, all_breaths = load_datasets(
    datasets, file_format, fs_dataset=fs_dataset, assertions=False
)
all_files, all_breaths = TEMP_assert_file_quality(all_files, all_breaths)

# add next_amplitude and next_type
for col in ["type", "amplitude"]:
    all_breaths[f"next_{col}"] = all_breaths[col].groupby(by="audio_filename").shift(-1)

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
# load embedding

embedding_folder = Path(r"M:\randazzo\breathing\embeddings")

embedding_name = "embedding000-insp"
train_idx_path = embedding_folder / "train_idx.pickle"

embedding_path = embedding_folder / f"{embedding_name}.pickle"
with open(embedding_path, "rb") as f:
    model = pickle.load(f)

with open(train_idx_path, "rb") as f:
    idx_train = pickle.load(f)

    idx_train = pd.MultiIndex.from_tuples(
        idx_train, names=["numpy_filename", "calls_index"]
    )

embedding = pd.DataFrame(
    data={f"UMAP{i}": model.embedding_[:, i] for i in range(model.embedding_.shape[1])},
    index=idx_train,
)

print(all_breaths.loc[idx_train, "dataset"].value_counts())

embedding

# %%
# plot embedding by dataset

ax = plot_embedding_data(
    model.embedding_,
    embedding_name,
    df=all_breaths.loc[idx_train],
    ax=None,
    plot_type="dataset",
    scatter_kwargs=dict(s=0.8, alpha=0.5),
)

# %%
# plot embedding density

fig, ax = plt.subplots()

cmap = plt.get_cmap("magma")
cmap.set_bad("k")

hist, x_edges, y_edges, im = ax.hist2d(
    embedding["UMAP0"],
    embedding["UMAP1"],
    norm="log",
    bins=500,
    cmap=cmap,
)

fig.colorbar(im, ax=ax, label="log(sample count)")

# %%
# clustering

plt.close("all")

clusterer = HDBSCAN(
    metric="l1",
    min_cluster_size=20,
    # min_samples=40,
    cluster_selection_method="eom",  # "leaf",
    # cluster_selection_epsilon=0.9,
    # in hdbscan package impl, but not sklearn:
    gen_min_span_tree=True,
    prediction_data=True,  # speeds up subsequent predictions
)

clusterer.fit(model.embedding_)

embedding["cluster"] = pd.Series(index=idx_train, data=clusterer.labels_)

# ADD CUSTOM CLUSTERS
custom_clusters = [
    # [[-0.8, 0.2], [3.0, 6.0]],
    # [[6.8, 8.5], [8.1, 9.0]],
]

manual_clusters = []

for cc in custom_clusters:
    cl_lbl = max(embedding["cluster"]) + 1

    ii_range = embedding.apply(
        lambda x: is_point_in_range(x[["UMAP0", "UMAP1"]], cc), axis=1
    )

    print(f"Creating custom cluster: {cl_lbl} (n={sum(ii_range)})")
    print(
        f"\tOverwriting {sum(embedding.loc[ii_range, 'cluster'] != -1)} non-noise clusters!"
    )
    manual_clusters.append(cl_lbl)

    embedding.loc[ii_range, "cluster"] = cl_lbl


# PLOT
fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
ax_clusters = plot_embedding_data(
    embedding=np.array(embedding[["UMAP0", "UMAP1"]]),
    embedding_name=embedding_name,
    plot_type="clusters",
    c=embedding["cluster"],
    scatter_kwargs=dict(s=0.5, alpha=0.2),
    set_bad=dict(c="k", alpha=1),
    ax=axs[0],
)
clusterer.condensed_tree_.plot(select_clusters=True, axis=axs[1])

embedding["cluster"].value_counts().sort_index()

# %%
# plot mean traces by cluster

figs, axs, axs_dict = prepare_clusters_axs_dict(
    sorted(embedding.cluster.unique()),
    nrows=4,
    ncols=4,
    figsize=(30, 15),
    sharex=True,
    sharey=True,
)

c = "k"

for cluster, df_cluster in embedding.groupby(by="cluster"):
    ax = axs_dict[cluster]

    cluster_traces = np.vstack(df_traces.loc[df_cluster.index, "data"])

    mean = cluster_traces.mean(axis=0)
    std = cluster_traces.std(axis=0)
    x = np.linspace(0, 1, cluster_traces.shape[1])

    ax.plot(x, mean, color=c)

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2,
        color=c,
        label="$\pm$ std",
    )

    # * indicates manually created cluster
    ax.set(
        title=f"cl{cluster} (n={cluster_traces.shape[0]}){(cluster in manual_clusters)*'*'}",
    )

[fig.tight_layout() for fig in figs]
