# %%
# umap-02.00-add_spontaneous.py
#
# embed spontaneous breaths in umap space of callback data
# see umap-analyze_spontaneous.py for analysis
#
# NOTE: pretty large dataset, with some large files.


import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hdbscan

# from hdbscan.flat import approximate_predict_flat  # doesn't work.

from sklearn.neighbors import KNeighborsClassifier

import umap

from utils.umap import plot_embedding_data

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# paths

data_parent = Path(
    r"C:\Users\ciro\Documents\code\callbacks-breathing\data\spontaneous-pre_lesion"
)

save_folder = data_parent

umap_parent = Path(rf"./data/umap-all_breaths/v3")
embedding_name = "embedding003-insp"

# %%
# load spontaneous data
#
# dfs: all_breaths, all_files

with open(data_parent.joinpath("all_breaths.pickle"), "rb") as f:
    all_breaths = pickle.load(f)

with open(data_parent.joinpath("all_files.pickle"), "rb") as f:
    all_files = pickle.load(f)

# %%
# load umap model & hdbscan clusterer

# umap model
umap_pickle_path = umap_parent.joinpath(f"{embedding_name}.pickle")

with open(umap_pickle_path, "rb") as f:
    model = pickle.load(f)

print("umap loaded!")

# cluster
clusterer_pickle_path = umap_parent.joinpath(f"{embedding_name}-clusterer.pickle")

with open(clusterer_pickle_path, "rb") as f:
    clusterer = pickle.load(f)

    if isinstance(clusterer, dict):
        clusterer = clusterer["clusterer"]

print("clusterer loaded!")

# %%
model

# %%
clusterer

# %%
# fetch & stack breath segments
#
# this takes a while - ~30min

fs = 32000
interpolate_length = int(15253)  # used for first-insp umap embeddings


def get_breath_seg(trial, fs, npy_folder, interpolate_length):

    # load file data
    cbin_name = Path(trial.name[0]).stem
    print(f"Loading {cbin_name} #{trial.name[1]}...")
    breath = np.load(npy_folder.joinpath(f"{cbin_name}.npy"))[1, :]

    # get start/end indices of breath
    start_s, end_s = trial[["start_s", "end_s"]]
    ii_audio = (np.array([start_s, end_s]) * fs).astype(int)

    # cut
    cut_breath = breath[np.arange(*ii_audio)]

    # interpolate
    l = len(cut_breath)
    cut_breath = np.interp(
        np.linspace(0, l, interpolate_length),
        np.arange(l),
        cut_breath,
    )

    return cut_breath


all_insps = all_breaths.loc[all_breaths["type"] == "insp"]

insps_interp = np.vstack(
    all_insps.apply(get_breath_seg, axis=1, args=[fs, data_parent, interpolate_length])
)

# %%
# save interpolated data (.npy)
#
# warning: big file. takes ~5min

insps_interp_save_path = data_parent.joinpath("insps_interp.npy")

np.save(insps_interp_save_path, insps_interp)

# insps_interp = np.load(insps_interp_save_path)

# %%
# save downsampled data (.npy)
#
# this is what will be plotted in umap-analyze_spontaneous.py

insps_interp_downsample_savepath = data_parent.joinpath("insps_interp-downsampled.npy")

np.save(insps_interp_downsample_savepath, insps_interp[:, 0::8])


# %%
# embed spontaneous data in callback umap space
#
# ~7min

embedding = model.transform(insps_interp)

del insps_interp  # free up memory

embedding

# %%
# estimate clusters with knn
#
# WARNING: hdbscan approx_predict doesn't play well with `cluster_selection_epsilon` param
# eg, see here: https://github.com/scikit-learn-contrib/hdbscan/issues/375
#
# so, we use knn to estimate clusters instead
#
# <1min

knn_clusterer = KNeighborsClassifier(n_neighbors=10)

knn_clusterer.fit(X=model.embedding_, y=clusterer.labels_)

labels = knn_clusterer.predict(embedding)

print("Cluster counts:")
pd.Series(labels).value_counts().sort_index()

# %%
# save embedding + clusters as pickled df

embedding_save_path = save_folder.joinpath(f"{embedding_name}-spontaneous.pickle")

for umap_dim in range(embedding.shape[1]):
    all_insps.loc[:, f"UMAP{umap_dim}"] = embedding[:, umap_dim]

all_insps.loc[:, "cluster"] = labels

all_insps.loc[:, ["UMAP0", "UMAP1", "cluster"]]

all_insps.to_pickle(
    embedding_save_path
)  # ... thanks copilot for showing me this method

# %%
# plot embedding

scatter_kwargs = dict(
    s=0.1,
    alpha=0.1,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

ax = plot_embedding_data(
    embedding=embedding,
    embedding_name=embedding_name,
    plot_type="clusters",
    c=labels,
    set_kwargs=set_kwargs,
    scatter_kwargs=scatter_kwargs,
    set_bad=dict(c="k", alpha=1),
    show_colorbar=True,
)

cluster_cmap = ax.collections[0].cmap
