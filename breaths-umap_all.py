# %%
# breaths-umap_all.py
#

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hdbscan

from utils.file import parse_birdname
from utils.umap import (
    get_time_since_stim,
    loc_relative,
    plot_cluster_traces_pipeline,
    plot_embedding_data,
    plot_violin_by_cluster,
)

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# load umap, all_breaths data

embedding_name = "embedding003-insp"
fs = 44100

parent = Path(rf"./data/umap-all_breaths")
# parent = Path(rf"M:\public\Ciro\callback-breaths\umap-all_breaths")

all_breaths_path = parent.joinpath("all_breaths.pickle")
umap_pickle_path = parent.joinpath(f"{embedding_name}.pickle")

all_trials_path = Path(r"./data/breath_figs-spline_fit/all_trials.pickle")

# breath data
print("loading all breaths data...")
with open(all_breaths_path, "rb") as f:
    all_breaths = pickle.load(f)
print("all breaths data loaded!")

# trial data
print("loading all trials data...")
with open(all_trials_path, "rb") as f:
    all_trials = pickle.load(f)
print("all trials data loaded!")

# UMAP
# note: ensure environment is EXACTLY the same as when the model was trained.
# otherwise, the model may not load.
print("loading umap embedding...")
with open(umap_pickle_path, "rb") as f:
    model = pickle.load(f)
print("umap embedding loaded!")

embedding = model.embedding_

model

# %%
# kwargs consistent across
scatter_kwargs = dict(
    s=.2,
    alpha=0.5,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

# %%
# add time since stim

all_breaths["time_since_stim_s"] = all_breaths.apply(
    get_time_since_stim,
    axis=1,
    all_trials=all_trials,
)

# %%
# take only type in embedding
ii_type = all_breaths["type"] == embedding_name.split("-")[-1]

other_breaths = all_breaths.loc[~ii_type]
all_breaths = all_breaths.loc[ii_type]

# %%
# indices for next breath

# example usage
ii_next = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=1, field="index"),
    axis=1,
)

# %%
# embedding plots

# PUTATIVE CALL

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="putative_call",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# AMPLITUDE

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="amplitude",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# DURATION

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="duration",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
    vmin=0,
    vmax=400,
    cmap_name="viridis",
)

# BREATHS SINCE LAST STIM

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="breaths_since_stim",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
    n_breaths=6,
)

# BY BIRD

plot_embedding_data(
    embedding,
    embedding_name,
    all_breaths,
    plot_type="bird_id",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# clustering

clusterer = hdbscan.HDBSCAN(
    metric="l1",
    min_cluster_size=130,
    min_samples=1,
    cluster_selection_method="leaf",
    gen_min_span_tree=True,
    cluster_selection_epsilon=0.2,
)

clusterer.fit(embedding)

ax_clusters = plot_embedding_data(
    embedding=embedding,
    embedding_name=embedding_name,
    plot_type="clusters",
    clusterer=clusterer,
    set_kwargs=set_kwargs,
    scatter_kwargs=scatter_kwargs,
    set_bad=dict(c="k", alpha=1),
)

cluster_cmap = ax_clusters.collections[0].get_cmap()

# or: highlight certain clusters
# plot_embedding_data(
#     embedding=embedding,
#     embedding_name=embedding_name,
#     plot_type="clusters",
#     clusterer=clusterer,
#     set_kwargs=set_kwargs,
#     scatter_kwargs=scatter_kwargs,
#     masked_clusters=[-1, 5, 10, 12, 13],
#     # or use: highlighted_clusters
#     set_bad=dict(c="k", alpha=1),
# )

# %%
# plot traces by cluster

cluster_set_kwargs = dict(
    ylabel="amplitude",
    ylim=[-1.05, 0.05],
)

# =========SELECTIONS=========#

# which trace to plot : select one trace_kwargs dict
trace_kwargs = dict(
    trace_type="breath_interpolated",
    aligned_to=None,
    padding_kwargs=None,
    set_kwargs={**cluster_set_kwargs, "xlim": [-0.05,1.05]},
)

# trace_kwargs = dict(
#     trace_type="breath_norm",
#     aligned_to="onset",
#     padding_kwargs=dict(pad_method="end", max_length=None),
#     set_kwargs={**cluster_set_kwargs},
# )

# which trials to plot: select "all", "call", or "no call"
select = "all"
# select = "call"
# select = "no call"


axs_cluster_traces = plot_cluster_traces_pipeline(
    **trace_kwargs,
    df=all_breaths,
    fs=fs,
    cluster_labels=clusterer.labels_,
    select=select,
    cluster_cmap=cluster_cmap,
)

# %%
# look at pre + post breath missing values

pre_breaths = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=-1, field="breath_interpolated"),
    axis=1,
)

post_breaths = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=1, field="breath_interpolated"),
    axis=1,
)

ii_prepost_dne = (pre_breaths.isna() | post_breaths.isna())

print(f"Cluster membership of breath segments where previous or next segment doesn't exist (usually: file boundaries).")
pd.Series(clusterer.labels_[ii_prepost_dne]).value_counts().sort_index()

# %%
# remove those

pre_breaths = pre_breaths.loc[~ii_prepost_dne]
breaths = all_breaths.loc[~ii_prepost_dne, "breath_interpolated"]
post_breaths = post_breaths.loc[~ii_prepost_dne]


# %%
# plot normalized-length traces w/ pre + post
# warning: takes a few minutes

axs = {k: plt.subplots()[1] for k in np.unique(clusterer.labels_)}

for i, traces in enumerate((pre_breaths, breaths, post_breaths)):
    trace_kwargs = dict(
        trace_type="breath_interpolated",
        aligned_to=None,
        padding_kwargs={"aligned_at": i - 1},
        set_kwargs={
            **cluster_set_kwargs,
            "xlim": [-1.05, 2.05],
            "ylim": [-1.05, 6.5],
        },
    )

    plot_cluster_traces_pipeline(
        **trace_kwargs,
        df=traces,
        fs=fs,
        cluster_labels=clusterer.labels_[~ii_prepost_dne],
        select=select,
        cluster_cmap=cluster_cmap,
        axs = axs,
    )


# %%
# VIOLIN PLOT BY CLUSTER

# duration
ax, parts = plot_violin_by_cluster(
    data = all_breaths.duration_s,
    cluster_labels=clusterer.labels_,
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="duration",
        ylabel="duration (s)",
        ylim=[-0.1, 0.7],
    ),
)

# amplitude
ax, parts = plot_violin_by_cluster(
    data=all_breaths.amplitude,
    cluster_labels=clusterer.labels_,
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="amplitude",
        ylabel="amplitude (normalized)",
    ),
)

# breaths since stim
ax, parts = plot_violin_by_cluster(
    data=all_breaths.stims_index,
    cluster_labels=clusterer.labels_,
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="breath segs since stim",
        ylabel="breath segs since stim",
    )
)

# %%
# PUTATIVE CALL PERCENTAGE

cluster_data = {
    i_cluster: sum(all_breaths["putative_call"] & (clusterer.labels_ == i_cluster))
    / sum((clusterer.labels_ == i_cluster))
    for i_cluster in np.unique(clusterer.labels_)
}

fig, ax = plt.subplots()

clusters, heights = cluster_data.keys(), cluster_data.values()

ax.bar(clusters, heights)
ax.set_xticks(list(clusters))

ax.set(
    xlabel="cluster",
    ylabel="% of trials with call",
    title="putative call pct",
)

# %%
# CLUSTER SIZE

cluster_data = {
    i_cluster: sum((clusterer.labels_ == i_cluster))
    for i_cluster in np.unique(clusterer.labels_)
}

fig, ax = plt.subplots()

clusters, heights = cluster_data.keys(), cluster_data.values()

ax.bar(clusters, heights)
ax.set_xticks(list(clusters))

ax.set(
    xlabel="cluster",
    ylabel="count (# trials)",
    title="cluster size",
)
