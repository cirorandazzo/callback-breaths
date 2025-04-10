# %%
# umap-add_spontaneous.py
#
# analyzing umap on all breaths (implemented for either insp or exp)


import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hdbscan
import umap

from utils.umap import (
    plot_embedding_data,
)

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

fs = 32000
interpolate_length = int(15253)  # used for first-insp umap embeddings

def get_breath_seg(trial, fs, npy_folder, interpolate_length):
    
    # load file data
    cbin_name = Path(trial.name[0]).stem
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

insps_interp = np.vstack(all_insps.apply(get_breath_seg, axis=1, args=[fs, data_parent, interpolate_length]))

# %%
# embed spontaneous data in callback umap space

embedding = model.transform(insps_interp)

embedding

# %%
# estimate clusters

labels, strengths = hdbscan.approximate_predict(clusterer, embedding)

np.unique(labels)

# %%
# plot embedding
#
# WARNING: hdbscan approx_predict doesn't play well with `cluster_selection_epsilon` param
# eg, see here: https://github.com/scikit-learn-contrib/hdbscan/issues/375
#
# figure out how to resolve this

scatter_kwargs = dict(
    s=.1,
    alpha=0.1,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs)

ax.set(**set_kwargs, title=f"{embedding_name} - spontaneous insps")

# plot_embedding_data(
#     embedding=embedding,
#     embedding_name=embedding_name,
#     plot_type="clusters",
#     c=labels,
#     set_kwargs=set_kwargs,
#     scatter_kwargs=scatter_kwargs,
#     set_bad=dict(c="k", alpha=1),
# )


# %%

for umap_dim in range(embedding.shape[1]):
    all_insps.loc[:, f"UMAP{umap_dim}"] = embedding[:, umap_dim]

save_path_embedded = save_folder.joinpath(f"{embedding_name}-spontaneous.pickle")

with open(save_path_embedded, "wb") as f: 
    pickle.dump(all_insps, f)

# %%
raise Exception("Below are copied from umap-all_breaths. Need to build spontaneous data before running the following")


# %%
# take only type in embedding
type = embedding_name.split("-")[-1]

if type == "call_exp":
    ii_type = get_call_exps(all_breaths, return_index=True)
else:
    ii_type = all_breaths["type"] == type

other_breaths = all_breaths.loc[~ii_type]
all_breaths = all_breaths.loc[ii_type]

# %%
# indices for next breath: example usage

ii_next = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=1, field="index"),
    axis=1,
)

# %%
# embedding plots

# PHASE
raise Exception("ADD PHASE")

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


# all_breaths["cluster"] = clusterer.labels_

# ax_clusters = plot_embedding_data(
#     embedding=embedding,
#     embedding_name=embedding_name,
#     plot_type="clusters",
#     clusterer=clusterer,
#     set_kwargs=set_kwargs,
#     scatter_kwargs=scatter_kwargs,
#     set_bad=dict(c="k", alpha=1),
# )

cluster_cmap = ax_clusters.collections[0].get_cmap()

vmin = min(clusterer.labels_)
vmax = max(clusterer.labels_)
norm = Normalize(vmin, vmax)

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
# 3d plot: embedding + duration

embedding_plus = np.vstack([embedding.T, all_breaths.duration_s])

x, height, z = np.split(embedding_plus, 3, axis=0)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(x, height, z, alpha=0.2, c=clusterer.labels_, cmap=cluster_cmap, norm=norm)

ax.set(xlabel="UMAP1", ylabel="UMAP2", zlabel="insp duration (ms)")

# %%
# plot traces by cluster

figs, axs, axs_dict = prepare_clusters_axs_dict(
    labels=np.unique(clusterer.labels_),
    nrows=3,
    ncols=5,
    sharex=True,
    sharey=True,
    figsize=(13, 5.45),
)

cluster_set_kwargs = dict(
    # ylabel="amplitude",
    ylim=[-0.05, 7.05],
    ylabel=None,
    xlabel=None,
    xlim=[-10, 400],
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
    axs=axs_dict,
)

for fig in figs:
    fig.tight_layout()

# %%
# look at pre + post breath missing values

pre_breaths = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=-1, field="breath_interpolated"),
    axis=1,
)

post_ampl = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=1, field="breath_interpolated"),
    axis=1,
)

ii_prepost_dne = (pre_breaths.isna() | post_ampl.isna())

print(f"Cluster membership of breath segments where previous or next segment doesn't exist (usually: file boundaries).")
pd.Series(clusterer.labels_[ii_prepost_dne]).value_counts().sort_index()

# %%
# remove those

pre_breaths = pre_breaths.loc[~ii_prepost_dne]
breaths = all_breaths.loc[~ii_prepost_dne, "breath_interpolated"]
post_ampl = post_ampl.loc[~ii_prepost_dne]


# %%
# plot normalized-length traces w/ pre + post
# warning: takes a few minutes

# axs = {k: plt.subplots()[1] for k in np.unique(clusterer.labels_)}

# for i, traces in enumerate((pre_breaths, breaths, post_ampl)):
#     trace_kwargs = dict(
#         trace_type="breath_interpolated",
#         aligned_to=None,
#         padding_kwargs={"aligned_at": i - 1},
#         set_kwargs={
#             **cluster_set_kwargs,
#             "xlim": [-1.05, 2.05],
#             "ylim": [-1.05, 6.5],
#         },
#     )

#     plot_cluster_traces_pipeline(
#         **trace_kwargs,
#         df=traces,
#         fs=fs,
#         cluster_labels=clusterer.labels_[~ii_prepost_dne],
#         select=select,
#         cluster_cmap=cluster_cmap,
#         axs = axs,
#     )


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
ax.set(ylim=[0, None])

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

ax.bar(clusters, heights, color=[cluster_cmap(norm(cl)) for cl in clusters])
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

ax.bar(clusters, heights, color=[cluster_cmap(norm(cl)) for cl in clusters])
ax.set_xticks(list(clusters))

ax.set(
    xlabel="cluster",
    ylabel="count (# trials)",
    title="cluster size",
)

# %%
# BIRD COMPOSITION

all_breaths["birdname"] = all_breaths.apply(lambda x: parse_birdname(x.name[0]), axis=1)

# by cluster
# fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(9, 8.5))

ncols = 6
nrows = 5

figs, axs, axs_dict = prepare_clusters_axs_dict(
    labels=np.unique(clusterer.labels_),
    nrows=nrows,
    ncols=ncols,
    sharex=True,
    sharey=True,
    figsize=(11, 8.5),
)

for cluster, cluster_calls in all_breaths.groupby("cluster"):
    ax = axs_dict[cluster]
    ax.set_aspect("equal")

    count = cluster_calls["birdname"].value_counts().sort_index()

    ax.pie(x=count, labels=count.index, labeldistance=None)
    ax.set(title=f"cluster {cluster}\n(n={len(cluster_calls)} calls)")

axs.ravel()[ncols - 1].legend(bbox_to_anchor=(1.1, 1))

for fig in figs:
    fig.tight_layout()

# %%
# scatter w histograms on axes (jointplot)

post_ampl = all_breaths.apply(
    lambda x: loc_relative(*x.name, df=other_breaths, i=1, field="amplitude"),
    axis=1,
)

# Assuming `all_breaths`, `post_ampl`, and `clusterer` are already defined, as in your code.
# Example of what the data and clusterer might look like:
# clusterer.labels_ = np.random.randint(0, 3, size=len(all_breaths))  # Dummy cluster labels
# all_breaths = pd.DataFrame({'amplitude': np.random.randn(100)})  # Example data
# post_ampl = np.random.randn(100)  # Example transformed data

# Set up the grid for the scatter plot and marginal histograms
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[0.05, 1])

# Main scatter plot in the middle of the grid
ax = fig.add_subplot(gs[1, 0])

# Scatter plot
scatter = ax.scatter(
    x=all_breaths.amplitude,
    y=post_ampl,
    c=clusterer.labels_,
    cmap=cluster_cmap,
    norm=norm,
    **scatter_kwargs,
)

# Set titles and labels
ax.set(title="amplitude, insp vs next exp", xlabel="insp amp", ylabel="exp amp")

# Marginal histograms for each cluster
unique_labels = np.unique(clusterer.labels_)

# Marginal histograms
ax_x_hist = fig.add_subplot(gs[0, 0], sharex=ax)
ax_y_hist = fig.add_subplot(gs[1, 1], sharey=ax)

hist_kwargs = dict(density=True)

for label in unique_labels:
    ii_cluster = (clusterer.labels_ == label) & (~post_ampl.isna())

    hist, bin_edges = np.histogram(
        all_breaths[ii_cluster].amplitude, bins=np.linspace(0, 1, 50), **hist_kwargs
    )

    ax_x_hist.stairs(
        hist, bin_edges, label=f"Cluster {label}", color=cluster_cmap(label)
    )

    hist, bin_edges = np.histogram(
        post_ampl[ii_cluster], bins=np.linspace(0, 7, 50), **hist_kwargs
    )

    ax_y_hist.stairs(
        hist,
        bin_edges,
        label=f"Cluster {label}",
        color=cluster_cmap(label),
        orientation="horizontal",
    )

ax_x_hist.set_ylabel("Frequency")
ax_y_hist.set_xlabel("Frequency")
ax_x_hist.axis("off")
ax_y_hist.axis("off")

# Adjust colorbar to be outside plot (to the right)
cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label("Cluster")

# Adjust layout for better spacing
plt.tight_layout()

fig
