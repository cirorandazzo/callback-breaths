# %%
# map-clusters.py
#
# how do inspiratory & expiratory clusters map onto each other?
# 

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hdbscan

from utils.umap import (
    loc_relative,
    plot_embedding_data,
)

from utils.classifiers import (
    make_subcondition_confusion_matrix,
    subcondition_confusion_matrix_plot,
)

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap
# %aimport utils.classifiers

# %%

fs = 44100

parent = Path(rf"./data/umap-all_breaths")
all_breaths_path = parent.joinpath("all_breaths.pickle")

print("loading all breaths data...")
with open(all_breaths_path, "rb") as f:
    all_breaths = pickle.load(f)
print("all breaths data loaded!")

clusters = {}  # will store cluster ids

# %%
# load insp

embedding_name = "embedding003-insp"
umap_pickle_path = parent.joinpath(f"{embedding_name}.pickle")

print("loading umap embedding...")
with open(umap_pickle_path, "rb") as f:
    model = pickle.load(f)
print("umap embedding loaded!")

embedding_insp = model.embedding_
del model

clusterer_insp = hdbscan.HDBSCAN(
    metric="l1",
    min_cluster_size=130,
    min_samples=1,
    cluster_selection_method="leaf",
    gen_min_span_tree=True,
    cluster_selection_epsilon=0.2,
)

clusterer_insp.fit(embedding_insp)
clusters["insp"] = clusterer_insp.labels_

# %%

exp_path = "./data/umap-all_breaths/v2/embedding035-exp-emb_only.pickle"

print("loading umap embedding...")
with open(exp_path, "rb") as f:
    embedding_exp = pickle.load(f)
print("umap embedding loaded!")

clusterer_exp = hdbscan.HDBSCAN(
    metric="l1",
    min_cluster_size=20,
    min_samples=10,
    cluster_selection_method="leaf",
    gen_min_span_tree=True,
    cluster_selection_epsilon=0.5,
)

clusterer_exp.fit(embedding_exp)
clusters["exp"] = clusterer_exp.labels_


# %%
# plot clusters

# kwargs consistent across
scatter_kwargs = dict(
    s=0.2,
    alpha=0.5,
)

set_kwargs = dict(
    xlabel="UMAP1",
    ylabel="UMAP2",
)

ax_clusters_insp = plot_embedding_data(
    embedding=embedding_insp,
    embedding_name=embedding_name,
    plot_type="clusters",
    clusterer=clusterer_insp,
    set_kwargs=set_kwargs,
    scatter_kwargs=scatter_kwargs,
    set_bad=dict(c="k", alpha=1),
)

ax_clusters_exp = plot_embedding_data(
    embedding=embedding_exp,
    embedding_name=embedding_name,
    plot_type="clusters",
    clusterer=clusterer_exp,
    set_kwargs=set_kwargs,
    scatter_kwargs=scatter_kwargs,
    set_bad=dict(c="k", alpha=1),
)

# fix cbar ticks
#
# for ax, cl in zip(
#     [ax_clusters_exp, ax_clusters_insp],
#     [clusters["exp"], clusters["insp"]],
# ):
#     cbar = ax.collections[-1].colorbar

#     cbar.set_ticks(cl)
#     cbar.set_ticklabels(cl)


# %%
# add cluster to df

for label, rows in all_breaths.set_index("type", append=True).groupby(level="type"):

    all_breaths.loc[rows.index, "cluster"] = [
        f"{label}{int(cl):2d}" for cl in clusters[label]
    ]

all_breaths["cluster_prev"] = all_breaths.apply(
    lambda x: loc_relative(
        *x.name, df=all_breaths, i=-1, field="cluster", default=pd.NA
    ),
    axis=1,
)

all_breaths["cluster_next"] = all_breaths.apply(
    lambda x: loc_relative(
        *x.name, df=all_breaths, i=1, field="cluster", default=pd.NA
    ),
    axis=1,
)

all_breaths["cluster"]

# %%
# plot cluster mappings


def plot_cluster_map(this, rel, this_label=None, rel_label=None, cmap="Reds", **kwargs):
    """
    Plot mapping between one cluster and another.
    """

    this_unique = sorted(this.unique())
    rel_unique = sorted(rel.unique())

    cm = make_subcondition_confusion_matrix(
        true_labels_mapped=this,
        predicted_labels=rel,
        y_true_unique_labels=this_unique,
        y_pred_unique_labels=rel_unique,
    )

    ax = subcondition_confusion_matrix_plot(
        cm,
        rel_unique,
        this_unique,
        set_kwargs=dict(ylabel=this_label, xlabel=rel_label),
        cmap=cmap,
        **kwargs,
    )

    return ax


ii = all_breaths["type"] == "insp"

# % of insp cluster going to each exp cluster
ax_i2e = plot_cluster_map(
    this=all_breaths.loc[ii, "cluster"].fillna("NA"),
    rel=all_breaths.loc[ii, "cluster_next"].fillna("NA"),
    this_label="this insp cluster",
    rel_label="next exp cluster",
    cmap="Reds",
)
ax_i2e.set(title="insps to which exp cluster")

# # % of insp cluster from each exp cluster
# ax_iFe = plot_cluster_map(
#     this = all_breaths.loc[ii, "cluster"].fillna("NA"),
#     rel = all_breaths.loc[ii, "cluster_prev"].fillna("NA"),
#     this_label = "this insp cluster",
#     rel_label = "prev exp cluster",
#     cmap="Oranges",
# )
# ax_iFe.set(title="insps from which exp cluster")

ii = all_breaths["type"] == "exp"

fig, ax_eFi = plt.subplots(figsize=(12, 4))
# % of exp cluster from each insp cluster
ax_eFi = plot_cluster_map(
    this=all_breaths.loc[ii, "cluster"].fillna("NA"),
    rel=all_breaths.loc[ii, "cluster_prev"].fillna("NA"),
    this_label="this exp cluster",
    rel_label="prev insp cluster",
    cmap="Purples",
    fmt=".2f",
    ax=ax_eFi,
)
ax_eFi.set(title="exps from which insp cluster")
