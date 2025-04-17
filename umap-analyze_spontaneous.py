# %%
# umap-analyze_spontaneous.py
#
# analyzing umap on all breaths (implemented for either insp or exp)

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from matplotlib.colors import Normalize

import hdbscan

# from hdbscan.flat import approximate_predict_flat  # doesn't work.

from sklearn.neighbors import KNeighborsClassifier

import umap

from utils.breath import (
    get_segment_duration,
)
from utils.breath import (
    get_phase,
    plot_duration_distribution,
)
from utils.umap import (
    plot_embedding_data,
    plot_violin_by_cluster,
    prepare_clusters_axs_dict,
)

# %load_ext autoreload
# %autoreload 1
# %aimport utils.breath

# %%
# set paths

data_parent = Path(
    r"C:\Users\ciro\Documents\code\callbacks-breathing\data\spontaneous-pre_lesion"
)

save_folder = data_parent

umap_parent = Path(rf"./data/umap-all_breaths/v3")
embedding_name = "embedding003-insp"

# %%
# load downsampled data (.npy)
#

insps_interp_downsample_save_path = data_parent.joinpath("insps_interp-downsampled.npy")

insps_interp = np.load(insps_interp_downsample_save_path)

# %%
# load embedding/cluster data

embedding_save_path = save_folder.joinpath(f"{embedding_name}-spontaneous.pickle")

all_insps = pd.read_pickle(embedding_save_path)

# %%
# load all breaths data

all_breaths_save_path = data_parent.joinpath("all_breaths.pickle")

all_breaths = pd.read_pickle(all_breaths_save_path)

# %%
# plot embedding

embedding = np.array(all_insps.loc[:, ["UMAP0", "UMAP1"]])
labels = np.array(all_insps["cluster"])

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
norm = Normalize(min(labels), max(labels))

# %%
# embedding plots

# AMPLITUDE
plot_embedding_data(
    embedding,
    embedding_name,
    all_insps,
    plot_type="amplitude",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# DURATION
plot_embedding_data(
    embedding,
    embedding_name,
    all_insps,
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
    all_insps,
    plot_type="bird_id",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# PUTATIVE CALL
plot_embedding_data(
    embedding,
    embedding_name,
    all_insps,
    plot_type="putative_call",
    scatter_kwargs=scatter_kwargs,
    set_kwargs=set_kwargs,
)

# %%
# PUTATIVE CALL PERCENTAGE

cluster_data = {
    i_cluster: sum(all_insps["putative_call"] & (labels == i_cluster))
    / sum((labels == i_cluster))
    for i_cluster in np.unique(labels)
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
    i_cluster: sum((labels == i_cluster)) for i_cluster in np.unique(labels)
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
# get trace by cluster

cluster_traces = {cl: insps_interp[labels == cl] for cl in np.unique(labels)}

# %%
# plot cluster means

figs, axs, axs_dict = prepare_clusters_axs_dict(
    labels=set(labels),
    nrows=4,
    ncols=4,
    sharex=True,
    sharey=True,
    figsize=(15, 15),
)

x = np.linspace(0, 1, insps_interp.shape[1])

for cluster, traces in cluster_traces.items():

    ax = axs_dict[cluster]

    ax.plot(x, traces.mean(axis=0), color="k", label="mean")
    ax.fill_between(
        x,
        traces.mean(axis=0) - traces.std(axis=0),
        traces.mean(axis=0) + traces.std(axis=0),
        alpha=0.2,
        color="k",
        label="$\pm$ std",
    )
    ax.set_title(
        f"cl{cluster} (n={traces.shape[0]})",
        color=cluster_cmap(norm(cluster)),
    )

axs.ravel()[-1].legend()

# %%
# plot duration distributions

no_call = all_breaths.loc[~all_breaths.putative_call]

birds = no_call["birdname"].unique()

hk = dict(color="indigo")
kk = dict(color="darkorange")

fig, axs = plot_duration_distribution(no_call, hist_kwargs=hk, kde_kwargs=kk)
fig.suptitle(f"all birds (n={len(birds)})")
fig.tight_layout()

for bird in birds:
    fig, axs = plot_duration_distribution(
        no_call.loc[no_call["birdname"] == bird], hist_kwargs=hk, kde_kwargs=kk
    )
    fig.suptitle(bird)
    fig.tight_layout()


# %%
# %%
# get mean breath duration by bird


def get_mean_durations(df_no_call_breaths):
    means = {}
    for type, breaths in df_no_call_breaths.groupby("type"):
        means[type] = np.mean(breaths["duration_s"])

    return means


birds = no_call["birdname"].unique()

mean_duration_by_bird = {
    bird: get_mean_durations(no_call.loc[no_call["birdname"] == bird]) for bird in birds
}
mean_duration_by_bird["all"] = get_mean_durations(no_call)
mean_duration_by_bird = pd.DataFrame(mean_duration_by_bird).T

mean_duration_by_bird

# %%
# get phases
#
# get duration of prev exp + this insp. so, phase corresponds
# to onset of following expiration

do_phase_wrap = False

all_insps["cycle_duration"] = all_insps.apply(
    get_segment_duration,
    axis=1,
    df=all_breaths,
    rel_index=[-1, 0],
)


def get_phase_wrapper(trial, mean_duration_by_bird):
    t = trial["cycle_duration"]
    bird = trial["birdname"]

    if pd.isna(t):
        return t

    mean_exp_dur, mean_insp_dur = [
        mean_duration_by_bird.loc[bird, type] for type in ["exp", "insp"]
    ]

    try:
        phase = get_phase(
            breathDur=t,
            avgExpDur=mean_exp_dur,
            avgInspDur=mean_insp_dur,
            wrap=do_phase_wrap,
        )
    except AssertionError:
        # see assert in get_phase (if breath is too long)
        phase = pd.NA

    return phase


all_insps["phase"] = all_insps.apply(
    get_phase_wrapper,
    args=[mean_duration_by_bird],
    axis=1,
)


# %%
# VIOLINS

# duration
ax, parts = plot_violin_by_cluster(
    data=all_insps.duration_s,
    cluster_labels=labels,
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="duration",
        ylabel="duration (s)",
        ylim=[-0.01, 0.3],
    ),
)

# amplitude
ax, parts = plot_violin_by_cluster(
    data=all_insps.amplitude,
    cluster_labels=labels,
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="amplitude",
        ylabel="amplitude (normalized)",
    ),
)

# phase
ax, parts = plot_violin_by_cluster(
    data=all_insps.phase.to_numpy(na_value=np.nan, dtype=float),
    cluster_labels=labels,
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="phase",
        ylabel="phase (unwrapped)",
    ),
)

# %%
# bins of phase

n_bins = 12
bins = np.pi * np.linspace(0, 4, n_bins)

all_insps["phase_bin"] = -1 * np.ones_like(all_insps["phase"], dtype=int)

for i, (st, en) in enumerate(zip(bins[:-1], bins[1:])):
    ii = (all_insps.phase >= st) & (all_insps.phase < en)

    all_insps.loc[ii, "phase_bin"] = i

print(
    "\t".join(
        [
            "cluster",
            "phase_bin",
            "calls",
            "/",
            "breaths",
        ]
    )
)

for (cluster, phase_bin), breaths in all_insps.groupby(by=["cluster", "phase_bin"]):

    print(
        "\t".join(
            [
                f"cl {cluster}",
                f"ph {phase_bin}",
                f"{sum(breaths.putative_call)}",
                f"/",
                f"{len(breaths)}",
            ]
        )
    )


# %%
# plot by cluster & phase

raise Exception("Need to implement")

n_bins = 12
window_s = np.array([-0.75, 0.5])

trace_kwargs = dict(
    linewidth=0.1,
    color="k",
    alpha=0.4,
)

axline_kwarg = dict(
    linewidth=1,
    color="tab:blue",
)

trace_folder = Path("./data/cleaned_breath_traces")


def get_wav_snippet(trial, window_fr, fs, trace_folder):
    """
    post time includes breath length

    trace_folder should contain .npy copies of the .wav files with matching file names & no folder structure
    """

    # get file
    wav_file = trial.name[0]
    np_file = trace_folder.joinpath(Path(wav_file).stem + ".npy")
    breath = np.load(np_file)

    # get indices
    onset = int(fs * trial["start_s"])
    ii = np.arange(*window_fr) + onset

    try:
        return breath[ii]
    except IndexError:
        return pd.NA


window_fr = (fs * window_s).astype(int)
x = np.linspace(*window_s, np.ptp(window_fr))

phase_bins = np.linspace(0, 4, n_bins + 1) * np.pi

figs = {}

for insp_cluster, cluster_calls in all_breaths.groupby("cluster"):
    phases = cluster_calls["phase"]

    cols = 4
    fig, axs = plt.subplots(
        figsize=(11, 8.5),
        ncols=cols,
        nrows=np.ceil(n_bins / cols).astype(int),
        sharex=True,
        sharey=True,
    )

    for st_ph, en_ph, ax in zip(
        phase_bins[:-1],
        phase_bins[1:],
        axs.ravel()[:n_bins],
    ):
        calls_in_phase = cluster_calls.loc[(phases > st_ph) & (phases <= en_ph)]
        traces = calls_in_phase.apply(
            get_wav_snippet, axis=1, args=[window_fr, fs, trace_folder]
        )

        ax.axhline(**axline_kwarg)
        ax.axvline(**axline_kwarg)

        if len(traces) != 0:
            traces = np.vstack(traces.loc[traces.notnull()])

            ax.plot(x, traces.T, **trace_kwargs)
            ax.plot(x, traces.mean(axis=0), color="r")

        ax.set(
            title=f"({st_ph:.2f},{en_ph:.2f}], n={traces.shape[0]}",
            xlim=window_s,
        )

    fig.suptitle(f"Cluster: {insp_cluster} (n={len(cluster_calls)} call exps)")

    figs[insp_cluster] = fig

print("Done! Figures by cluster in dict `figs`")
