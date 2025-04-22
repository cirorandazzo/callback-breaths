# %%
# phase.py
#
# initial implementation of phase, some descriptive stuff, and first pass at tying into UMAP clusters

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.breath import (
    get_phase,
    plot_duration_distribution,
    plot_traces_by_cluster_and_phase,
)
from utils.file import parse_birdname
from utils.umap import (
    get_time_since_stim,
    loc_relative,
    plot_violin_by_cluster,
    get_call_segments,
    get_discrete_cmap,
)

# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# load umap, all_breaths data

do_phase_wrap = False

parent = Path(rf"./data/umap-all_breaths/v3")
fs = 44100
all_breaths_path = parent.joinpath("all_breaths.pickle")
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

all_breaths["birdname"] = all_breaths.apply(lambda x: parse_birdname(x.name[0]), axis=1)

all_trials["birdname"] = all_trials.apply(lambda x: parse_birdname(x.name[0]), axis=1)

all_breaths

# %%
# load clusters

clusters_path = parent.joinpath("clusters.pickle")

with open(clusters_path, "rb") as f:
    df_clusters = pickle.load(f)

for col in df_clusters.columns:
    all_breaths[col] = df_clusters[col]

# get preceding cluster

for field in ["cluster", "UMAP0", "UMAP1"]:
    all_breaths[f"{field}_previous"] = all_breaths.apply(
    lambda x: loc_relative(*x.name, all_breaths, field=field, i=-1),
    axis=1,
)

# %%
# select non-call breaths

no_call = all_breaths.loc[~all_breaths["putative_call"]]

no_call.type.value_counts()

# %%
# plot duration distributions and density etimation

birds = no_call["birdname"].unique()

fig, axs = plot_duration_distribution(no_call)
fig.suptitle(f"all birds (n={len(birds)})")
fig.tight_layout()

for bird in birds:
    fig, axs = plot_duration_distribution(no_call.loc[no_call["birdname"] == bird])
    fig.suptitle(bird)
    fig.tight_layout()


# %%
# get last pre-stim exp for each stim trial


def get_prestim_exp(trial, exps):
    """
    Given a stim trial (ie, a row from `all_trials` df), return
    some information about the directly preceding exp. Namely:

    i_prestim_exp:
    t_prestim_exp:
    dt: time elapsed between exp onset & stim onset
    """
    file, i_stim = trial.name

    # get stim onset & all exp onsets
    t_stim_onset = trial["trial_start_s"]
    file_exp_onsets = exps["start_s"].xs(level="wav_filename", key=file)

    # select only pre-stim exps
    pre_stim_exps = file_exp_onsets.loc[file_exp_onsets < t_stim_onset]

    try:
        # for directly preceding exp, get...
        i_prestim_exp = pre_stim_exps.idxmax()  # calls_index (ie, use df.loc)
        t_prestim_exp = pre_stim_exps.loc[i_prestim_exp]  # time of exp in file (s)
        dt = t_stim_onset - t_prestim_exp  # time between exp & onset
    except ValueError as e:
        # usually: no exp before first stim
        print(
            f"Failed on stim {i_stim} in file {file}. ({len(pre_stim_exps)} pre-stim exps found.)"
        )
        i_prestim_exp, t_prestim_exp, dt = [pd.NA] * 3

    return [i_prestim_exp, t_prestim_exp, dt]


exps = all_breaths.loc[all_breaths["type"] == "exp"]

# pd.Series forces df output
out = all_trials.apply(lambda x: pd.Series(get_prestim_exp(x, exps)), axis=1)
out.columns = ["i_prestim_exp", "t_prestim_exp", "dt_prestim_exp"]

# add prestim exp info to all_trials
for col in ["i_prestim_exp", "dt_prestim_exp"]:
    all_trials[col] = out[col]

all_trials

# %%
# get mean breath duration by bird
#
# TODO: compute a similar df for spline-fit max


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
# compute breath phase @ stim onsets

# phase wrapper for STIM TRIAL
def get_phase_wrapper(trial, mean_duration_by_bird):
    t = trial["dt_prestim_exp"]  # time since last exp @ stim onset
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


all_trials["phase"] = all_trials.apply(
    get_phase_wrapper,
    args=[mean_duration_by_bird],
    axis=1,
)

# %%
# summary histogram of phases


def stim_phase_subplot(
    times,
    phases,
    figsize=(12, 6),
    t_binwidth=0.01,
    cumulative_polar=False,
    density=False,
    linear_set_kwargs=None,
    polar_set_title_kwargs=None,
    polar_stair_kwargs=None,
    color="tab:blue",
):
    size = [1, 2]  # nrows, ncols

    fig = plt.figure(figsize=figsize)
    axs = []

    # dt histogram (lin)
    ax = fig.add_subplot(*size, 1)
    axs.append(ax)

    times = times.loc[~times.isna()]

    ax.hist(
        x=times,
        bins=np.arange(0, max(times) + t_binwidth, t_binwidth),
        color=color,
        density=density,
    )
    ax.set(**linear_set_kwargs)

    # phase histogram (polar)
    ax = fig.add_subplot(*size, 2, projection="polar")
    axs.append(ax)

    phases = phases.loc[~phases.isna()]

    height, edges = np.histogram(phases, bins=20, density=density)
    if cumulative_polar:
        height = np.cumsum(height) / np.sum(height)

    if polar_stair_kwargs is None:
        polar_stair_kwargs = {}
    polar_stair_kwargs = {**{"color": color}, **polar_stair_kwargs}

    ax.stairs(height, edges, fill=False, **polar_stair_kwargs)
    ax.set_title(**polar_set_title_kwargs)
    ax.set_rmin(0)

    return fig, axs


lsk = dict(title="last exp before stim", xlabel="delay (s)", ylabel="count")
pstk = dict(label="breath phase during stim_onset", pad=25)

fig, axs = stim_phase_subplot(
    times=all_trials["dt_prestim_exp"],
    phases=all_trials["phase"],
    linear_set_kwargs=lsk,
    polar_set_title_kwargs=pstk,
)
fig.suptitle("all birds")
axs[0].set(xlim=[0, 1.8])

for bird, data in all_trials.groupby("birdname"):
    fig, axs = stim_phase_subplot(
        times=data["dt_prestim_exp"],
        phases=data["phase"],
        linear_set_kwargs=lsk,
        polar_set_title_kwargs=pstk,
        cumulative_polar=False,
    )
    fig.suptitle(bird)

    mean_exp_dur, mean_insp_dur = [
        mean_duration_by_bird.loc[bird, type] for type in ["exp", "insp"]
    ]

    for a in [mean_exp_dur, mean_exp_dur + mean_insp_dur]:
        axs[0].axvline(a, c="k")
    axs[0].set(xlim=[0, 0.5], ylim=[0, 40])


# %%
# get call exps

# NOTE: this code didn't consider call status of subsequent call before
# refactor; so, first syll was presumably included. refactor considers
# the next call & excludes these when exclude_song = True
call_exps = get_call_segments(all_breaths, exclude_song=True)

print("Calls per bird:")
call_exps.value_counts("birdname")

# %%
# get n-1 breath segment durations

call_exps["dur_exp_nMin1"] = call_exps.apply(
    lambda x: loc_relative(
        *x.name,
        df=all_breaths.loc[all_breaths["type"] == "exp"],
        i=-2,
        field="duration_s",
    ),
    axis=1,
)

call_exps["dur_insp_nMin1"] = call_exps.apply(
    lambda x: loc_relative(
        *x.name,
        df=all_breaths.loc[all_breaths["type"] == "insp"],
        i=-1,
        field="duration_s",
    ),
    axis=1,
)

# %%
# get call breath phases

# phase wrapper for BREATHS
def get_phase_wrapper(trial, mean_duration_by_bird):
    t = trial["dur_exp_nMin1"] + trial["dur_insp_nMin1"]
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


call_exps["phase"] = call_exps.apply(
    get_phase_wrapper,
    args=[mean_duration_by_bird],
    axis=1,
)

# %%
# plot call exp phases

lsk = dict(
    title="time since last exp onset",
    xlabel="inter exp interval (s)",
    ylabel="count",
    xlim=[0, 1],
)
pstk = dict(label="phase of call exp onset", pad=25)


fig, axs = stim_phase_subplot(
    times=call_exps["dur_exp_nMin1"] + call_exps["dur_insp_nMin1"],
    phases=call_exps["phase"],
    linear_set_kwargs=lsk,
    polar_set_title_kwargs=pstk,
    color="c",
)
fig.suptitle("all birds")


for bird, data in call_exps.groupby("birdname"):
    fig, axs = stim_phase_subplot(
        times=data["dur_exp_nMin1"] + data["dur_insp_nMin1"],
        phases=data["phase"],
        linear_set_kwargs= {**lsk, "title": f"n-1 breath dur (n={len(data)} calls)"},
        polar_set_title_kwargs=pstk,
        cumulative_polar=False,
        color="c"
    )
    fig.suptitle(bird)

    mean_exp_dur, mean_insp_dur = [
        mean_duration_by_bird.loc[bird, type] for type in ["exp", "insp"]
    ]

    for a in [mean_exp_dur, mean_exp_dur + mean_insp_dur]:
        axs[0].axvline(a, c="k")


# %%
# call exps: time since stim

call_exps["time_since_stim_s"] = call_exps.apply(
    get_time_since_stim,
    axis=1,
    all_trials=all_trials,
)

t = call_exps["time_since_stim_s"]
ph = call_exps["phase"]

ii_good = ~t.isna() & ~ph.isna() 

# select a particuar time range (call exp onset)
# ii_good = ii_good & (t >= 2)
# ii_good = ii_good & (t >= 1)  & (t <= 2)
# ii_good = ii_good & (t >= 0) & (t <= 1)
# ii_good = ii_good & (t >= 0.1) & (t <= 0.4)

t = t[ii_good]
ph = ph[ii_good]

# PLOT 2D HIST
fig, ax = plt.subplots()

h, xedge, yedge, im = ax.hist2d(
    t,
    ph,
    bins=30,
    cmap="magma",
)
fig.colorbar(im, ax=ax, label="count")

ax.set(xlabel="latency to call (s, since stim)", ylabel="call phase")

# collapsed into hists
lsk = dict(
    xlabel="latency to call exp (s)",
)
pstk = dict(label="phase of call exp onset", pad=25)

fig, axs = stim_phase_subplot(
    times=t,
    t_binwidth=0.05,
    phases=ph,
    linear_set_kwargs=lsk,
    polar_set_title_kwargs=pstk,
    # polar_stair_kwargs=dict(hatch="o"),
    density=True,
    color="c",
)
fig.suptitle("all birds")

# %%
# do insp categories map nicely to phase?

phases = call_exps["phase"]

df_clusters = call_exps["cluster_previous"].map(
    lambda x: int(x.replace("insp", "").strip()), na_action="ignore"
)

ii_good = ~phases.isna() & ~df_clusters.isna()

cluster_cmap = get_discrete_cmap(min(df_clusters), max(df_clusters)+1, cmap_name="jet") 

ax, parts = plot_violin_by_cluster(
    data = np.array(phases.loc[ii_good]).astype(float),
    cluster_labels=np.array(df_clusters.loc[ii_good]).astype(int),
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="phase (call exps only)",
        ylabel="phase",
    ),
)

# %%
# do insp categories map nicely to latency?

latencies = call_exps["time_since_stim_s"]

df_clusters = call_exps["cluster_previous"].map(
    lambda x: int(x.replace("insp", "").strip()), na_action="ignore"
)

ii_good = ~latencies.isna() & ~df_clusters.isna() & (latencies > 0)

cluster_cmap = get_discrete_cmap(min(df_clusters), max(df_clusters)+1, cmap_name="jet") 

ax, parts = plot_violin_by_cluster(
    data=np.array(latencies.loc[ii_good]).astype(float),
    cluster_labels=np.array(df_clusters.loc[ii_good]).astype(int),
    cluster_cmap=cluster_cmap,
    set_kwargs=dict(
        title="call latency",
        ylabel="latency of call exp (s since stim onset)",
        ylim=[0, 1.5],
    ),
)

# %%
# plot traces sorted by insp cluster & phase

n_bins = 12
window_s = np.array([-0.75, 0.5])

trace_folder = Path("./data/cleaned_breath_traces")

phase_bins = np.linspace(0, 4, n_bins + 1) * np.pi

figs = plot_traces_by_cluster_and_phase(
    df_breaths=call_exps,
    fs=fs,
    window_s=window_s,
    trace_folder=trace_folder,
    phase_bins=phase_bins,
    cluster_col_name="cluster_previous",
)

print("Done! Figures by cluster in dict `figs`")

# %%
# save phase/cluster traces as pdf

pdf_filename = "./data/call_exps__cluster_phase-no_song_pre_next-no_wrap.pdf"

pdf_pgs = PdfPages(pdf_filename)

for fig in list(figs.values()):
    fig.savefig(pdf_pgs, format="pdf")

pdf_pgs.close()

print(f"Saved to {pdf_filename}")

# %%
# insp cluster composition (bird)

# by cluster
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(9, 8.5))

for (insp_cluster, cluster_calls), ax in zip(
    call_exps.groupby("cluster_previous"), axs.ravel()
):

    ax.set_aspect("equal")

    count = cluster_calls["birdname"].value_counts().sort_index()

    ax.pie(x=count, labels=count.index, labeldistance=None)
    ax.set(title=f"cluster {insp_cluster}\n(n={len(cluster_calls)} calls)")

axs.ravel()[3].legend(bbox_to_anchor=(1.1, 1))
fig.tight_layout()

# all calls
fig, ax = plt.subplots()

ax.set_aspect("equal")

count = call_exps["birdname"].value_counts().sort_index()

ax.pie(x=count, labels=count.index, labeldistance=None)

ax.set(title=f"all calls\n(n={len(call_exps)} calls)")
ax.legend(bbox_to_anchor=(1.1, 1))


# %%
# 3d plot: embedding + phase


def strip_cluster(c):
    if c is None:
        cl = -1
    else:
        cl = int(c.split("insp")[1])
    return cl


fields = ["UMAP0_previous", "UMAP1_previous", "phase"]

ii_nans = call_exps[fields].isna().apply(any, axis=1)

data = [call_exps.loc[~ii_nans, f] for f in fields]
colors = [strip_cluster(c) for c in call_exps.loc[~ii_nans, "cluster_previous"]]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(
    *data,
    alpha=0.7,
    c=colors,
    cmap=cluster_cmap,
)

ax.set(
    xlabel=fields[0],
    ylabel=fields[1],
    zlabel=fields[2],
)
