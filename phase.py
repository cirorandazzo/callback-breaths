# %%
# phase.py
#

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.breath import (
    get_kde_distribution,
    get_phase,
)
from utils.file import parse_birdname
from utils.umap import (
    get_time_since_stim,
    loc_relative,
    plot_violin_by_cluster,
    get_discrete_cmap
)


# %load_ext autoreload
# %autoreload 1
# %aimport utils.umap

# %%
# load umap, all_breaths data

do_phase_wrap = False

parent = Path(rf"./data/umap-all_breaths")
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

clusters_path = "./data/umap-all_breaths/clusters.pickle"

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


def plot_distrs(no_call):
    fig, axs = plt.subplots(nrows=2, sharex=True)

    for (type, breaths), ax in zip(no_call.groupby("type"), axs):
        data = breaths["duration_s"]
        title = f"{type} (n={len(data)})"

        ax.hist(data, bins=np.linspace(0, 1.6, 200), label="data", density=True)
        ax.set(title=title, ylabel="density")

        kde, x_kde, y_kde = get_kde_distribution(data, xlim=(0, 1.6), xsteps=200)

        ax.plot(
            x_kde,
            y_kde,
            label="kde",
        )

        ax.axvline(x=np.mean(data), c="r", linewidth=0.5, linestyle="--", label="mean")

    fig.tight_layout()

    axs[0].legend()
    ax.set(xlim=[-0.01, 0.6], xlabel="duration_s")

    return fig, axs


birds = no_call["birdname"].unique()

fig, axs = plot_distrs(no_call)
fig.suptitle(f"all birds (n={len(birds)})")
fig.tight_layout()

for bird in birds:
    fig, axs = plot_distrs(no_call.loc[no_call["birdname"] == bird])
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


def get_phase_wrapper(trial, mean_duration_by_bird):
    t = trial["dt_prestim_exp"]
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

ii_exp = all_breaths["type"] == "exp"
ii_call = all_breaths["putative_call"]

# exclude song: if the prev exp was a call, it's probably song.
ii_prev_call = all_breaths.apply(
    lambda x: loc_relative(*x.name, all_breaths, field="putative_call", i=-2, default=False,),
    axis=1,
)

call_exps = all_breaths.loc[ii_exp & ii_call & ~ii_prev_call]

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


def get_phase_wrapper_call_exps(trial, mean_duration_by_bird):
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
    get_phase_wrapper_call_exps,
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

for insp_cluster, cluster_calls in call_exps.groupby("cluster_previous"):
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
        traces = calls_in_phase.apply(get_wav_snippet, axis=1, args=[window_fr, fs, trace_folder])

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

# %%
# save phase/cluster traces as pdf

pdf_filename = "./data/call_exps__cluster_phase-no_song-no_wrap.pdf"

pdf_pgs = PdfPages(pdf_filename)

for fig in list(figs.values()):
    fig.savefig(pdf_pgs, format="pdf")

pdf_pgs.close()

print(f"Saved to {pdf_filename}")
