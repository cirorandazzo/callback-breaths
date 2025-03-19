# %%
# phase.py
#

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.file import parse_birdname
from utils.breath import get_kde_distribution, get_phase

# %load_ext autoreload
# %autoreload 1
# %aimport utils.breath

# %%
# load umap, all_breaths data

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
            t_nMin1Exp_to_Call=t,
            avgExpDur=mean_exp_dur,
            avgInspDur=mean_insp_dur,
        )
    except AssertionError:
        # see assert in get_phase (if breath is too long)
        phase = pd.NA

    return phase


all_trials["phases"] = all_trials.apply(
    get_phase_wrapper,
    args=[mean_duration_by_bird],
    axis=1,
)

# %%
# summary histogram of phases


def stim_phase_subplot(
    all_trials,
    figsize=(12, 6),
    t_binwidth=0.01,
    cumulative_polar=False,
):
    size = [1, 2]  # nrows, ncols

    fig = plt.figure(figsize=figsize)
    axs = []

    # dt histogram (lin)
    ax = fig.add_subplot(*size, 1)
    axs.append(ax)

    dts = all_trials["dt_prestim_exp"]
    dts = dts.loc[~dts.isna()]

    ax.hist(x=dts, bins=np.arange(0, max(dts) + t_binwidth, t_binwidth))
    ax.set(title="last exp before stim", xlabel="delay (s)", ylabel="count")

    # phase histogram (polar)
    ax = fig.add_subplot(*size, 2, projection="polar")
    axs.append(ax)

    phases = all_trials["phases"]
    phases = phases.loc[~phases.isna()]

    height, edges = np.histogram(phases, bins=20)
    if cumulative_polar:
        height = np.cumsum(height) / np.sum(height)

    ax.stairs(height, edges, fill=False)
    ax.set_title("breath phase during stim_onset", pad=25)

    return fig, axs


fig, axs = stim_phase_subplot(all_trials)
fig.suptitle("all birds")
axs[0].set(xlim=[0, 1.8])

for bird, data in all_trials.groupby("birdname"):
    fig, axs = stim_phase_subplot(
        data,
        cumulative_polar=False,
    )
    fig.suptitle(bird)

    mean_exp_dur, mean_insp_dur = [
        mean_duration_by_bird.loc[bird, type] for type in ["exp", "insp"]
    ]

    for a in [mean_exp_dur, mean_exp_dur + mean_insp_dur]:
        axs[0].axvline(a, c="k")
    axs[0].set(xlim=[0, 0.5], ylim=[0, 40])
