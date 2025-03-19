# %%
# phase.py
#

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.file import parse_birdname
from utils.breath import get_kde_distribution

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
# attempt at phase a la ziggy


def get_phase(t_nMin1Exp_to_Call, avgExpDur, avgInspDur):
    """
    python implementation of ziggy phase computation code ("phase2.m")

    t_nMin1Exp_to_Call: time between preceding expiration & call [waiting to hear from ziggy what event is meant by "call"]
    """

    phase = None

    avgBreathDur = avgExpDur + avgInspDur

    assert t_nMin1Exp_to_Call < (
        2 * avgBreathDur
    ), "algorithm only implmented for callT within 2 normal breath lengths!"

    t_nMin1Exp_to_Call = t_nMin1Exp_to_Call % avgBreathDur

    # call happens before the expiration before the call... (ie, oops)
    if t_nMin1Exp_to_Call < 0:
        phase = 0.1

    # call happens during this expiration
    elif t_nMin1Exp_to_Call < avgExpDur:
        # expiration is [0, pi]
        phase = np.pi * (t_nMin1Exp_to_Call / avgExpDur)

    # call happens during inspiration after that
    elif t_nMin1Exp_to_Call >= avgExpDur and t_nMin1Exp_to_Call < avgBreathDur:
        # inspiration is [pi, 2pi]
        phase = np.pi * (1 + (t_nMin1Exp_to_Call - avgExpDur) / avgInspDur)

    else:
        ValueError("this really shouldn't happen...")

    return phase


# %%
# compare output to ziggy matlab code
d = dict(delimiter=",")

# load matlab data saved as csv
phase = np.loadtxt("phase.csv", **d)
nMin1Exps = np.loadtxt("exps.csv", **d).astype(int)
avgExpDur, avgInspDur = np.loadtxt("avgs.csv", **d)

# compute phase (call exp onsets aligned at preWin + 1 samples)
preWin = 1000 * 32000 / 1000  # 1 second @ fs = 32000

new_phase = [get_phase(preWin - exp, avgExpDur, avgInspDur) for exp in nMin1Exps]

# summary plots
fig, axs = plt.subplots(nrows=3)

axs[0].hist(phase, bins=20, color=[0.7, 0.1, 0.1])
axs[0].set(title="matlab phases", xlabel="phase (radians)")

axs[1].hist(new_phase, bins=20, color=[0.1, 0.5, 0.1])
axs[1].set(title="python phases", xlabel="phase (radians)")

axs[2].hist(new_phase - phase, bins=20, color=[0.1, 0.3, 0.6])
axs[2].set(
    title="rd81pu26 spont; implementation comparsion",
    xlabel="difference from matlab (radians)",
)

fig.tight_layout()

