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

# breath data
print("loading all breaths data...")
with open(all_breaths_path, "rb") as f:
    all_breaths = pickle.load(f)
print("all breaths data loaded!")


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
    ax.set(xlim=[-.01, 0.6], xlabel="duration_s")

    return fig, axs

birds = no_call["birdname"].unique()

fig, axs = plot_distrs(no_call)
fig.suptitle(f"all birds (n={len(birds)})")
fig.tight_layout()

for bird in birds:
    fig, axs = plot_distrs(no_call.loc[no_call["birdname"] == bird])
    fig.suptitle(bird)
    fig.tight_layout()
