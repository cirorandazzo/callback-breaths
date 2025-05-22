# %%
# long_calls.py
#
# thresholding and analyzing long calls

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality

# %%
# load data

file_format = r"M:\randazzo\breathing\processed\{dataset}\{dataset}-{file}.pickle"

datasets = [
    "callback",
    "spontaneous",
    # "hvc_lesion_callback",
    # "hvc_lesion_spontaneous",
]

fs_dataset = {
    "callback": 44100,
    "spontaneous": 32000,
    # "hvc_lesion_callback": 44100,
    # "hvc_lesion_spontaneous": 32000,
}


print("Loading metadata datasets...")

all_files, all_breaths = load_datasets(
    datasets, file_format, fs_dataset=fs_dataset, assertions=False
)
all_files, all_breaths = TEMP_assert_file_quality(all_files, all_breaths)

# cut to only this breath type
all_breaths = all_breaths.loc[all_breaths.type == "exp"]

# sort indices
all_breaths.sort_index(inplace=True)
all_files.sort_index(inplace=True)

all_breaths

# %%
# amplitude histogram

fig, ax = plt.subplots()

# ii = all_breaths.amplitude > 3

ax.hist(
    all_breaths.amplitude,
    # all_breaths.loc[ii, "amplitude"],
    bins=200,
)

ax.set(
    xlabel="amplitude (norm)",
    ylabel="count",
    xlim=(2.5, 45),
    ylim=(None, 4500),
)

# %%
# amplitude histogram (by bird)

fig, ax = plt.subplots()

for bird, df_bird in all_breaths.groupby(by="birdname"):

    ax.hist(
        df_bird.amplitude,
        bins=200,
        histtype="step",
        label=bird,
        alpha=0.7,
    )

    ax.set(
        xlabel="amplitude (norm)",
        ylabel="count",
        # title=bird,
        xlim=(2.5, 45),
        ylim=(None, 4500),
    )

ax.legend()

# %%
# 2d hist: amplitude/duration

fig, ax = plt.subplots()

ii = (
    (all_breaths.amplitude < 20)
    & (all_breaths.duration_s < 0.4)
    & (all_breaths.amplitude > 5)
    # & (all_breaths.duration_s > 0.4)
)

h, xedge, yedge, im = ax.hist2d(
    all_breaths.loc[ii, "amplitude"],
    all_breaths.loc[ii, "duration_s"],
    bins=50,
    cmap="magma",
)
fig.colorbar(im, ax=ax, label="count")

ax.set(xlabel="amplitude (norm)", ylabel="duration (s)")

# %%

all_breaths
