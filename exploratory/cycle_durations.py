# %%
# exp_length.py
#
# Is mean cycle duration a good baseline for phase computation?
#

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import linregress

import matplotlib.pyplot as plt

from utils.audio import AudioObject
from utils.breath import get_segment_duration
from utils.umap import loc_relative

# %%
# paths

data_parent = Path(
    r"C:\Users\ciro\Documents\code\callbacks-breathing\data\spontaneous-pre_lesion"
)

fs = 32000

# %%
# load spontaneous data
#
# dfs: all_breaths, all_files

with open(data_parent.joinpath("all_breaths.pickle"), "rb") as f:
    all_breaths = pickle.load(f)

with open(data_parent.joinpath("all_files.pickle"), "rb") as f:
    all_files = pickle.load(f)

# %%
# get cycle duration: this cycle and previous cycle

exps = all_breaths[all_breaths["type"] == "exp"]


exps["cycle_duration"] = exps.apply(
    get_segment_duration,
    axis=1,
    df=all_breaths,
    rel_index=[0, 1],
)


# %%
# rejections

rejected = []

# too short
thr = 0.1  # seconds

ii = exps["cycle_duration"] > thr
rejected.append(exps.loc[~ii])
exps = exps.loc[ii]

print(f"Rejecting {sum(~ii)} cycles (of {len(exps)}) shorter than {thr}s.")

# calls
ii = exps["amplitude"] < 1.1
rejected.append(exps.loc[~ii])
exps = exps.loc[ii]

print(
    f"Rejecting {sum(~ii)} cycles (of {len(exps)}) which are probably calls (rel. amplitude >= {ii})."
)

exps

# %%
# several previous cycle durations

n = 4  # number of previous cycles to consider

for i in range(1, n + 1):
    print(f"Getting cycle_duration_nMin{i}...")

    # get the duration of a previous cycle (nMin1 = previous cycle)
    exps[f"cycle_duration_nMin{i}"] = exps.apply(
        lambda trial: loc_relative(
            *trial.name,
            exps,
            field="cycle_duration",
            i=-2 * i,
            default=np.nan,
        ),
        axis=1,
    )

    print(f"\t{exps[f'cycle_duration_nMin{i}'].notna().sum()} non-NA.")

exps

# %%
# plot n-i vs n cycle durations

fig, axs = plt.subplots(ncols=n, figsize=(18, 5), sharex=True, sharey=True)

mean = True  # whether to plot vs n-i, or mean of (n-i, ..., n+1)

for i, ax in enumerate(axs):
    nMin = i + 1

    if mean:
        cols = [f"cycle_duration_nMin{j}" for j in range(1, nMin + 1)]
        color = "C2"
        suptitle = f"Cycle durations vs last n-i MEAN (all birds)"
    else:
        cols = [f"cycle_duration_nMin{nMin}"]
        color = "C1"
        suptitle = f"Cycle durations vs n-i (all birds)"

    # all necessary columns exists
    ii = (exps["cycle_duration"].notna()) & exps[cols].notna().apply(
        lambda x: x.all(), axis=1
    )

    x = exps.loc[ii, "cycle_duration"]
    y = exps.loc[ii, cols].apply(np.mean, axis=1)

    m, b, r_value, p_value, std_err = linregress(x, y)

    ax.scatter(
        x=x,
        y=y,
        s=0.05,
        alpha=0.5,
        label=f"$n={sum(ii)}$",
    )

    ax.plot(
        x,
        (m * x) + b,
        label=f"$y={m:.2f}x+{b:.2f}$\n$r^2={r_value**2:.2f}$",
        color=color,
    )

    ax.set_aspect("equal")
    ax.set(
        # xlabel="this cycle duration",
        # ylabel=f"$n-{nMin}$ cycle duration",
        title=f"$n-{i+1}$",
    )

    ax.legend(loc="upper right")

axs[int(n / 2)].set(xlabel="this cycle duration (s)")
axs[0].set(ylabel="$n-i$ cycle duration (s)")

fig.suptitle(suptitle)

fig.tight_layout()

# %%
# plot cycle durations

fig, ax = plt.subplots()


ii = exps["amplitude"] < 1.1

ii = ii & (exps["cycle_duration_nMin1"].notna()) & (exps["cycle_duration"].notna())

bird = None
# bird = exps.birdname.unique()[0]
# ii = ii & (exps["birdname"] == bird)

ax.set_aspect("equal")

ax.scatter(
    x=exps.loc[ii, "cycle_duration_nMin1"],
    y=exps.loc[ii, "cycle_duration"],
    # marker="o",
    s=0.05,
    alpha=0.5,
)

mean = exps.loc[ii, "cycle_duration"].mean()

ax.scatter(
    [mean],
    [mean],
    s=100,
    color="r",
    marker="+",
)

ax.set(
    xlabel="previous cycle duration (s)",
    ylabel="this cycle duration (s)",
    title=f"{bird}: spontaneous breathing cycle durations (n={sum(ii)})",
    # xlim=(-.1, 1),
    # ylim=(-.1, 1),
)

# %%
# or as hist

fig, ax = plt.subplots()

h, xedge, yedge, im = ax.hist2d(
    exps.loc[ii, "cycle_duration_nMin1"],
    exps.loc[ii, "cycle_duration"],
    bins=50,
    cmap="magma",
)
fig.colorbar(im, ax=ax, label="count")


# %%
# what are the super short ones?

thr = 0.1  # seconds
short = exps.loc[exps["cycle_duration"] < thr]

print(f"{len(short)} cycles shorter than {thr} seconds")

short

# %%
# report on which files have short cycles
# NOTE: you may have rejected these above.

short_files = (
    short.index.get_level_values("wav_filename")
    .value_counts()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots()

hist, edges = np.histogram(short_files, bins=max(short_files))
hist = hist.cumsum() / hist.sum()  # percent

# ax.stairs(hist, edges)
ax.plot(edges[:], [0, *hist], label="short breaths: file summary", color="C0")

ax.set(
    xlabel="number of short cycles per file",
    ylabel="% of files (cumulative)",
    title=f"short breaths (<{thr}s): file summary",
)

short_files

# %%
# plot some files with short cycles

file = short_files.index[-4010]
# file = r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\preLesion\spontaneous\pk19br8_150223_160149.164.cbin"

# file = short_files.loc[(short_files > 20) & (short_files < 50)].index[10]

# use processed numpy data
# npy = data_parent.joinpath(f"{Path(file).stem}.npy")
# data = np.load(npy)

# load raw data
aos = AudioObject.from_cbin(file, channel="all")
aos.reverse()
data = [ao.audio for ao in aos]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), sharex=True)

for i, ax in enumerate(axs):

    y = data[i]
    x = np.arange(len(y)) / fs

    ax.plot(x, y, c="C0")


file_breaths = exps.xs(file, level="wav_filename")
file_breaths = file_breaths.loc[file_breaths["cycle_duration"] < thr].apply(
    lambda x: (x.start_s, x.start_s + x.cycle_duration - 0.01), axis=1
)

for x in file_breaths:
    axs[1].plot(x, [0, 0], c="r", alpha=1, lw=1, marker="|")

fig.suptitle(Path(file).stem)
axs[0].set(title="audio")
axs[1].set(title="breath", xlabel="time (s)")
