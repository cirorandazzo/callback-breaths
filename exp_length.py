# %%
# exp_length.py
# 
# Is mean cycle duration a good baseline for phase computation?
# 

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.audio import AudioObject

from utils.umap import (
    loc_relative,
)

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


def get_segment_duration(trial, df, rel_index=[-2, -1]):
    """
    Get the total duration of breath segments in rel_index (where current == 0).

    Eg, default [-2, -1] will return the duration of the previous expiration and
    previous inspiration. (-2 = prev exp, -1 = prev insp).
    """

    # duration for: [prev exp, prev insp]
    durations = np.array(
        [
            loc_relative(*trial.name, df, field="duration_s", i=i, default=np.nan)
            for i in rel_index
        ]
    )

    return durations.sum()


exps = all_breaths[all_breaths["type"] == "exp"]


exps["this_cycle_duration"] = exps.apply(
    get_segment_duration,
    axis=1,
    df=all_breaths,
    rel_index=[0, 1],
)

# recomputing kinda slow, referencing previous is faster. (maybe, idk)
exps["prev_cycle_duration"] = exps.apply(
    lambda trial: loc_relative(
        *trial.name,
        exps,
        field="this_cycle_duration",
        i=-2,
        default=np.nan,
    ),
    axis=1,
)

# exps["prev_cycle_duration"] = exps.apply(
#     get_segment_duration,
#     axis=1,
#     df=all_breaths,
#     rel_index=[-2, -1],
# )

exps


# %%
# plot cycle durations

fig, ax = plt.subplots()


ii = exps["amplitude"] < 1.1

ii = ii & (exps["prev_cycle_duration"].notna()) & (exps["this_cycle_duration"].notna())

bird = None
# bird = exps.birdname.unique()[0]
# ii = ii & (exps["birdname"] == bird)

ax.set_aspect("equal")

ax.scatter(
    x=exps.loc[ii, "prev_cycle_duration"],
    y=exps.loc[ii, "this_cycle_duration"],
    # marker="o",
    s=0.05,
    alpha=0.5,
)

mean = exps.loc[ii, "this_cycle_duration"].mean()

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
    exps.loc[ii, "prev_cycle_duration"],
    exps.loc[ii, "this_cycle_duration"],
    bins=50,
    cmap="magma",
)
fig.colorbar(im, ax=ax, label="count")


# %%
# what are the super short ones?


thr = 0.1  # seconds
short = exps.loc[exps["this_cycle_duration"] < thr]

print(f"{len(short)} cycles shorter than {thr} seconds")

short

# %%
# report on which files have short cycles

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
# plot some of those files

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
file_breaths = file_breaths.loc[file_breaths["this_cycle_duration"] < thr].apply(
    lambda x: (x.start_s, x.start_s + x.this_cycle_duration - 0.01), axis=1
)

for x in file_breaths:
    axs[1].plot(x, [0, 0], c="r", alpha=1, lw=1, marker="|")

fig.suptitle(Path(file).stem)
axs[0].set(title="audio")
axs[1].set(title="breath", xlabel="time (s)")
