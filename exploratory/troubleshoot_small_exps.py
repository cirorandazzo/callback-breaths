# %%
# troubleshoot_small_exps.py
#
# With previous normalization strategy, 1.1 * abs(min(file)) worked well for pulling out putative calls
#
# With new normalization based on the most common insp peak instead of abs(min(file)), what's a reasonable value?

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from utils.audio import AudioObject
from utils.breath.preprocess import load_datasets
from utils.breath.plot import plot_amplitude_dist
from utils.umap import run_umap_gridsearch

import pickle

# %%

file_format = r"M:\randazzo\breathing\processed\{dataset}\{dataset}-{file}.pickle"

datasets = [
    "callback",
    "spontaneous",
    "hvc_lesion_callback",
    "hvc_lesion_spontaneous",
]

fs_dataset = {
    "callback": 44100,
    "spontaneous": 32000,
    "hvc_lesion_callback": 44100,
    "hvc_lesion_spontaneous": 32000,
}


# %%
# load datasets

all_files, all_breaths = load_datasets(datasets, file_format, fs_dataset=fs_dataset)

# cut to only this breath type
all_breaths = all_breaths.loc[all_breaths.type == "exp"]

# sort indices
all_breaths.sort_index(inplace=True)
all_files.sort_index(inplace=True)

all_breaths


# %%
#


fig, ax = plt.subplots()

# ii_range = (all_breaths.amplitude > 1) & (all_breaths.amplitude < 50)
# ii_range = (all_breaths.amplitude < 2)
ii_range = all_breaths.amplitude < 50

hist, edges = np.histogram(
    # all_breaths.amplitude,
    all_breaths.amplitude.loc[ii_range],
    bins=400,
    density=False,
)

ax.stairs(hist, edges, fill=True)

# %%

trial = all_breaths.iloc[210]

trace = np.load(trial["numpy_filename"])

fig, ax = plt.subplots()
ax.plot(trace[1, :])


# %%

small_threshold = 1

small_exps = all_breaths.loc[all_breaths.amplitude < small_threshold]

small_exps.dataset.value_counts()

# %%

all_breaths.dataset.value_counts()

# %%

print(
    f"{len(small_exps.numpy_filename.unique())} / {len(all_breaths.numpy_filename.unique())}"
)

# %%
# view small exps by file

small_exps_by_file = all_breaths.groupby(by="numpy_filename").amplitude.agg(
    n_small_exps=lambda x: sum(x < small_threshold),
    n_total_exps=lambda x: len(x),
)

small_exps_by_file["percent_small"] = (
    small_exps_by_file.n_small_exps / small_exps_by_file.n_total_exps
)

small_exps_by_file["dataset"] = [
    str(Path(x).parent.stem) for x in small_exps_by_file.index
]

small_exps_by_file.sort_values(
    axis=0, by="percent_small", inplace=True, ascending=False
)

small_exps_by_file

# %%

fig, ax = plt.subplots()

ax.hist(small_exps_by_file["percent_small"], bins=50)
ax.set(xlabel=f"percent of small exps (<{small_threshold})", ylabel="# of files")


# %%

trial = small_exps_by_file.iloc[0]

fs = all_files.loc[all_files.numpy_filename == trial.name, "fs"]
assert len(fs) == 1
fs = fs[0]

trace = np.load(trial.name)

fig, ax = plt.subplots()

for y in [0, -1]:
    ax.axhline(y=y, color="k", linewidth=0.5)

ax.plot(np.arange(trace.shape[1]) / fs, trace[0, :])

# %%
aos = AudioObject.from_file(
    r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\rd99rd72\postLesion\spontaneous\rd99rd72_080323_121412.17.cbin",
    channel="all",
)

fig, axs = plt.subplots(nrows=2, sharex=True)

for i in range(2):
    ax = axs[i]
    ax.plot(aos[i].audio)
axs[0].set_title("hvc_lesion_spontaneous: " + str(Path(trial.name).stem))


# %%

fig, ax = plt.subplots()

for dataset, df in small_exps_by_file.groupby(by="dataset"):

    ax.hist(df.percent_small, label=dataset, density=True, bins=np.linspace(0, 1, 20), histtype="step")

ax.set(ylabel="small exp pct per file", xlabel="density (files)")

ax.legend()


# %%
# plot histogram of exp amplitude by dataset 

fig, ax = plt.subplots()

def plot_amplitude_hist(df, label, ax, bins=np.linspace(0, 30, 100)):
    amplitudes = df.loc[df.type == "exp"].amplitude

    ii_out_of_range = (amplitudes < min(bins)) | (amplitudes > max(bins))

    print(
        f"Rejecting {sum(ii_out_of_range)} out of range breaths (of {len(amplitudes)} in dataset `{label}`)."
    )

    amplitudes = amplitudes.loc[~ii_out_of_range]

    hist, edges = np.histogram(amplitudes, bins=bins)  # counts

    # normalize (y = % of dataset, instead of binwidth-normalized density from plt.hist)
    hist = hist / len(amplitudes)

    ax.stairs(hist, edges, label=f"{label} ({len(amplitudes)})", alpha=0.5)


for bird, df in all_breaths.groupby(by="dataset"):
    plot_amplitude_hist(df, bird, ax)

ax.set(ylabel="density per dataset", xlabel="breath amplitude (norm)")

ax.legend()


# %%

fig, ax = plt.subplots()

callback = all_breaths.loc[all_breaths.dataset == "callback"]

for bird, df in callback.groupby(by="birdname"):
    plot_amplitude_hist(df, bird, ax)

# %%

traces_path = r"M:\randazzo\breathing\processed\insp_traces.pickle"

with open(traces_path, "rb") as f:
    traces = pickle.load(f)

traces

# %%