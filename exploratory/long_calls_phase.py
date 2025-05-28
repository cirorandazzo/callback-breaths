# %%
# long_calls_phase.py
#
# cf phase of long and short calls across datasets
#
# uses dataframes from long_calls.py

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality
from utils.breath.phase import get_phase

# %%
# load long call data

long_call_save_folder = Path(r"M:\randazzo\breathing\long_calls")

# breaths - putative call only
with open(long_call_save_folder / "thresholded_breaths-long_calls.pickle", "rb") as f:
    thresholded = pickle.load(f)

# files
with open(long_call_save_folder / "all_files-long_calls.pickle", "rb") as f:
    all_files = pickle.load(f)

# %%
# load raw all_breaths (nec. for phase durations)

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

print("Loading metadata datasets...")

all_trials, all_breaths = load_datasets(
    datasets, file_format, fs_dataset=fs_dataset, assertions=False
)
all_trials, all_breaths = TEMP_assert_file_quality(all_trials, all_breaths)

# sort indices
all_breaths.sort_index(inplace=True)
all_trials.sort_index(inplace=True)

all_breaths

# %%
# cut out things longer than 0.8s, probably garbage

for label, dset in [("all_breaths", all_breaths), ("putative calls", thresholded)]:
    fig, axs = plt.subplots(nrows=2, sharex=True)

    for type, ax in zip(dset.type.unique(), axs):
        d = dset.loc[dset.type == type].duration_s

        ax.hist(d, bins=np.linspace(0, 1, 100))
        ax.set(
            xlabel="duration",
            ylabel="breath_count",
            title=f"{label}: durations ({type}; n={len(d)})",
        )

    fig.tight_layout()

# %%
# threshold duration (<=0.8s)

duration_max = 0.8

long_files = all_breaths.loc[all_breaths.duration_s > duration_max]

assert len(long_files) > 0, "Already rejected long breaths!"

all_breaths = all_breaths.loc[all_breaths.duration_s <= duration_max]
thresholded = thresholded.loc[thresholded.duration_s <= duration_max]

print(f"Rejected breaths (>{duration_max}s, insp + exp):")
print(long_files.groupby(by="type").dataset.value_counts())


# %%
# plot breath durations by bird/dataset

bins = np.linspace(0, all_breaths.duration_s.max(), 100)

for bird, df_bird in all_breaths.groupby(by="birdname"):

    fig, axs = plt.subplots(nrows=df_bird.type.nunique(), sharex=True, figsize=(4, 7))

    fig.suptitle(bird)

    for ax, (type, df_type) in zip(axs, df_bird.groupby(by="type")):

        ax.set(title=f"{type}")

        for dataset, dset_df in df_type.groupby(by="dataset"):
            ax.hist(
                dset_df.duration_s,
                bins=bins,
                histtype="step",
                label=dataset,
            )

    axs[0].set(ylabel="breath count")
    axs[-1].set(xlabel="duration (s)")
    axs[-1].legend()


# %%
# get mean breath durations

breath_durations = (
    all_breaths.loc[all_breaths.duration_s >= 0.05]
    .groupby(by=["birdname", "type", "dataset"])
    .duration_s.agg(["mean", "std", len])
)

breath_durations

# %%
# get duration for previous segments/cycles

n_cycles = 2  # inclusive. eg, 2 --> n-1, n-2 cycles.

# previous segments
for seg in np.arange(1, 2 * n_cycles + 1):
    all_breaths.loc[:, f"duration_nMin{seg}"] = all_breaths.groupby(
        by="audio_filename"
    ).duration_s.shift(seg)

# previous cycles
for cycle in np.arange(1, n_cycles + 1):
    seg1 = 2 * cycle - 1
    seg2 = 2 * cycle

    all_breaths.loc[:, f"cycle_duration_nMin{cycle}"] = (
        all_breaths[f"duration_nMin{seg1}"] + all_breaths[f"duration_nMin{seg2}"]
    )

# %%
# plot previous cycle histogram by bird, dataset

# plot breath durations by bird/dataset

bins = np.linspace(0, all_breaths.duration_s.max(), 100)

fig, axs = plt.subplots(
    nrows=all_breaths.birdname.nunique(), sharex=True, figsize=(20, 10)
)

for ax, (bird, df_bird) in zip(axs, all_breaths.groupby(by="birdname")):

    ax.set(
        title=bird,
        # ylabel="# breath cycles",
        ylabel="density (breath cycles)",
    )

    for dataset, dset_df in df_bird.groupby(by="dataset"):
        ax.hist(
            dset_df.cycle_duration_nMin1,
            bins=bins,
            histtype="step",
            label=dataset,
            density=True,
            alpha=0.7,
        )

    axs[-1].set(xlabel="cycle duration (s)")
    axs[0].legend()

fig.suptitle("breath cycle durations")
fig.tight_layout()

# %%
# compute phase (only for expirations!)
#
# <3 min


def get_phase_wrapper(row):
    """
    Return phase given a row of all_breaths (or thresholded) which has been processed to include
    """

    # n-2 breath as yardstick
    # yardstick = dict(
    #     avgExpDur=row.duration_nMin4,
    #     avgInspDur=row.duration_nMin3,
    # )

    # avg breath as yardstick
    yardstick = dict(
        avgExpDur=breath_durations.loc[(row.birdname, "exp", row.dataset), "mean"],
        avgInspDur=breath_durations.loc[(row.birdname, "insp", row.dataset), "mean"],
    )

    if row.type == "exp":
        try:
            p = get_phase(
                breathDur=row.cycle_duration_nMin1,
                **yardstick,
                wrap=True,
            )

        except AssertionError:
            # AssertionError: algorithm only implmented for callT within 2 normal breath lengths!
            p = None

    elif row.type == "insp":
        p = None

    return p


all_breaths.loc[:, "phase"] = all_breaths.apply(get_phase_wrapper, axis=1)

all_breaths

# %%
# plot polar histograms by bird/dset (all_breaths)


def plot_polar_histograms(all_breaths):
    fig = plt.figure(
        figsize=(11, 8.5),
    )
    axs = []

    nrows = all_breaths.birdname.nunique()
    ncols = all_breaths.dataset.nunique()

    bins = np.linspace(0, all_breaths.phase.max(), 40)

    for i_bird, birdname in enumerate(sorted(all_breaths.birdname.unique())):
        bird_axs = []

        for i_dataset, dataset in enumerate(sorted(all_breaths.dataset.unique())):
            # create subplot
            ax = fig.add_subplot(
                nrows,
                ncols,
                (i_bird * ncols) + i_dataset + 1,
                projection="polar",
            )

            data = all_breaths.loc[
                (all_breaths.birdname == birdname)
                & (all_breaths.dataset == dataset)
                & (all_breaths.phase.notna()),
                "phase",
            ]

            height, edges = np.histogram(data, bins=bins, density=False)

            ax.stairs(height, edges, fill=False, color=f"C{i_dataset}")
            ax.set_title(f"{birdname}\n{dataset}\n($n={len(data)}$)", pad=10)
            ax.set_rmin(0)

            n_ticks = 8
            ax.set_xticks(np.pi * np.linspace(0, 2, n_ticks + 1))
            ax.set_xticklabels([None] * (n_ticks + 1))

            bird_axs.append(ax)

        axs.append(bird_axs)

    fig.subplots_adjust(bottom=0)

    return fig, axs


fig, axs = plot_polar_histograms(all_breaths)
fig.suptitle("Phases: all breaths", y=1)

# %%
# phases for all calls

threshold = 8
ii_call = (all_breaths.type == "exp") & (all_breaths.amplitude > threshold)
fig, axs = plot_polar_histograms(all_breaths.loc[ii_call])
fig.suptitle("Phases: all calls", y=1)

# %%
# phases for long calls

ii_long_call = thresholded["is_long_call"].reindex(all_breaths.index, fill_value=False)
fig, axs = plot_polar_histograms(all_breaths.loc[ii_long_call])
fig.suptitle("Phases: long calls", y=1)

# %%
# phases for nonlong calls

ii_nonlong_call = (~thresholded["is_long_call"]).reindex(
    all_breaths.index, fill_value=False
)
fig, axs = plot_polar_histograms(all_breaths.loc[ii_nonlong_call])
fig.suptitle("Phases: nonlong calls", y=1)
