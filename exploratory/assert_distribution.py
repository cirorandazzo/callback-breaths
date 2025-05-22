# %%
# assert_distribution.py
#
# what are good ways to automatically sort out files with strange amplitude distributions?
# 
# answer: control for respiratory rate (2-5Hz) and bimodality (hartigans' dip stat >= 0.025)
# 
# also: reasonable putative call threshold for new normalization ~6.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from utils.audio import AudioObject
from utils.breath.plot import plot_trace_amplitude_distribution
from utils.breath.preprocess import load_datasets, reject_files

# %load_ext autoreload
# %autoreload 1
# %aimport utils.breath.plot
# %aimport utils.breath.preprocess
# %aimport utils.audio

# %%

output_folder = r"M:\randazzo\breathing\amplitude_distributions"  # where to store

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

all_trials, all_breaths = load_datasets(
    datasets, file_format, fs_dataset=fs_dataset, assertions=False
)

# only first "stim" in each file
all_files = all_trials.xs(key=0, level="stims_index")

all_files.sort_index(inplace=True)  # sort indices

# %%
# add columns for checking file quality.
#
# will be added to future preprocessing versions.

new_cols = ["n_breath_segs_per_file", "file_length_s", "resp_rate_file_avg"]

assert not all([x in all_files.columns for x in new_cols])

n_breath_segs_per_file = all_breaths.end_s.groupby(by="audio_filename").agg(len)
file_length_s = all_breaths.end_s.groupby(by="audio_filename").agg(max)

all_files["n_breath_segs_per_file"] = n_breath_segs_per_file
all_files["file_length_s"] = file_length_s
all_files["resp_rate_file_avg"] = n_breath_segs_per_file / (2 * file_length_s)

# %%
# reject files

print(f"starting with {len(all_files)} files...")
all_files, all_breaths, rejections = reject_files(all_files, all_breaths)
print(f"left with {len(all_files)} files after assertions")


# %%

file = r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_010323_072427.99.cbin"

# file = r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_010323_072449.100.cbin"

aos = AudioObject.from_file(file, channel="all")

fig, axs = plt.subplots(nrows=len(aos), sharex=True)

for ao, ax in zip(aos, axs.ravel()):
    ax.plot(ao.get_x(), ao.audio)

axs[0].set(title=Path(file).name)
axs[-1].set(xlabel="time (s)")

# %%
# get files

# parent = Path(
#     r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\preLesion\spontaneous"
# )
parent = Path(
    r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous"
)

files = list(parent.glob("*.cbin"))

files = [
    r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_010323_072427.99.cbin",
    r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_010323_072452.101.cbin",
    #     r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_010323_072612.103.cbin",
    #     r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_010323_072706.105.cbin",
]

print(f"{len(files)} files.")

# %%
# plot traces or distrs for files

assert len(files) < 10, "you sure you wanna plot {} files?".format(len(files))

fig, axs = plt.subplots(
    nrows=len(files),
    sharex=True,
    sharey=True,
    # figsize=(6, 12),
)

for file, ax in zip(files, axs):
    ao = AudioObject.from_file(file, channel=0)

    # distrs
    ax, _ = plot_trace_amplitude_distribution(ao.audio, ax=ax, hist_bins=200)

    # traces
    # ax.plot(ao.get_x(), ao.audio)

    ax.set(title=Path(file).name)

fig.tight_layout()

# %%
# plot combined distr for all files

# really slow...

# print(str(parent))

# concat_traces = np.concatenate(
#     [AudioObject.from_file(file, channel=0).audio for file in files]
# )
# print(f"concat files have total length: {len(concat_traces)} samples")

# fig, ax = plt.subplots()
# ax, _ = plot_trace_amplitude_distribution(concat_traces, ax=ax, hist_bins=100)

# ax.set_title(f"{parent} ({len(files)} files)", wrap=True)

# %%
# try hartigan dip test

import diptest

# %%

fig, axs = plt.subplots(ncols=3, figsize=(11, 4))

for (bird, bird_df), ax in zip(all_files.groupby(by="birdname"), axs.ravel()):
    for dset, dataset_df in bird_df.groupby(by="dataset"):
        ax.hist(
            dataset_df.breath_zero_point,
            density=True,
            label=dset,
            # bins=np.linspace(-5000, 5000, 100),  # everything but outliers
            bins=np.linspace(-2000, 2000, 100),
            histtype="step",
        )

    ax.set(title=bird)

axs[0].set(ylabel="density")
axs[1].set(xlabel="zero point - dc offset")
axs[2].legend(loc="upper right")

fig.tight_layout()

# %%

all_files

# %%

fig, ax = plt.subplots()

ax.hist(all_files.dipstat, bins=200)

# %%

sorted_dips = all_files.sort_values(by="dipstat")

sorted_dips

# %%

for i in range(-10, -1):
    # for i in range(-50, -1, 5):
    # for i in range(0, 50, 5):
    fname = sorted_dips.iloc[i]["numpy_filename"]
    fs = sorted_dips.iloc[i]["fs"]

    trace = np.load(fname)[1, :]

    fig, axs = plt.subplots(nrows=2)

    ax = axs[0]
    for y in [0, -1]:
        ax.axhline(y=y, linewidth=0.5, color="k")
    ax.plot(np.arange(len(trace)) / fs, trace)

    ax.set(
        title=f"{Path(fname).stem} (dip={sorted_dips.iloc[i]['dipstat']:.03})",
        xlabel="time (s)",
        ylabel="amplitude",
    )

    ax = axs[1]
    plot_trace_amplitude_distribution(trace, ax=ax, hist_bins=200)

    fig.tight_layout()

# %%

all_files["n_breath_segs_per_file"] = all_breaths.groupby(
    by="audio_filename"
).end_s.agg("count")
all_files["file_length_s"] = all_breaths.groupby(by="audio_filename").end_s.agg("max")


all_files["resp_rate_file_avg"] = all_files["n_breath_segs_per_file"] / (
    2 * all_files["file_length_s"]
)

all_files


# %%
# nothing above 10Hz resp rate is real...

stupid_fast = all_files.loc[all_files["resp_rate_file_avg"] > 10]


for i, f in stupid_fast.iterrows():
    fig, ax = plt.subplots()
    fs = f["fs"]
    trace = np.load(f["numpy_filename"])[1, :]

    ax.plot(np.arange(len(trace)) / fs, trace)


# %%
# plot respiratory rate distr

fig, ax = plt.subplots()

ax.hist(all_files["resp_rate_file_avg"], bins=np.linspace(0, 10, 200))

ax.set(xlabel="respiratory rate, avg/file (Hz)", ylabel="file count")

# %%
# plot respiratory rate distr, grouped by dset

fig, ax = plt.subplots()

a = [
    (dataset, df.resp_rate_file_avg) for dataset, df in all_files.groupby(by="dataset")
]

datasets, distrs = zip(*a)

ax.hist(distrs, bins=np.linspace(0, 10, 100), label=datasets, histtype="step")

ax.set(
    xlabel="respiratory rate, avg/file (Hz)",
    ylabel="file count",
    title="respiratory rate by dataset",
)

for i in [2, 5]:
    ax.axvline(x=i, color="k")

ax.legend()

# %%


def plot_trace_and_dist(f, plot_breath_segs=False):

    fname = f["numpy_filename"]
    fs = f["fs"]
    trace = np.load(fname)[1, :]

    fig, axs = plt.subplots(nrows=2)

    ax = axs[0]
    for y in [0, -1]:
        ax.axhline(y=y, linewidth=0.5, color="k")
    ax.plot(np.arange(len(trace)) / fs, trace, c="k")

    if plot_breath_segs:
        f_breaths = all_breaths.loc[f.name]

        for type, breaths in f_breaths.groupby(by="type"):

            if type == "exp":
                c = "r"
            elif type == "insp":
                c = "b"
            else:
                raise ValueError(f"Unknown breath type: {type}")

            segs = breaths.apply(lambda x: [(x.start_s, 0), (x.end_s, 0)], axis=1)

            lc = LineCollection(segs, colors=c)

            ax.add_collection(lc)

    ax.set(
        title=f"{Path(fname).stem} (dip={f['dipstat']:.3f} | resp rate={f['resp_rate_file_avg']:.2f}Hz)",
        xlabel="time (s)",
        ylabel="amplitude",
    )

    ax = axs[1]
    plot_trace_amplitude_distribution(trace, ax=ax, hist_bins=200, kde_bins=200)

    fig.tight_layout()

    return fig, axs


# %%

slow = all_files.loc[
    (all_files.resp_rate_file_avg < 2)
    # & (all_files.dataset == "hvc_lesion_spontaneous")
]

# print(slow.dataset.value_counts())

i = -10

fig, axs = plot_trace_and_dist(slow.iloc[i], plot_breath_segs=True)

fig.tight_layout()

# %%

fast = all_files.loc[
    (all_files.resp_rate_file_avg > 5)
    # & (all_files.dataset == "hvc_lesion_spontaneous")
]

# print(slow.dataset.value_counts())
fast.sort_values(by="resp_rate_file_avg", inplace=True)

# %%

plt.close("all")

i = 6

fig, axs = plot_trace_and_dist(fast.iloc[i], plot_breath_segs=True)

fig.tight_layout()


# %%
# in range

in_range = all_files.loc[
    (all_files.resp_rate_file_avg >= 2)
    & (all_files.resp_rate_file_avg <= 5)
    & (all_files.dipstat >= 0.025)
]

in_range.sort_values(by="dipstat", inplace=True, ascending=False)

print(f"{len(in_range)} files in range")

# %%
plt.close("all")

# i = 0
for i in range(10):
    fig, axs = plot_trace_and_dist(in_range.iloc[i], plot_breath_segs=True)
    fig.tight_layout()

# %%

fig, ax = plt.subplots()

ax.hist(in_range.dipstat, bins=50)

# %%

plt.xkcd()  # hamish mode

fig, ax = plt.subplots()

ax.hist(all_files.dipstat, bins=50)
ax.axvline(0.025, color="k")
ax.set(
    title="bimodality",
    xlabel="hartigans' dip stat",
    ylabel="file count",
)


# %%
# plot bimodality distr, grouped by dset

fig, ax = plt.subplots()

a = [(dataset, df.dipstat) for dataset, df in all_files.groupby(by="dataset")]

datasets, distrs = zip(*a)

ax.hist(distrs, bins=np.linspace(0, 0.2, 100), label=datasets, histtype="step")

ax.set(
    title="bimodality by dataset",
    xlabel="hartigans' dip stat",
    ylabel="file count",
)

for i in [0.025]:
    ax.axvline(x=i, color="k")

ax.legend()


# %%
# expiration amplitudes

fig, ax = plt.subplots()

exp_amps = all_breaths.loc[all_breaths.type == "exp", "amplitude"].sort_values(
    ascending=False
)

ax.hist(exp_amps, bins=np.linspace(0, 30, 100))

ax.set(
    title="expiratory amplitude",
    xlabel="amplitude (normalized)",
    ylabel="count of expirations",
)

# %%
fname = r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION\pk19br8\postLesion\spontaneous\pk19br8_030323_134724.107.cbin"

# AudioObject.plot_file(fname)

np.load(all_files.loc[fname, "numpy_filename"])
