# %%
# long_calls.py
#
# thresholding and analyzing long calls

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.audio import AudioObject
from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality
from utils.breath.traces import cut_segment

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

all_trials, all_breaths = load_datasets(
    datasets, file_format, fs_dataset=fs_dataset, assertions=False
)
all_trials, all_breaths = TEMP_assert_file_quality(all_trials, all_breaths)

file_cols = [
    "birdname",
    "dataset",
    "numpy_filename",
    "fs",
    "n_breath_segs_per_file",
    "file_length_s",
    "resp_rate_file_avg",
    "breath_zero_point",
]

all_files = all_trials.xs(key=0, level="stims_index")[file_cols]

# cut to only this breath type
all_breaths = all_breaths.loc[all_breaths.type == "exp"]

# sort indices
all_breaths.sort_index(inplace=True)
all_trials.sort_index(inplace=True)

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

# %%
# try threshold: duration above amplitude
#
# require at least (duration) seconds are above (amplitude; normalized)
# note: not necessarily contiguous
#
# can also reject many breaths from the start, since they must be longer than this
#
# intentionally set strictly

amplitude_threshold = 8
duration_threshold = 0.1

thresholded = all_breaths.loc[
    (all_breaths.duration_s >= duration_threshold)
    & (all_breaths.amplitude >= amplitude_threshold)
]

# some reports

# number of expirations
print(
    "{n_exps_thr} / {n_exps} expirations have duration >= {duration_threshold}s & amplitude >= {amplitude_threshold}.".format(
        n_exps_thr =len(thresholded),
        n_exps=len(all_breaths),
        duration_threshold = duration_threshold,
        amplitude_threshold = amplitude_threshold,
    )
)

# number of files
print(
    "{n_long_call_files} / {n_total_files} files have >=1 such expiration.".format(
        n_long_call_files=thresholded.numpy_filename.nunique(),
        n_total_files=all_breaths.numpy_filename.nunique(),
    )
)

# breaths by bird
print("Long call expirations by bird:")
for k, v in thresholded.birdname.value_counts().items():
    print(f"\t- {k}\t{v}")

# %%
# check for long calls
#
# 10min for callback + spontaneous (~6k files after initial rejections)

n_files = len(thresholded.numpy_filename.unique())
audio_row, breath_row = 0, 1

for i, (file, df_file) in enumerate(thresholded.groupby(by="numpy_filename")):
    fstem = Path(file).stem
    print(f"File {i}/{n_files} started.")
    print(f"\t{fstem}")

    # get fs
    assert df_file.fs.nunique() == 1, f"Diff fs found for diff breaths in file: {file}"
    fs = df_file.fs.unique()[0]

    # min number of thresholded samples to be a long call
    min_samples = int(fs * duration_threshold)

    # load data
    data = np.load(file)

    audio = data[audio_row, :]
    breath = data[breath_row, :]

    def check_long_call(row):
        segment = cut_segment(
            breath,
            row,
            fs,
            interpolate_length=None,
            pad_frames=[0, 0],
        )

        return sum(segment >= amplitude_threshold) >= min_samples

    thresholded.loc[df_file.index, "is_long_call"] = df_file.apply(
        check_long_call,
        axis=1,
    )

# assert bool to ensure ~ operator works as expected
thresholded.is_long_call = thresholded.is_long_call.astype(bool)

# %%
# long_calls_by_file = thresholded.groupby(by="numpy_filename").is_long_call.agg("sum")
all_files["n_long_calls"] = thresholded.groupby(by="audio_filename").is_long_call.agg(
    "sum"
)
all_files.loc[all_files.n_long_calls.isna(), "n_long_calls"] = 0

all_files["n_long_calls"] = all_files["n_long_calls"].astype(int)

all_files.n_long_calls.sort_values(ascending=False)

# %%
# n long calls per file

fig, ax = plt.subplots()
ax.hist(all_files.n_long_calls, bins=np.arange(-0.5, 31.5, 1))
ax.set(
    xlabel="no. long calls / file",
    ylabel="file count",
    title="long calls by file",
    xlim=[-1.5, 31.5],
)

# %%
#

fig, ax = plt.subplots()
for dset, dataset_df in all_files.groupby(by="dataset"):
    ax.hist(
        dataset_df.n_long_calls,
        density=True,
        label=dset,
        # bins=np.linspace(-5000, 5000, 100),  # everything but outliers
        bins=np.arange(-0.5, 31.5, 1),
        histtype="step",
    )

ax.set(
    xlabel="no. long calls / file",
    ylabel="file count",
    title="long calls by file (per dataset)",
    xlim=[-1.5, 31.5],
)

ax.legend()

# %%
# some useful plots

# pick a file
file = thresholded.index.get_level_values(0).unique()[17]


# load data
df_file = thresholded.xs(key=file, level=0)
data = np.load(df_file.numpy_filename.iloc[0])

audio = data[audio_row, :]
breath = data[breath_row, :]


def plot_seg(row, ax, align=None, c=None):
    segment = cut_segment(
        breath,
        row,
        fs,
        interpolate_length=None,
        pad_frames=[0, 0],
    )

    x = np.arange(len(segment)) / row.fs

    if align is not None:
        x += row[align]

    ax.plot(x, segment, c=c)


def plot_long_call_trace(ax, df_file, fs, breath, plot_seg):
    # plot trace
    ax.plot(np.arange(len(breath)) / fs, breath, c="k", linewidth=0.5)

    # highlight considered traces
    df_file.loc[df_file.is_long_call].apply(
        plot_seg, axis=1, args=[ax, "start_s", "green"]
    )
    df_file.loc[~df_file.is_long_call].apply(
        plot_seg, axis=1, args=[ax, "start_s", "red"]
    )

    ax.set(ylabel="breath amplitude (norm)")


# %%
# plot breath trace for this file

fig, ax = plt.subplots(figsize=(20, 15))
plot_long_call_trace(ax, df_file, fs, breath, plot_seg)

# %%
# plot spectrogram & breath trace for this file
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(20, 15))
axs[0].set(title=fstem, xlabel=None)
axs[1].set(xlabel="time (s)")

# plot breath
ax = axs[0]
ao = AudioObject(audio, fs)
ao.filtfilt_butter_default()
ao.plot_spectrogram(ax=ax, cmap="jet", vmin=0.7)

# plot breath trace (putative long calls highlighted)
ax = axs[1]
plot_long_call_trace(ax, df_file, fs, breath, plot_seg)

# %%
# plot 2 panel: long calls vs. non-long calls

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
ax = axs[0]
df_file.loc[df_file.is_long_call].apply(plot_seg, axis=1, args=[ax])
ax.set(title="long call")

ax = axs[1]
df_file.loc[~df_file.is_long_call].apply(plot_seg, axis=1, args=[ax])
ax.set(title="not long call")
