# %%
# spontaneous_long_calls.py
#
# does my breath-based thresholding reliably pull out long calls?
# verify with audio data.
#
# check the previous datasets to see if there's usable audio & notmats.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatreader import read_mat

from utils.audio import AudioObject
from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality
from utils.breath.plot import plot_histogram_by_bird_type_dset
from utils.callbacks import read_calls_from_mat
from utils.file import parse_birdname

# %load_ext autoreload
# %autoreload 1
# %aimport utils.callbacks

# %%
# lesion birds

notsmat = Path(r"M:\randazzo\FromHamish\hvc_lesion_notmats")

files = notsmat.glob("**/*.not.mat")
files = [f for f in files]

stems = list(map(lambda x: str(x.stem).split(".")[0], files))

print(f"{len(files)} not mats. {len(np.unique(stems))} unique files.")

# %%
# lookup notmat path from fname

notmats = {str(x.name): str(x) for x in files}
notmats

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
    "dipstat",
]

all_files = all_trials.xs(key=0, level="stims_index")[file_cols]

# %%
# assign notmat files based on fname

# notmat name should be {fname}.not.mat
all_files["notmat_path"] = all_files.index.to_series().map(
    lambda x: f"{Path(x).name}.not.mat"
)

# look that up in list of notmats
all_files["notmat_path"] = all_files["notmat_path"].map(notmats)

print(all_files)
print("{} files had no notmat.".format(sum(all_files["notmat_path"].isnull())))


# %%
# report on files missing notmats

summary = all_files.groupby(by=["dataset", "birdname"]).agg(
    n_no_notmat=pd.NamedAgg("notmat_path", lambda x: sum(x.isnull())),
    n_files=pd.NamedAgg("notmat_path", len),
)

summary["pct_with_notmat"] = 1 - (summary.n_no_notmat / summary.n_files)

summary

# %%
# read whisperseg .not.mats
#
# - concat into `all_calls` df, 1 row/call (Note: files without calls will not be represented)
# - add n_calls (per file) into all_files


def read_file(notmat_path):

    notmat_path = str(notmat_path)

    if Path(notmat_path).exists():
        notmat_data = read_mat(notmat_path)
        n_calls = len(notmat_data["labels"])
    else:
        return None, np.nan

    if n_calls == 0:
        df = None
    else:
        df = read_calls_from_mat(notmat_data, from_notmat=True, notmat_lookup_dict={})
        df["file"] = notmat_path
        df["birdname"] = parse_birdname(notmat_path)
        df.index.name = "calls_index"
        df = df.reset_index().set_index(["file", "calls_index"])

    return df, n_calls


out = all_files.notmat_path.map(read_file)
dfs, n_calls = zip(*list(out))

all_files["n_calls_whisperseg"] = pd.Series(data=n_calls, index=out.index)
all_calls = pd.concat(dfs).sort_index()
del dfs

all_calls

# %%
# plot # calls / file

fig, ax = plt.subplots()

ax.hist(
    all_files.n_calls_whisperseg,
    bins=np.arange(all_files.n_calls_whisperseg.max()),
)

ax.set(
    title="n calls per file (whisperseg)",
    xlabel="n calls",
    ylabel="n files",
    # xlim=[-.5,20],
)

print(all_files.n_calls_whisperseg.value_counts(dropna=False).sort_index())

# %%
# plot diff WS labels by bird, dataset

reindexed = all_files.set_index("notmat_path")

all_calls["dataset"] = all_calls.index.map(lambda x: reindexed.loc[x[0], "dataset"])

del reindexed

plots = plot_histogram_by_bird_type_dset(
    all_calls,
    "duration_s",
    bins=np.linspace(0, 0.3, 70),
    # bins=np.linspace(0, np.ceil(all_calls.duration_s.max()), 70),
)

for _, (fig, axs) in plots.items():
    for ax in axs:
        ax.legend()
        ax.set(ylabel="# calls")

    fig.tight_layout()


# %%
# reject song

a = all_calls.groupby(by="file").time_from_prev_onset_s.agg([len, "mean"])

a.sort_values(by="len", ascending=False)

# %%
# plot calls/file & mean ici for all files

fig, axs = plt.subplots(ncols=2, figsize=(13.33, 7.5))
fig.suptitle("whisperseg summary stats")

ax = axs[0]
ax.hist(
    a["len"],
    bins=70,
)
ax.set(xlabel="# whisperseg calls per file", title="calls per file")

ax = axs[1]
ax.hist(
    a["mean"],
    bins=70,
)
ax.set(xlabel="mean ici per file", title="mean ici per file")

fig.tight_layout()

# %%
# reject song
#
# UPSHOT: drop files with any gaps <=1s


gap_threshold = 1  # s; shorter than this presumed to be song.


song_report = all_calls.groupby(by="file").time_from_prev_onset_s.agg(
    n_short_gaps=lambda x: np.sum(x <= gap_threshold),
    n_syls=len,
)

song_report

# %%
# show # of `short` gaps


def plot_hist(x, ax=None, **set_kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(
        x,
        bins=np.arange(x.max()),
    )
    ax.set(xlabel=x.name, **set_kwargs)

    return ax


fig, axs = plt.subplots(ncols=3, figsize=(13.33, 4))

plot_hist(
    song_report.n_short_gaps,
    title=f"# gaps <= {gap_threshold}s in all syllables",
    ax=axs[0],
)

x = song_report.loc[song_report.n_short_gaps == 0, "n_syls"]
plot_hist(
    x,
    title=f"# syls in files with no `short` gaps\n(n={len(x)}/{len(song_report)} files)",
    ax=axs[1],
)

x = song_report.loc[song_report.n_short_gaps > 0, "n_syls"]
plot_hist(
    x,
    title=f"# syls in files with >1 `short` gap\n(n={len(x)}/{len(song_report)} files)",
    xticks=np.arange(0, song_report.n_syls.max(), 5),
    ax=axs[2],
)


# %%
# drop putative song files

good_files = song_report.index[song_report.n_short_gaps == 0]

calls_only = all_calls[all_calls.index.get_level_values("file").isin(good_files)]

# remove segmentation failures.
# no real call is longer than this (500ms is overkill, probably would be ok w ~200ms)
calls_only = calls_only.loc[calls_only.duration_s <= 0.5]

calls_only

# %%
# report on call-only duration

fig, ax = plt.subplots()

x = calls_only.duration_s
ax.hist(x, bins=100)
ax.set(
    xlabel="duration (s)",
    title=f"call duration - call-only files\n(n={len(x) } calls/{calls_only.index.get_level_values('file').nunique()} files)",
)

# %%
# report on call-only duration by dataset/birdname

fig, ax = plt.subplots()

bins = np.linspace(0, calls_only.duration_s.max(), 100)

gb_levels = ["dataset", "birdname"]
# gb_levels = ["dataset"]

for key, x in calls_only.groupby(by=gb_levels).duration_s:
    if len(x) < 10:
        print(f"Skipping {key} -- only {len(x)} calls.")
        continue

    ax.hist(
        x,
        bins=bins,
        label="-".join(key) + f" (n={len(x)})",
        histtype="step",
        # density=True,
    )

ax.set(
    xlabel="duration (s)",
    title=f"call duration - call-only files\n(n={len(x) } calls/{calls_only.index.get_level_values('file').nunique()} files)",
    # ylabel="density",
    ylabel="count",
)

ax.legend(loc=(0.5, 0.5))

# %%

calls_only


# %%
# reindex calls_only

calls_only["audio_name"] = calls_only.index.to_frame().file.apply(
    lambda x: str(Path(x).name).replace(".not.mat", "")
)

calls_only = calls_only.reset_index().set_index(["audio_name", "calls_index"])

# %%
# reindex all_breaths

all_breaths["audio_name"] = all_breaths.index.to_frame().audio_filename.apply(
    lambda x: str(Path(x).name)
)

all_breaths = all_breaths.reset_index().set_index(["audio_name", "calls_index"])
all_breaths.index.rename(
    ["audio_name", "breaths_index"], inplace=True
)  # rename for clarity - distinct from calls_only index level (where calls_index means audio signal)


# %%
# tie each call to a breath.
#
# nearly every call is wholly contained within a breath


def get_breath(call, buffer_s=0):
    file_stem, _ = call.name

    file_breaths = all_breaths.xs(key=file_stem, level="audio_name")

    breath = file_breaths.loc[
        ((file_breaths.start_s - buffer_s) <= call.start_s)
        & ((file_breaths.end_s + buffer_s) >= call.end_s)
    ]

    return list(breath.index)


calls_only["breaths"] = calls_only.apply(get_breath, axis=1)

calls_only["breaths"].apply(len).value_counts()

# %%
# TODO: figure out when audio call isn't fully encompassed by breath

calls_only.loc[calls_only["breaths"].apply(len) == 0]
