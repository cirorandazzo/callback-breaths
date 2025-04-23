# %%
# make_spontaneous_dfs.py

import glob
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize

from scipy.signal import butter

import hdbscan

from utils.audio import AudioObject
from utils.breath import (
    get_kde_threshold,
    loc_relative,
    make_notmat_vars,
    segment_breaths,
)
from utils.callbacks import call_mat_stim_trial_loader
from utils.file import parse_birdname

# %load_ext autoreload
# %autoreload 1
# %aimport utils.audio

# %%
# set directories

parent = Path(r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION")

pre_lesion_spontaneous = [
    parent.joinpath(bird, "preLesion", "spontaneous", "*.cbin")
    for bird in ["rd57rd97", "pk19br8", "rd99rd72"]
]

# %%
# get filelist

paths = pre_lesion_spontaneous

files = [file for path in paths for file in sorted(glob.glob(os.path.normpath(path)))]

print(f"Found {len(files)} files!")

# %%

trace_folder = Path("./data/spontaneous-pre_lesion")

assert len(files) == len(set(files)), "Duplicate file names found!"
assert len(files) > 0, "No files found!"

all_files = []
all_breaths = []
errors = {}

# filter for breath segmentation
fs = 32000
bw = dict(N=2, Wn=50, btype="low")
b, val = butter(**bw, fs=fs)


def get_trial_amplitude(x, breath):

    s, e = (np.array([x.start_s, x.end_s]) * fs).astype(int)

    if x.type == "exp":
        amp = max(breath[s:e])
    elif x.type == "insp":
        amp = min(breath[s:e])
    else:
        amp = np.nan

    return amp


for i, file in enumerate(files):
    file = Path(file)

    print(f"{i:05}/{len(files)}, Processing {file.name}...")

    try:
        # load
        ao_breath, ao_audio = AudioObject.from_cbin(file, channel="all")
        assert ao_breath.fs == fs, f"fs mismatch: {ao_breath.fs} != {fs}"

        # filter
        ao_audio.filtfilt_butter_default()
        ao_breath.filtfilt(b, val)

        audio = ao_audio.audio_filt
        breath = ao_breath.audio_filt

        assert len(audio) == len(
            breath
        ), f"audio and breath lengths don't match: {len(audio)} != {len(breath)}"

        # center & normalize breath
        breath_zero_point = get_kde_threshold(breath)
        breath -= breath_zero_point
        breath = breath / abs(np.min(breath))

        # create callback-like dfs

        # segment breaths
        #   - do_filter refilters at 50Hz (better for coarse structure)
        #   - (threshold = lambda x: 0) reflects pre-centering (use 0 as breath center)
        exps, insps = segment_breaths(
            breath, ao_breath.fs, threshold=lambda x: 0, do_filter=True
        )

        # make breaths into .notmat structure
        onsets, offsets, labels = make_notmat_vars(
            exps, insps, len(breath), exp_label="exp", insp_label="insp"
        )
        onsets = onsets / fs
        offsets = offsets / fs

        stims = np.array([0.0])  # treat each spontaneous file as a single trial

        data = {
            "onsets": np.concatenate([onsets, stims]) * 1000,
            "offsets": np.concatenate([offsets, stims + 0.1]) * 1000,
            "labels": np.concatenate([labels, ["Stimulus"] * len(stims)]),
        }

        calls, stim_trials, rejected_trials, file_info, call_types = (
            call_mat_stim_trial_loader(
                file=None,
                data=data,
                from_notmat=True,
                verbose=False,
                acceptable_call_labels=["Stimulus", "exp", "insp"],
            )
        )

        # drop meaningless columns (from callbacks)
        stim_trials.drop(
            columns=["calls_index", "latency_s", "stim_duration_s"], inplace=True
        )

        calls.drop(columns=["ici"], inplace=True)

        # drop dummy stimulus from calls
        calls = calls.loc[calls["type"] != "Stimulus"]

        # add amplitude/putative call
        calls["amplitude"] = calls.apply(get_trial_amplitude, axis=1, breath=breath)
        call_exp = (calls["amplitude"] > 1.1) & (calls["type"] == "exp")

        stim_trials["n_putative_calls"] = call_exp.sum()

        # filtered amplitude, but not centered/normalized
        stim_trials["breath_amplitude_min"] = ao_breath.audio_filt.min()
        stim_trials["breath_amplitude_max"] = ao_breath.audio_filt.max()
        stim_trials["breath_zero_point"] = breath_zero_point

        # add file info to dfs
        # .... it's not a wav file, but keeping 'wav_filename' for consistency
        calls["wav_filename"] = str(file)
        stim_trials["wav_filename"] = str(file)

        stim_trials["trial_end_s"] = len(breath) / fs  # replace np.inf

        # add birdname
        birdname = parse_birdname(file.name)
        calls["birdname"] = birdname
        stim_trials["birdname"] = birdname

        # save data
        all_files.append(
            stim_trials.reset_index().set_index(["wav_filename", "stims_index"])
        )

        all_breaths.append(
            calls.reset_index().set_index(["wav_filename", "calls_index"])
        )

        # save processed files as npy
        np_file = trace_folder.joinpath(file.stem + ".npy")
        np.save(np_file, np.vstack([audio, breath]))

        print(f"\tSuccess!")

    except Exception as e:
        errors[str(file)] = e
        print(f"\tError: {e}")
        print(f"\tContinuing...")
        pass

all_files = pd.concat(all_files).sort_index()
all_breaths = pd.concat(all_breaths).sort_index()

# pickle errors
# pickle all_breaths & all_trials after adding columns below!

with open(trace_folder.joinpath("errors.pickle"), "wb") as f:
    pickle.dump(errors, f)


log_lines = [
    "# LOG",
    "",
    "Stored in this folder are numpy files containing processed spontaneous data. Files are named according to the original cbin file names, and contain a 2xN array:",
    "",
    "- row 0: audio (default AudioObject filter)",
    "- row 1: breath (500Hz filtered, centered, normalized)",
    "",
    "The following data are also stored:",
    "",
    "- all_files: 1 row per spontaneous file",
    "- all_breaths: 1 row per breath, merged across all files",
    "- errors: 1 row per file that errored out" "",
    "Parameters:" f"- fs: {fs}",
    f"- butterworth filter: {bw}",
]

with open(trace_folder.joinpath("log.txt"), "w") as f:
    for line in log_lines:
        f.write(line + "\n")

print(f"{len(all_files)} files processed successfully! See `all_files`.")
print(f"{len(errors)} files errored out. See `errors`.")

# %%

all_files["date"] = all_files.apply(
    lambda x: Path(x.name[0]).stem.split("_")[1],
    axis=1,
)

# %%
# add putative call

def check_call(trial, threshold, all_breaths):

    if trial["type"] == "exp":
        amplitude = trial["amplitude"]
    elif trial["type"] == "insp":
        amplitude = loc_relative(
            *trial.name,
            all_breaths,
            field="amplitude",
            i=1,
            default=np.nan,
        )
    else:
        raise ValueError(f"`{trial.type}` is an unknown breath type. Must be `insp` or `exp`.")

    return ~(np.isnan(amplitude)) and (amplitude > threshold)

all_breaths["putative_call"] = all_breaths.apply(
    check_call,
    axis=1,
    threshold=1.1,
    all_breaths=all_breaths,
)

# %%
# tests for putative call

# call expirations should have amplitude > threshold
ii_call_exps = (all_breaths.type == "exp") & (all_breaths.putative_call)
assert all(all_breaths.loc[ii_call_exps, "amplitude"] > 1.1), "These should all be call exps!"

# noncall expirations should have amplitude <= threshold
ii_non_call_exps = (all_breaths.type == "exp") & (~all_breaths.putative_call)
assert all(all_breaths.loc[ii_non_call_exps, "amplitude"] <= 1.1), "These should all be non-call exps!"

# all insps should match next expiration (or have putative_call = False if next exp DNE)
ii_insps = (all_breaths.type == "insp")
assert all(
    all_breaths.loc[ii_insps].apply(
        lambda x: (
            (x["putative_call"])
            == (loc_relative(*x.name, all_breaths, "putative_call", 1, False))
        ),
        axis=1,
    )
), "Insp should match next exp, or have putative_call = False if next exp DNE"

print("All putative call tests passed!")

# %%
# all_breaths: save & report

save_path_all_breaths = trace_folder.joinpath("all_breaths.pickle")
all_breaths.to_pickle(save_path_all_breaths)

print(f"Saved all_breaths to: {save_path_all_breaths}")

all_breaths

# %%
# all_trials: save & report

save_path_all_files = trace_folder.joinpath("all_files.pickle")
all_files.to_pickle(save_path_all_files)

print(f"Saved all_files to: {save_path_all_files}")

all_files

# %%

l = len([a for a,b in all_files.groupby(by=["birdname", "date"])])
print(f"{l} unique birds/dates found")


hist_kwargs = dict(
    bins=100,
    alpha=0.5,
)

fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True, figsize=[25.6, 13.35])

for ax, ((birdname, date), df_bird_date) in zip(axs.ravel(), all_files.groupby(by=["birdname", "date"])):

    ax.set(title=f"{birdname} {date} (n={len(df_bird_date)})")

    for val in ["amplitude_min", "amplitude_max", "zero_point"]:
        ax.hist(
            df_bird_date[f"breath_{val}"],
            **hist_kwargs,
            label=val,
        )

axs.ravel()[0].legend()
axs.ravel()[0].set(ylabel="count")

fig.suptitle("Breath amplitude distribution by day")
fig.tight_layout()
