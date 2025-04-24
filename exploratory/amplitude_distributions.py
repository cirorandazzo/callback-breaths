# %%
# amplitude_distributions.py
#
# plot & fit breath amplitude distribution for many files; important step in determining zero point algorithm

import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter

import matplotlib.pyplot as plt

from utils.audio import AudioObject
from utils.breath import (
    plot_amplitude_dist,
    fit_breath_distribution,
)
from utils.file import parse_birdname

# %load_ext autoreload
# %autoreload 1
# %aimport utils.breath

# %%
# get filelist

paths = [
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd99rd72/preLesion/callbacks/rand/230215/*-B*.wav",
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/pk19br8/preLesion/callback/rand/**/*-B*.wav",
    # r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd56/preLesion/callbacks/male_230117/*-B*.wav",
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd57rd97/preLesion/callbacks/male_230117/*-B*.wav",
]

# `*-B*` excludes "-PostBlock"

# get all files matching above paths
files = [file for path in paths for file in sorted(glob.glob(os.path.normpath(path)))]

assert len(files) != 0, "No files found!"

print("Files: ")
for i, f in enumerate(files):
    print(f"{i}. {os.path.split(f)[1]}")


# %%
# do amplitude KDE

do_plots = False

fs = 44100
b, a = butter(N=2, Wn=50, btype="low", fs=fs)

plot_kwargs_distr_marker = {
    "marker": "+",
    "s": 100,
    "linewidth": 2,
    "zorder": 3,
}

figure_save_folder = "./data/distributions"

records = []

for f in files:

    basename = os.path.splitext(os.path.basename(f))[0]

    try:
        birdname = parse_birdname(basename)
    except TypeError:
        birdname = "default"

    # load audio
    channels = AudioObject.from_wav(
        f, channels="all", channel_names=["audio", "breathing", "trigger"]
    )

    assert fs == channels[1].fs, "Wrong sample rate!"

    channels[1].filtfilt(b, a)  # filter breathing
    breath = channels[1].audio_filt

    x_dist, dist_kde, trough_ii, peaks_ii = fit_breath_distribution(breath)

    peaks = dist_kde[peaks_ii]
    threshold = dist_kde[trough_ii]

    entry = {
        "file": f,
        "birdname": birdname,
        "x_dist": x_dist,
        "dist_kde": dist_kde,
        "threshold": threshold,
        "insp_peak": peaks[0],
        "exp_peak": peaks[1],
        "trough_ii": trough_ii,
        "peaks_ii": peaks_ii,
    }

    records.append(entry)

    if do_plots:
        fig, ax = plt.subplots()
        bird_folder = os.path.join(figure_save_folder, birdname)
        os.makedirs(bird_folder, exist_ok=True)

        plot_amplitude_dist(breath, ax, median_multiples=None, percentiles=None)

        ax.plot(x_dist, dist_kde, color="k")

        points = peaks_ii + [trough_ii]

        ax.scatter(  # mark trough between those peaks
            x_dist[points],
            dist_kde[points],
            color="r",
            label="peaks & threshold",
            **plot_kwargs_distr_marker,
        )

        ax.vlines(  # add vertical lines from points to x axis
            x_dist[points],
            ymin=0,
            ymax=dist_kde[points],
            color="r",
            linewidth=1,
            linestyle="--",
        )

        ax.set(
            title=basename,
            xlabel="breath amplitude",
            ylabel="density",
        )
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(bird_folder, f"{basename}.jpg"), dpi=300)

        plt.close(fig)

df_kde = pd.DataFrame.from_records(records)

df_kde

# %%
# pickle distributions/thresholds

with open(os.path.join(figure_save_folder, "distributions.pickle"), "wb") as f:
    pickle.dump(df_kde, f)
