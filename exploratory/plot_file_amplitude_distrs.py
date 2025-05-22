# %%
# plot_file_amplitude_distrs.py
#
# batch plot amplitude histograms w/ spline fit + normalization points of interest

from multiprocessing import Pool
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.breath import (
    load_datasets,
)

from utils.breath.plot import plot_trace_amplitude_distribution

# %%
# dataset information

n_jobs = 7

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


def process_one_file(row):
    """
    multiprocessing helper function
    """
    pois = [np.nan] * 3

    try:
        stem = Path(row["numpy_filename"]).stem
        trace = np.load(row["numpy_filename"])[1, :]

        fig, ax = plt.subplots()
        ax, pois = plot_trace_amplitude_distribution(trace, bins=200, ax=ax)
        ax.set(title=stem)

        output_subfolder = Path(output_folder) / row["birdname"] / row["dataset"]
        os.makedirs(output_subfolder, exist_ok=True)

        fig.savefig(output_subfolder / f"{stem}.jpg")
        plt.close(fig)

        e = None

    except Exception as e:
        pass

    return [row["numpy_filename"], *pois, e]


# %%

if __name__ == "__main__":
    st = time.time()  # timer

    # %%
    # load dataset
    print(">> Loading datasets...")

    all_files, _ = load_datasets(datasets, file_format, fs_dataset=fs_dataset)

    all_files = all_files.xs(
        key=0, level="stims_index"
    )  # only first "stim" in each file
    all_files.sort_index(inplace=True)  # sort indices

    print(f"\tLoaded datasets! [Total elapsed time: {time.time() - st}s]\n")

    # %%
    # plot

    records = all_files.reset_index().to_dict("records")

    print("Starting pool...")
    with Pool(n_jobs) as pool:
        print(f"\tStarted pool! [Total elapsed time: {time.time() - st}s]\n")
        print("Plotting...")
        results = pool.map(
            process_one_file,
            records,
        )

    print(f"\tFinished plotting! [Total elapsed time: {time.time() - st}s]\n")

    # results = list(map(_multi_func, records))

    df = pd.DataFrame.from_records(
        results,
        columns=["numpy_filename", "insp_amp", "zero_amp", "exp_amp", "error"],
    ).set_index("numpy_filename")

    df_path = Path(output_folder) / "amplitude_pois.pickle"
    with open(df_path, "wb") as f:
        df.to_pickle(f)

    print(f"Saved POI df to: {df_path}")
    print(f"All done! [Total elapsed time: {time.time() - st}s]\n")
