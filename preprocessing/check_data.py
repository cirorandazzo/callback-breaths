# %%
# check_data.py
#
# brief script for checking sizes/counts of data

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# %%
# load & report

datasets = ["callback", "spontaneous", "hvc_lesion_callback", "hvc_lesion_spontaneous"]

fs_dataset = {
    "callback": 44100,
    "spontaneous": 32000,
    "hvc_lesion_callback": 44100,
    "hvc_lesion_spontaneous": 32000,
}

file_format = r"M:\randazzo\breathing\processed\{dataset}\{dataset}-{file}.pickle"

to_load = ["all_files", "all_breaths", "errors"]

for dataset in datasets:
    data = {}

    for df_name in to_load:
        with open(file_format.format(dataset=dataset, file=df_name), "rb") as f:
            data[df_name] = pickle.load(f)

    report_str = "{dataset}: {n_breaths} breaths across {n_trials} trials from {n_files} files. {n_errors} files errored out."

    print(
        report_str.format(
            dataset=dataset,
            n_breaths=len(data["all_breaths"]),
            n_trials=len(data["all_files"]),
            n_files=len(data["all_files"]["numpy_filename"].unique()),
            n_errors=len(data["errors"]),
        )
    )
