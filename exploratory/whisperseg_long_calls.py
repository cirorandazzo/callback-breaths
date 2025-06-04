# %%
# whisperseg_long_calls.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatreader import read_mat

from utils.callbacks import read_calls_from_mat
from utils.file import parse_birdname

# %load_ext autoreload
# %autoreload 1
# %aimport utils.callbacks

# %%

long_call_report = r".\hamish_long_calls.csv"

with open(long_call_report, "r") as f:
    data = f.readlines()


def process_line(line):
    # remove trailing newline
    cells = line.replace(",\n", "").split(",")

    # paths
    old_parent = "E:\\Data\\Ziggy3"
    path = Path(cells[0].replace(old_parent, ""))
    folder = str(path.parent)
    file = str(path.name)

    # call indices
    calls = [int(a) for a in cells[1:]]

    return folder, file, calls


data = pd.DataFrame.from_records(
    list(map(process_line, data)),
    columns=["folder", "file", "ii_long_calls"],
)

# matlab 1-index --> python 0-index
data["ii_long_calls"] = data["ii_long_calls"].map(lambda x: np.array(x) - 1)

data.loc[:, "bird"] = data["file"].apply(
    lambda x: (
        parse_birdname(
            str(Path(x).stem),
            birdname_regex=r"(?:[a-z]{1,2}[0-9]{1,2}){1,2}",
        )
    ),
)

data.set_index(keys=["bird", "folder", "file"], inplace=True)
data.sort_index(inplace=True)

data

# %%
# folders per bird
#
# "bird" pulled from filename; show all folders containing such a file

for bird, df_bird in data.groupby(by="bird"):
    print(bird)

    for f, n in dict(df_bird.reset_index().folder.value_counts()).items():
        print(f"\t- {f} ({n})")
        # print(f)

# %%
# look at good files
# NOTE: this is done in preprocessing/load_raw_files.py

# parent = Path(r"M:\public\from_egret\egret\eszter\data\air sac\MALES\noCapacitor")

# file_type = "cbin"

# good_folders = [
#     r"gr56bu23\postImplant\baseline",  # also incl. rd16gr95 files
#     r"gr92gr19\postCannula\muscimol\right HVC\baseline",
# ]

# files = (
#     pd.DataFrame.from_records(
#         [
#             (
#                 parse_birdname(
#                     str(file.stem), birdname_regex=r"(?:[a-z]{1,2}[0-9]{1,2}){1,2}"
#                 ),
#                 f"\\{folder}",
#                 f"{file.name}.not.mat",
#                 file,
#             )
#             for folder in good_folders
#             for file in (parent / folder).glob(f"*.{file_type}")
#         ],
#         columns=["bird", "folder", "file", "audio_file"],
#     )
#     .set_index(["bird", "folder", "file"])
#     .sort_index()
# )

# print(f"{len(files)} files found.")

# files

# %%
# store long calls

# %%

processed_parent = Path(r"M:\randazzo\breathing\processed\spontaneous_long_calls")


errors = pd.read_pickle(processed_parent / "spontaneous_long_calls-errors.pickle")
all_breaths = pd.read_pickle(
    processed_parent / "spontaneous_long_calls-all_breaths.pickle"
)
all_files = pd.read_pickle(processed_parent / "spontaneous_long_calls-all_files.pickle")

all_files

# %%
# match indices for `data` with `all_files` & `all_breaths`
#
# file stem (just name, without parent or extension)

# all files
all_files["notmat_name"] = all_files.index.to_series().map(
    lambda x: f"{Path(x[0]).name}.not.mat"
)
all_files = all_files.reset_index().set_index("notmat_name", verify_integrity=True)

# all breaths (just add notmat_name col for now)
all_breaths["notmat_name"] = all_breaths.index.to_series().map(
    lambda x: f"{Path(x[0]).name}.not.mat"
)

# data
data = data.reset_index().set_index("file")
data

all_files


# %%
# add long_calls cols to all_files

all_files["ii_long_calls"] = data["ii_long_calls"].reindex(
    all_files.index, fill_value=[]
)

all_files["n_long_calls"] = all_files["ii_long_calls"].apply(len)

all_files.ii_long_calls.value_counts()

# %%
# reset index and resave

all_files = all_files.reset_index().set_index(["audio_filename", "stims_index"])

pd.to_pickle(all_files, processed_parent / "spontaneous_long_calls-all_files.pickle")

# %%
# add long call designation to all_breaths

notmat_folder = Path(r"M:\whmehaff\ZiggySpontBirdsNotMats")

assert all(
    all_files.notmat_name.map(lambda x: (notmat_folder / x).exists())
), f"Missing >=1 notmat file in {notmat_folder}"

# %%
# read whisperseg .not.mats
#
# - concat into `all_calls` df, 1 row/call (Note: files without calls will not be represented)
# - add n_calls (per file) into all_files


def read_file(notmat_file, notmat_folder):

    notmat_data = read_mat(notmat_folder / notmat_file)
    n_calls = len(notmat_data["labels"])

    if n_calls == 0:
        df = None
    else:
        df = read_calls_from_mat(notmat_data, from_notmat=True)
        df["file"] = notmat_file
        df.index.name = "calls_index"
        df = df.reset_index().set_index(["file", "calls_index"])

    return df, n_calls


out = all_files.notmat_name.apply(
    read_file,
    args=[notmat_folder],
)

dfs, n_calls = zip(*out)

all_files["n_calls_whisperseg"] = pd.Series(data=n_calls, index=out.index)

all_calls = pd.concat(dfs).sort_index()
all_calls["is_long_call"] = False

del dfs

all_files

# %%
# transfer long_call designation

for idx, file in all_files.iterrows():

    fname = f"{Path(idx[0]).name}.not.mat"

    try:
        all_calls.loc[(fname, file.ii_long_calls), "is_long_call"] = True
    except KeyError:
        # print(f"No calls in file: {Path(idx[0]).name}")
        continue

assert (
    all_calls.is_long_call.sum() == all_files.n_long_calls.sum()
), f"Not every long call in `all_files` ({all_files.n_long_calls.sum()}) is listed in `all_calls` ({all_calls.is_long_call.sum()})."

all_calls

# %%
# save all_files, all_calls

pd.to_pickle(all_files, processed_parent / "spontaneous_long_calls-all_files.pickle")

pd.to_pickle(all_calls, processed_parent / "spontaneous_long_calls-all_calls.pickle")


# %%
# plot call durations

bins = np.linspace(
    all_calls.duration_s.min() * 1000,
    all_calls.duration_s.max() * 1000,
    70,
)

fig, ax = plt.subplots()

for lc, df in all_calls.groupby(by="is_long_call"):

    l = "long call" if lc else "other"
    l += f" ({len(df)})"

    ax.hist(
        df.duration_s * 1000,
        bins=bins,
        label=l,
        histtype="step",
        alpha=0.7,
    )

ax.set(title="call durations by type", xlabel="duration (ms)", ylabel="count")
ax.legend()

# %%
# mark breaths as long calls

all_breaths["assoc_call"] = None
all_breaths["is_call_ws"] = False
all_breaths["is_long_call_ws"] = False

padding = 0.005

empty = []
excess = []

for idx, call in all_calls.iterrows():

    ii_call_breath = (
        (all_breaths.notmat_name == idx[0])
        & (all_breaths.type == "exp")
        & (all_breaths.start_s - padding <= call.start_s)
        & (all_breaths.end_s + padding >= call.end_s)
    )

    # must map to exactly 1 expiration
    if sum(ii_call_breath) == 0:
        empty.append(idx)
        continue
    elif sum(ii_call_breath) > 1:
        excess.append(idx)
        continue
    else:  # exactly one
        assert sum(ii_call_breath) == 1
        ii_call_breath = ii_call_breath.idxmax()

    all_breaths.at[ii_call_breath, "assoc_call"] = idx
    all_breaths.at[ii_call_breath, "is_call_ws"] = True
    all_breaths.at[ii_call_breath, "is_long_call_ws"] = call.is_long_call

all_breaths


# %%
# save all_breaths with added info

pd.to_pickle(
    all_breaths, processed_parent / "spontaneous_long_calls-all_breaths.pickle"
)
