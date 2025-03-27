# %%
# preprocess_breath_traces.py
#
# process wav files to filter, center, and normalize breath traces,
# saving as npy arrays

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import butter
from utils.audio import AudioObject

from utils.breath import get_kde_threshold

# %%
# paths

original_files = []

new_save_folder = Path("./data/cleaned_breath_traces")

# %%
# check unique names
# (dir structure is collapsed)

stems = list(map(lambda x: Path(x).stem, original_files))

assert len(original_files) == len(
    stems
), "Not all files in `original_files` have unique stems!"


# %%
# get trace, amplitude for all breaths
cleaned_filemap = {}

# filter
fs = 44100

filt_params = dict(N=2, Wn=50, btype="low", fs=fs)

b, a = butter(**filt_params)

for wav_path in original_files:

    ao = AudioObject.from_wav(wav_path, channels=1, b=b, a=a)
    breath = ao.audio_filt

    assert ao.fs == fs, "Wrong sample rate!"

    zero_point = get_kde_threshold(breath)

    #  map [biggest_insp, zero_point] --> [-1, 0]
    breath -= zero_point
    breath /= np.abs(breath.min())

    stem = Path(wav_path).stem
    np_path = new_save_folder.joinpath(f"{stem}.npy")

    cleaned_filemap[wav_path] = np_path
    np.save(np_path, breath)


log_lines = [
    f"# LOG ({new_save_folder})",
    f"",
    f"The following steps have been taken for each wav file:",
    f"",
    f"1. Butterworth filter. Params: {filt_params}",
    f"2. Center (using amplitude distribution)",
    f"3. Normalize to amplitude of deepest inspiration in file.",
    "",
    "Files have been mapped as follows:",
    "",
    *[f'"{wav}", "{np}"' for wav, np in cleaned_filemap.items()],
]

with open(new_save_folder.joinpath("LOG.md"), "w") as f:
    f.writelines([l + "\n" for l in log_lines])
