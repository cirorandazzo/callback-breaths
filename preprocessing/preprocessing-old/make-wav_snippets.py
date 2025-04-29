# %%
# make_wav_snippets.py
#
# save snippets of audio + breath as .wav files for breaths in df all_breaths

from pathlib import Path
import pickle
import wave

import numpy as np
import pandas as pd

from scipy.signal import butter
from utils.audio import AudioObject

from utils.breath import get_kde_threshold

# %%
# paths

save_folder = Path("./data/wav_snippets")

with open( save_folder.joinpath("all_breaths.pickle"), "rb") as f:
    all_breaths = pickle.load(f)

# %%
# check unique names
# (dir structure is collapsed, so don't want 2 identically named files in 2 diff directories)

files = list(all_breaths.index.get_level_values("wav_filename").unique())

stems = list(map(lambda x: Path(x).stem, files))

assert len(files) == len(
    stems
), "Not all files in `original_files` have unique stems!"


# %%
# save traces for all breaths

saved_files = {}

def save_row(trial, save_data, save_folder, pre_frames, post_frames, fs):

    assert save_data.dtype == np.int16, "save_data must be int16!"
    n_channels, data_length = save_data.shape

    # prep savepath
    wav_path, calls_index = trial.name

    name = f"{Path(wav_path).stem}-seg{calls_index:03}.wav"
    save_path = Path(save_folder).joinpath(name)

    # get audio snippet
    st, en = (trial[["start_s", "end_s"]] * fs).astype(int)

    st = max(0, st - pre_frames)
    en = min(data_length, en + post_frames)


    with wave.open(str(save_path), "wb") as f:
        f.setnchannels(n_channels)  # 2 channels
        f.setsampwidth(2)  # 2 bytes per sample (ie, int16)
        f.setframerate(fs)

        to_save = save_data[:, st:en].T.tobytes()

        f.writeframes(to_save)

    return name


# prep filter
fs = 44100
filt_params = dict(N=2, Wn=1000, btype="low", fs=fs)
b, a = butter(**filt_params)

for wav_path, calls in all_breaths.groupby(level="wav_filename"):

    # load channels
    audio = AudioObject.from_wav(wav_path, channel=0).audio

    ao = AudioObject.from_wav(wav_path, channel=1, b=b, a=a)
    breath = ao.audio_filt

    assert ao.fs == fs, "Wrong sample rate!"

    # zero point center breath
    zero_point = get_kde_threshold(breath)
    breath -= zero_point

    # saving
    # audio hasn't been changed, should already be int16
    breath = breath.astype(np.int16)
    file_data = np.stack([audio, breath])

    saved_files[wav_path] = list(
        calls.apply(
            save_row,
            axis=1,
            save_data=file_data,
            save_folder=save_folder,
            pre_frames=int(0.5 * fs),
            post_frames=int(0.5 * fs),
            fs=fs,
        )
    )


log_lines = [
    f"# LOG ({save_folder})",
    f"",
    f"The following steps have been taken for each wav file:",
    f"",
    f"Audio",
    f"(n/a)",
    f"",
    f"Breath",
    f"1. Butterworth filter. Params: {filt_params}",
    f"2. Center (using amplitude distribution)",
    f"",
    f"Files:",
]

for i, (original, new) in enumerate(saved_files.items()):
    log_lines.append(f"{i:2}. {original}")

    for j, a in enumerate(new):
        log_lines.append(f"\t{j}. {a}")


with open(save_folder.joinpath("LOG.md"), "w") as f:
    f.writelines([l + "\n" for l in log_lines])
