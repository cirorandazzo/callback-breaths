# %%
# load_raw_files.py

import glob
import time
from utils.breath.preprocess import preprocess_files

if __name__ == "__main__":
    """
    NOTE: multiprocessing can be finicky. It doesn't work so well in notebook mode and usually needs to be run in `if __name__ == "__main__"` block to protect against recursively multithreading
    """

    # %%
    # load a set of test files - cbin

    cbin_files = [file for file in glob.glob("./data/test_input/cbin/*.cbin")]
    print(f"Processing {len(cbin_files)} cbin files...")

    st = time.time()

    preprocess_files(
        files=cbin_files,
        output_folder="./data/test_output_multiproc/CBIN",
        prefix="TEST_CBIN",
        fs=32000,
        channel_map={"audio": 0, "breath": 1},
        has_stims=False,
        stim_length=0.1,
        output_exist_ok=True,  # False,
        n_jobs=4,
    )

    print(f"Finished! ({time.time() - st}s)")

    # %%
    # load a set of test files - wav

    wav_files = [file for file in glob.glob("./data/test_input/wav/*-B*.wav")]
    print(f"Processing {len (wav_files)} wav files...")

    st = time.time()

    preprocess_files(
        files=wav_files,
        output_folder="./data/test_output_multiproc/WAV",
        prefix="TEST_WAV",
        fs=44100,
        channel_map={"audio": 0, "breath": 1, "stim": 2},
        has_stims=True,
        stim_length=0.1,
        output_exist_ok=True,  # False,
    )

    print(f"Finished! ({time.time() - st}s)")
