# %%
# load_raw_files.py

import glob

from utils.breath.preprocess import preprocess_files

if __name__ == "__main__":
    # Example usage

    cbin_files = [file for file in glob.glob("../data/test_input/cbin/*.cbin")]
    preprocess_files(
        files=cbin_files,
        output_folder="../data/test_output/CBIN",
        prefix="TEST_CBIN",
        fs=32000,
        channel_map={"audio": 0, "breath": 1},
        has_stims=False,
        stim_length=0.1,
        output_exist_ok=True,  # False,
    )

    wav_files = [file for file in glob.glob("../data/test_input/wav/*-B*.wav")]
    preprocess_files(
        files=wav_files,
        output_folder="../data/test_output/WAV",
        prefix="TEST_WAV",
        fs=44100,
        channel_map={"audio": 0, "breath": 1, "stim": 2},
        has_stims=True,
        stim_length=0.1,
        output_exist_ok=True,  # False,
    )
