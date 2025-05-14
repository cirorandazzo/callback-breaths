# %%
# get_traces
#
# ~13k seconds for ~26k files across 4 datasets (manakin)


import time
import pickle

from utils.breath.preprocess import load_datasets, TEMP_assert_file_quality
from utils.breath.traces import process_all_segments

# %%
if __name__ == "__main__":

    # %%
    # merge datasets
    #

    st = time.time()
    print("Getting dfs...")

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

    file_format = r"M:\randazzo\breathing\processed\{dataset}\{dataset}-{file}.pickle"

    all_files, all_breaths = load_datasets(
        datasets, file_format, fs_dataset, assertions=False
    )
    all_files, all_breaths = TEMP_assert_file_quality(all_files, all_breaths)

    print(f"dfs loaded! [total elapsed: {time.time() - st}s]")

    # %%
    # insps only

    type = "insp"
    all_breaths_type = all_breaths.loc[all_breaths["type"] == type]

    print(f"insps segmented! [total elapsed: {time.time() - st}s]")

    # %%
    # process segments

    print("Starting processing...")

    data = process_all_segments(
        all_breaths_type,
        data_row=1,
        interpolate_length=300,
        pad_frames=0,
        n_jobs=10,
        pickle_save_directory=r"M:\randazzo\breathing\processed\traces",  # None to suppress saving by file.
    )

    # %%
    # save
    save_path = r"M:\randazzo\breathing\processed\insp_traces.pickle"

    with open(save_path, "wb") as f:
        pickle.dump(data, f)


# %%
# sample code

# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt

# # load
# save_path = r"M:\randazzo\breathing\processed\insp_traces.pickle"
# with open(save_path, "rb") as f:
#     df = pickle.load(f)

# plot
# fig, ax = plt.subplots()
# ax.plot(df["data"].iloc[0])

# df.head()
