# %%
# load_raw_files.py
#
# each cell creates preprocessed npy files and metadata dataframes for the included paths

from pathlib import Path

from utils.breath.preprocess import preprocess_files


def main():
    # %%
    # PATHS

    n_jobs = 10
    input_parent = Path(r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION")
    output_parent = Path(r"M:\randazzo\breathing\processed")

    # %%
    # CALLBACKS

    paths = [
        input_parent.joinpath(p)
        for p in [
            # rd57rd97
            "rd57rd97/preLesion/callbacks/female_230117",
            "rd57rd97/preLesion/callbacks/female_230118",
            "rd57rd97/preLesion/callbacks/male_230117",  # original
            "rd57rd97/preLesion/callbacks/male_230118",
            "rd57rd97/preLesion/callbacks/male_230119",
            # pk19br8 (all original)
            "pk19br8/preLesion/callback/rand/male_230215",
            "pk19br8/preLesion/callback/rand/male_230216",
            "pk19br8/preLesion/callback/rand/male_230217",
            # rd99rd72
            "rd99rd72/preLesion/callbacks/rand/230215",  # original
            "rd99rd72/preLesion/callbacks/rand/230216",
        ]
        # note:
        #   - excludes iso. 2 days/folders for each bird
        #   - excludes bird rd56, with moving breath baseline
    ]

    files = [file for path in paths for file in path.glob("**/*-B*.wav")]

    preprocess_files(
        files=files,
        output_folder=output_parent.joinpath("callbacks"),
        prefix="callback",
        fs=44100,
        channel_map={"audio": 0, "breath": 1, "stim": 2},
        has_stims=True,
        stim_length=0.1,
        output_exist_ok=False,
        n_jobs=n_jobs,
    )

    # %%
    # SPONTANEOUS

    paths = [
        input_parent.joinpath(p)
        for p in [
            # callback birds
            "rd57rd97/preLesion/spontaneous",
            "pk19br8/preLesion/spontaneous",
            "rd99rd72/preLesion/spontaneous",
            # TODO: other birds?
        ]
    ]

    files = [file for path in paths for file in path.glob("**/*.cbin")]

    preprocess_files(
        files=files,
        output_folder=output_parent.joinpath("spontaneous"),
        prefix="spontaneous",
        fs=32000,
        channel_map={"audio": 0, "breath": 1},
        has_stims=False,
        stim_length=0.1,
        output_exist_ok=False,
        n_jobs=n_jobs,
    )

    assert False, "check hvc lesion cb sample rate"

    # %%
    # HVC LESION - CALLBACK
    #
    # TODO: pare these down to birds with complete lesion

    input_parent = Path(r"M:\ESZTER\BEHAVIOR\AIR SAC CALLS\HVC LESION\ASPIRATION")

    paths = [
        input_parent.joinpath(p)
        for p in [
            # rd57rd97
            "rd57rd97/postLesion/callback/random/male_230125",
            "rd57rd97/postLesion/callback/random/male_230126",
            "rd57rd97/postLesion/callback/random/male_230127",
            # pk19br8
            "pk19br8/postLesion/callback/rand/230302",
            "pk19br8/postLesion/callback/rand/230303",
            # rd99rd72
            "rd99rd72/postLesion/callback/rand",
        ]
        # note:
        #   - excludes iso. >=1 day for each bird
    ]

    files = [file for path in paths for file in path.glob("**/*.wav")]

    preprocess_files(
        files=files,
        output_folder=output_parent.joinpath("hvc_lesion_callback"),
        prefix="hvc_lesion_callback",
        fs=44100,
        channel_map={"audio": 0, "breath": 1, "stim": 2},
        has_stims=True,
        stim_length=0.1,
        output_exist_ok=False,
        n_jobs=n_jobs,
    )

    # %%
    # HVC LESION - SPONTANEOUS
    #

    assert False, "check hvc lesion spontaneous sample rate"

    paths = [
        input_parent.joinpath(p)
        for p in [
            # callback birds
            "rd57rd97/postLesion/spontaneous",
            "pk19br8/postLesion/spontaneous",
            "rd99rd72/postLesion/spontaneous",
            # TODO: other birds?
        ]
    ]

    files = [file for path in paths for file in path.glob("**/*.cbin")]

    preprocess_files(
        files=files,
        output_folder=output_parent.joinpath("hvc_lesion_spontaneous"),
        prefix="hvc_lesion_spontaneous",
        fs=32000,
        channel_map={"audio": 0, "breath": 1},
        has_stims=False,
        stim_length=0.1,
        output_exist_ok=False,
        n_jobs=n_jobs,
    )

    return


# %%
if __name__ == "__main__":

    main()
