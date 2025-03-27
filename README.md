# Callback Analysis for Breathing

Code for analyzing respiratory traces during callback experiments in the Brainard Lab (UCSF/HHMI). Initially split from [callback-analysis](https://github.com/cirorandazzo/callback-analysis) repo.

## Files

### Data Preprocessing

- `make_breath_dfs_plots.py`: make dataframe for cbpt metadata & `kde-threshold` plots
- `preprocess_breath_traces.py`: process wav files to filter, center, and normalize breath traces, saving as npy arrays
  - TODO: edit implementations to load & slice from these, rather than (1) making enormous dataframes or (2) reprocessing from raw wav file as needed. (UMAP stuff especially!)
- `phase.py`: initial implementation of phase, some descriptive stuff, and first pass at tying into UMAP clusters

### [Callback Breathing Plot Tool](https://cirorandazzo.github.io/callbacks-breathing-plot_tool/)

- `make_breath_dfs_plots.py`: `kde-threshold` plots for cbpt; see above.
- `plot_rolling_min_subtracted.py`: `lowpass_trace-rolling_min_seg` plots for cbpt
- `plot_spectrograms.py`: `spectrogram` plots for cbpt

### Breath Segmentation

- `amplitude_distributions.py`: plot & fit breath amplitude distribution for many files; important step in determining zero point algorithm
- `zero_point.py`: long file containing many attempts at addressing drift in respiratory trace over the course of a file (ie, non-constant zero point). I wound up giving up on those files, since it was only the case in 1/4 birds

### UMAP

- `umap-train_first_insp.py`: first pass at umap, using only the first inspiration of each callback trial
  - Analysis in `umap-first_insp_only.py`
- `umap-train_all.py`: analyzing umap on all breaths (implemented for either insp or exp)
  - Analysis in `umap-all_breaths.py`

- `map-clusters.py`: which expirations come from which inspirations? (map insp & exp clusters)
- `umap-input_walkthrough.py`: plot examples of the types of traces used for umap input
