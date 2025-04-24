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

#### 00: callback, first insp only

First pass at umap, using only the first inspiration of each callback trial

- `umap-00.00-train_first_insp.py`
- `umap-00.01-analyze_first_insp.py`

#### 01: callback, all breaths

Train UMAP on all of 1 breath segment type in callback data (eg, all insps). Can adjust which segment in files.

- `umap-01.00-train_all_cb.py`
- `umap-01.01-analyze_all_cb.py`
- `umap-01.02-map_clusters.py`
  - Which expirations come from which inspirations? Map insp and exp clusters.
  - I didn't refine expiration embedding/clustering particularly well, so this wasn't especially revealing.

#### 02: spontaneous

Consider how spontaneous breaths fall into callback-trained embedding. No additional UMAP training goes on here.

- `umap-02.00-add_spontaneous.py`
- `umap-02.01-analyze_spontaneous.py`

#### Miscellaneous

- `umap-input_walkthrough.py`
  - Plot examples of the types of traces used for umap input.
  - Note: nearly all of the later embeddings simply use interpolated - UMAP is most useful for determining shape, since duration, timing, etc. can be added back later.
