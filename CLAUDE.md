# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is the **AxialPolCap** paper repository: "Extending Deep-Learning P-Wave
Polarity Classification to Seafloor Microearthquakes at Axial Seamount."

The project trains a hybrid autoencoder + classifier (AxialPolCap) for
automatic P-wave first-motion polarity picking to support focal mechanism
analysis.  Seven OBS stations at Axial Seamount are used:
AS1, AS2, CC1, EC1, EC2, EC3, ID1.

The original development scripts (prefixed L_, M_, N_, O_) live in
`01-scripts/` for reference.  All active, publication-ready scripts are in
`scripts/` (see Workflow Pipeline below).

## Environment Setup

The conda environment is `tf-2.14.0` (Python 3.11, TensorFlow 2.13.1,
Keras 2.13.1):

```bash
conda activate tf-2.14.0
```

All scripts are run as standalone Python scripts from the repository root
(`FM_ML/`):

```bash
python scripts/<subdir>/<script_name>.py
```

There is no build system, test suite, or linter configured.

## Workflow Pipeline

Scripts live in `scripts/` organized by pipeline stage:

```
scripts/
  data_preparation/
    01_build_training_dataset.py      # augment + crop + split unified dataset
    02_build_eval_dataset.py          # build standalone eval set
    03_build_loso_training_dataset.py # augment + crop + split LOSO dataset
    04_build_loso_eval_dataset.py     # build LOSO eval set
  benchmark/
    eval_polarcap_baseline.py         # evaluate PolarCAP and other baselines
  training/
    train_axialpolcap.py              # train unified AxialPolCap model
    train_loso.py                     # LOSO cross-validation training
    transfer_learning.py              # transfer learning experiments
  evaluation/
    eval_model.py                     # evaluate unified model on test split
    eval_loso.py                      # evaluate LOSO models
    eval_transfer_learning.py         # evaluate transfer-learning models
  application/
    apply_to_catalog.py               # apply to 2015-2021 Axial Seamount catalog
```

### Key script descriptions

**01_build_training_dataset.py**
Replaces K_augument_mimic_org.m + L4_saveNPY_TMSF_001tran.py.
Reads directly from `Template_divide.mat`; fits a lognormal distribution to
per-station empirical SNR values; augments each template to reach ~15,000
samples per station; applies cubic-spline time shifts
(Normal(0, sigma=2 samples at 200 Hz)); center-crops 200->64 samples;
max-normalizes; stratified 80/10/10 split; saves to
`02-data/K_aug/TMSF_Tra_001/`.

**03_build_loso_training_dataset.py**
Replaces K_augument_all_step_SNRdis.m + L_saveNPY.py.
Reads from `Template.mat`; uses last 20% of each station's events;
step-SNR distribution (10% in [0,5] dB, 80% in [5,35] dB, 10% in [35,50] dB);
base_multi_trace=20 (ID1 gets 3x); no time shift; saves to
`02-data/K_aug/STEP010/`.

**02_build_eval_dataset.py** (was L3_saveNPY_TMSF_001_val.py)
Reads from the paired validation `.mat` files; no augmentation; saves
`timeseries_{STA}.npy` / `polarities_{STA}.npy` without a split prefix.

**04_build_loso_eval_dataset.py** (was L2_saveNPY_val.py)
Same as 02 but for the LOSO evaluation set.

**train_axialpolcap.py** (was M3_trainModel_newData.py)
Trains on merged all-station data from `K_aug/TMSF_Tra_001/`; saves JSON
training history + ROC curves; uses timestamped model filenames.

**train_loso.py** (was M_trainLOSO_CV.py)
LOSO training on `K_aug/STEP010/`; saves one model per held-out station as
`PolarPicker_LOSO_{STA}.keras`.

### Legacy scripts

`01-scripts/` contains earlier development iterations (prefixed L_, M_, N_,
O_) kept for reference only.  Do not run them on the paper datasets.

## Model Architecture (AxialPolCap)

Defined via `build_polarPicker()` in each training script.  It is a
**multi-output model** with a shared encoder:

- **Input**: 64-sample seismic waveform window (Z-component, P-arrival
  ±32 samples), shape `(64, 1)`
- **Encoder**: Conv1D(32,32) → Dropout → BN → MaxPool(2) → Conv1D(8,16) →
  BN → MaxPool(2) → output shape `(16, 8)`
- **Decoder** (autoencoder branch): Conv1D → BN → UpSample → Conv1D → BN →
  UpSample → Conv1D → shape `(64, 1)` — trained with MSE loss, weight=1
- **Classifier** (polarity head): Flatten → Dense(2, softmax) — trained with
  Huber loss, weight=200
- **Outputs**: `[decoder, classifier]`; prediction uses `model.predict(X)[1]`

The heavy loss weighting on the classifier (200x) means the encoder learns
representations driven by polarity discrimination, not just waveform
reconstruction.

## Data Conventions

**Raw data**: Per-station MATLAB `.mat` struct files with fields:
- `W_{STA}`: waveform (200 samples at 200 Hz)
- `Man_{STA}` or `Po_{STA}`: manual polarity label (-1 = Negative, +1 = Positive)

**Converted NPY files** (per station, saved in the active data subdirectory):
- `{split}_timeseries_{STA}.npy` — shape `(N, 64, 1)`, dtype float32,
  already normalized
- `{split}_polarities_{STA}.npy` — shape `(N,)`, dtype int32,
  values 0 (Negative) or 1 (Positive)
- `{split}_timeseries_all.npy` / `{split}_polarities_all.npy` — merged
  across all stations
- `timeseries_{STA}.npy` / `polarities_{STA}.npy` — no split prefix,
  used for standalone eval sets

**Train/val/test split ratio**: 80% train, 10% val, 10% test.
Implemented as `train_test_split(test_size=0.2)` then
`train_test_split(test_size=0.5)` on the remainder.  Splits are stratified
by polarity.

**Normalization**: each waveform is independently max-normalized:
`X / max(|X|)`.  Always apply before training or inference.

**LOSO** (Leave-One-Station-Out): each model is trained on 6 stations,
evaluated on the held-out 7th.  Model filenames follow
`PolarPicker_LOSO_{STA}.keras`.

**Output `.mat` files** (from `apply_to_catalog.py`): `Po_{STA}` field is
overwritten with a 4-element array
`[GroundTruth, Prediction, Confidence, Entropy]`.

## Directory Structure

```
FM_ML/
  scripts/                    # Active publication scripts
    data_preparation/
    benchmark/
    training/
    evaluation/
    application/
  01-scripts/                 # Legacy development scripts (reference only)
  02-data/                    # Input .mat files and converted .npy datasets
    K_aug/
      Template_divide.mat     # Templates for unified training (01_build_...)
      Template.mat            # All templates (03_build_... uses last 20%)
      STEP010/                # LOSO training .npy output
      TMSF_Tra_001/           # Unified training .npy output
    H_noi/
      H_Noise_200.mat         # Noise waveforms (200 samples, 200 Hz)
      H_noise_dB20_snrValue.mat  # Empirical per-station SNR distributions
  03-figs/                    # Output figures (confusion matrices, ROC curves)
    LOSO_010/                 # LOSO confusion matrices
  04-logs/                    # Log files
  05-tmp/                     # Temporary outputs (training history CSVs)
  06-models/                  # Saved Keras models (.keras format)
    LOSO_010/                 # LOSO models, one per held-out station
    history/                  # JSON training histories
```

## Key Implementation Notes

- `tf.config.run_functions_eagerly(True)` is set in training scripts to avoid
  TF graph serialization issues with Lambda layers used to name model outputs.
- **Two variants of `build_polarPicker()`** exist across scripts:
  - **List-style** (`train_axialpolcap.py`): `loss=['mse', hub]`,
    `loss_weights=[1, 200]`.  Models load without `safe_mode=False`.
  - **Dict-style with Lambda layers** (`train_loso.py`): outputs wrapped in
    named Lambda layers (`"decoder"`, `"classifier"`).  **These models require
    `safe_mode=False` on load.**
- When loading models with Lambda layers, use
  `keras.models.load_model(path, safe_mode=False)`.
- The model's two outputs are accessed as `y_raw[0]` (decoder) and
  `y_raw[1]` (classifier probabilities).  Some scripts check
  `isinstance(y_raw, (list, tuple))` to handle both single- and multi-output
  models.
- Use `scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)` to
  load MATLAB struct arrays; access fields with `getattr(struct_obj, 'field')`.
  Do NOT use `eval()` — use `mat_dict[station_name]` or `getattr()` instead.
- Data from HPC (Tallgrass cluster) uses absolute paths like
  `/caldera/projects/...`; local paths use
  `/Users/mcZhang/Documents/GitHub/FM_ML/`.
- The `02-data/` directory is not tracked in git.  Raw data are available
  from the OOI portal and the Wang et al. (2024) catalog at
  https://axialdd.ldeo.columbia.edu.
