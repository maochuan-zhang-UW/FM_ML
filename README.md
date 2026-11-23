# AxialPolCap

**Adapting deep-learning P-wave polarity classification to ocean-bottom seismometers for near-real-time focal mechanism monitoring at Axial Seamount**

Maochuan Zhang¹\*, Marine Denolle², William S. D. Wilcock¹, Felix Waldhauser³, Kaiwen Wang⁴, Maya Tolstoy¹, Yen Joe Tan⁵

¹ School of Oceanography, University of Washington, Seattle, WA, USA
² Department of Earth and Space Sciences, University of Washington, Seattle, WA, USA
³ Lamont-Doherty Earth Observatory, Columbia University, Palisades, NY, USA
⁴ Institute of Geology and Geophysics, Chinese Academy of Sciences, Beijing, China
⁵ Department of Earth and Environmental Sciences, The Chinese University of Hong Kong, Hong Kong S.A.R., China

\* Corresponding author: mczhang8@uw.edu

Manuscript submitted to *Geophysical Journal International*.

---

## Overview

AxialPolCap is a P-wave first-motion polarity classifier that adapts the PolarCAP
architecture (Chakraborty et al. 2022) to ocean-bottom seismometer data from the OOI
Regional Cabled Array at Axial Seamount. This repository holds the code that builds the
training and test data, benchmarks four published classifiers against waveform
cross-correlation (CC), compares training from scratch with transfer learning, applies
the model to the 2015–2021 catalogue, and generates the manuscript figures. It also
holds the trained model weights.

Headline results (station averages):

| Quantity | Value |
|---|---|
| AxialPolCap, σ = 0.01 s arrival-time shift in training and testing | ~96% |
| Model trained from scratch, unshifted training and test data (Fig. 4) | 98.0% |
| Transfer learning (PolarCAP, classifier frozen, encoder retrained), unshifted (Fig. 4) | 97.8% |
| Four published deep-learning models on the same test data, unshifted (Fig. 3) | 82–89% |
| Cross-correlation on the same test data (Fig. 3) | ~92% |
| Leave-one-station-out, unshifted (Fig. 4) | 94.8–99.6% |
| Agreement with CC, 2015–2021 catalogue, templates excluded (Section 4.1) | 82.3% of 230,565 picks |
| Median Kagan angle, CC vs AxialPolCap composite mechanisms | ~13° (3,145 clusters) |

Seven stations are used throughout: AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3, AXID1
(`AS1`, `AS2`, `CC1`, `EC1`, `EC2`, `EC3`, `ID1` in code).

---

## Model

A hybrid autoencoder plus classifier with a shared encoder (Fig. S2):

```
Input: (64, 1)   64-sample vertical-component window, P arrival ±0.32 s at 100 Hz

Encoder    Conv1D(32, k=32) -> Dropout(0.3) -> BatchNorm -> MaxPool(2)
           Conv1D( 8, k=16) ->                 BatchNorm -> MaxPool(2)
           latent: (16, 8)

Decoder    Conv1D -> BatchNorm -> UpSample -> Conv1D -> BatchNorm -> UpSample -> Conv1D
           reconstruction: (64, 1),  MSE loss, weight 1

Classifier Flatten -> Dense(2, softmax)
           polarity,  Huber loss (delta = 0.5), weight 200
```

**Training from scratch:** Adam, learning rate 0.001, batch size 256, up to 40 epochs,
early stopping after 5 epochs without improvement in validation loss.

**Transfer learning** starts from the pretrained PolarCAP weights. Three configurations
were tested:
- linear probing: encoder frozen, a new softmax classifier trained;
- full fine-tuning: all layers retrained;
- classifier frozen, encoder retrained.

The last one is the transfer-learning model of Figs 4–7. All three use cross-entropy
only, batch size 512, an initial learning rate of 10⁻⁴ (3 × 10⁻⁴ for linear probing)
halved after 2 epochs without improvement in validation loss, up to 40 epochs, and early
stopping on validation accuracy. Every configuration was trained with three random
seeds; the figures show the means.

**Data split.** 20 per cent of the templates, stratified by polarity, are set aside
before augmentation; their augmented waveforms form the test data behind every accuracy
in Section 3. The remaining templates are augmented and split 80/10/10 into training,
validation and internal test partitions, and the validation partition is used for
early stopping.

**Final model.** `06-models/AxialPolCap_final.keras` was trained from scratch with
σ = 0.01 s time shifts (seed 2 of the Fig. 7 runs). It scores 97.6% on unshifted test
data and 96.8% with σ = 0.01 s shifts.

---

## Trained models

| File | Content |
|---|---|
| `06-models/AxialPolCap_final.keras` (+ `.json`) | final AxialPolCap model, Section 3.3 |
| `06-models/retrain_20261122/*.keras`, `*.json` | the 45 training runs behind Figs 4–7 and S3: per-epoch validation accuracy, trainable-parameter count and per-station test accuracy on the unshifted, σ = 0.01 s and σ = 0.02 s test sets |
| `06-models/PolarPicker_h5_20260308_154910.keras` | model applied to the 2015–2021 catalogue for the CC comparison of Section 4.1 (`apply_polarpicker_h5_2015_2021_conf80.py`) |

Run names follow `<training set>_<strategy>_s<seed>[_loso<station>]`. The strategies are
`scratch`, `freezecls` (classifier frozen), `finetune` and `probe` (linear probing). The
training sets are listed under **Reproducing Figs 4–7 and S3**.

Load with `keras.models.load_model(path, compile=False)`; polarity probabilities are
`model.predict(X)[1]` (class 0 = negative, 1 = positive), with each 64-sample window
max-normalized first.

To pick polarities for a new catalogue with the final model:

```bash
python 01-scripts/application/apply_to_catalog.py --input <catalogue.mat> --output <predictions.mat>
```

The input holds a struct array `Felix` with 64-sample windows `W_{STA}` centred on the P
pick. Each `Po_{STA}` is returned as `[catalogue polarity, prediction, confidence, entropy]`.

---

## Manuscript figures

| Manuscript | File in `03-figs/` | Produced by |
|---|---|---|
| Figure 1 | `GJI_Figure01.png` | `Figure01_background.m`, `Figure01_panelCD_prep.m`, `figure_01_standalone.py` |
| Figure 2 | `GJI_Figure02.png` | `make_manuscript_figures.py --figures 15` |
| Figure 3 | `GJI_Figure03.png` | `make_manuscript_figures.py --figures 4` |
| Figures 4–7 | `GJI_Figure04.png` – `GJI_Figure07.png` | `evaluation/plot_training_strategy_figures.py` |
| Figure 8 | `GJI_Figure08.png` | `Figure08_mechanism_comparison.m` |
| Figure 9 | `GJI_Figure09.png` | schematic, drawn separately |
| Figure S1 | `GJI_FigureS1.png` | `make_manuscript_figures.py --figures 17` |
| Figure S2 | `GJI_FigureS2.jpg` | architecture diagram, modified from Chakraborty et al. (2022) |
| Figure S3 | `GJI_FigureS3.png` | `evaluation/plot_training_strategy_figures.py` |
| Figure S4 | `GJI_FigureS4.png` | `make_manuscript_figures.py --figures 10` |
| Figure S5 | `GJI_FigureS5.png` | `plot_figure11_2015_2021.py` (command below) |
| Figure S6 | `GJI_FigureS6.png` | `plot_figureS5_snr_map.py` (command below) |
| Figure S7 | `GJI_FigureS7.png` | `Figure09_kagan_angle.m` |
| Table S2 | `TableS1_benchmark_accuracy.md` | `make_manuscript_figures.py` |

The `GJI_*` files are the images as they appear in the submitted manuscript. The values
plotted in Figs 4–7 are in `03-figs/training_strategy_numbers.json`. The older
`FigureNN_python.png` files use an earlier internal numbering and are kept for history.

Commands whose defaults differ from the published figures:

```bash
# Figure S5: spatial disagreement between AxialPolCap and CC
python 01-scripts/plot_figure11_2015_2021.py \
    --grid-m 200 --min-per-cell 150 --ratio-max 30 --exclude-templates \
    --output 03-figs/GJI_FigureS5.png

# Figure S6: spatial variation of P-wave SNR, on the same grid and cell minimum
python 01-scripts/plot_figureS5_snr_map.py \
    --grid-m 200 --min-per-cell 150 \
    --output 03-figs/GJI_FigureS6.png
```

---

## Reproducing Figs 4–7 and S3

1. **Augmented waveforms** (MATLAB, `01-scripts/data_preparation/augmentation/`):
   - `K_augument_mimic_org.m` builds the original catalogue SNR distribution;
   - `K_augument_mimic_step.m` builds high SNR, 10–30 dB;
   - `K_augument_mimic_step_0_10.m` builds low SNR, 0–10 dB;
   - `L_saveNPY_Validate.m` builds the test templates.

   These are the scripts as used and contain absolute paths.
2. **Test sets** (`augmentation/L2_saveNPY_val.py`, `L3_saveNPY_TMSF_001_val.py`,
   `L3_saveNPY_TMSF_002_val.py`): unshifted `Val`, σ = 0.01 s `TMSF_Val_001`, and
   σ = 0.02 s `TMSF_Val`.
3. **Training sets** (`data_preparation/05_build_timeshift_training_sets.py`). All share
   the same templates, labels and split:

   | Set | Input | Shift σ (samples) | Used for |
   |---|---|---|---|
   | `TMSF_Tra_000` | `Org/*_T_add.mat` | 0 | Figs 4, 5 (original SNR), 6, S3 |
   | `TMSF_Tra_001` | `Org/*_T_add.mat` | 1 (0.01 s) | Fig. 7, final model |
   | `TMSF_Tra_002s` | `Org/*_T_add.mat` | 2 (0.02 s) | Fig. 7 |
   | `STEP` | `STEP/*_T_add_ST.mat` | 0 | Fig. 5, high SNR |
   | `STEP010` | `STEP010/*_T_add_ST.mat` | 0 | Fig. 5, low SNR |

4. **Training:** `bash 01-scripts/training/run_training_strategies.sh` runs all 45 jobs
   of `training/retrain_fig4_strategies.py`. Transfer-learning runs need the pretrained
   `PolarCAP.h5` from Chakraborty et al. (2022), located through `POLARCAP_H5`; the test
   sets are located through `AXIAL_EVAL_ROOT`.
5. **Figures:** `python 01-scripts/evaluation/plot_training_strategy_figures.py`.

---

## Repository layout

```
01-scripts/
  data_preparation/      training/test datasets (NPY and HDF5 variants);
                         augmentation/ holds the MATLAB augmentation and NPY conversion scripts
  benchmark/             evaluate the PolarCAP baseline
  training/              train AxialPolCap; LOSO; transfer learning; the Figs 4-7 runs
  evaluation/            evaluation and the Figs 4-7 / S3 plots
  application/           apply the model to the 2015-2021 catalogue
  build_skhash_input_*.py, build_hash_input_*.py, compare_skhash_cc_ml.py
                         focal-mechanism inputs and CC/ML mechanism comparison
  plot_figure11_2015_2021.py, plot_figureS4_2022_2026.py,
  plot_figureS5_snr_map.py, plot_disagreement_vs_snr.py
                         supplementary figures
  *.m                    MATLAB scripts for Figures 1, 8 and S7
  make_manuscript_figures.py
                         remaining manuscript figures; --figures N regenerates a subset
03-figs/                 figure outputs (GJI_* = submitted versions)
06-models/               trained Keras models and per-run results
04-logs/                 run logs
```

Not tracked: `02-data/` and `07-files/` (input waveforms, catalogues, bathymetry and
velocity models), the vendored third-party model repositories and pretrained weights
under `01-scripts/`, and the manuscript sources. See **Data availability**.

---

## Environment

The results in Figs 4–7 and S3 and the final model were produced with Python 3.10,
TensorFlow 2.16.2 and Keras 3.11.3 (`environment.yml`):

```bash
conda env create -f environment.yml
conda activate axialpolcap
```

Run every script from the repository root:

```bash
python 01-scripts/<subdir>/<script>.py
```

Notes:

- The TensorFlow Metal plugin crashes during `model.predict` and fine-tuning on macOS.
  Inference and training scripts call `tf.config.set_visible_devices([], 'GPU')`.
- Models whose outputs are wrapped in named `Lambda` layers (from `train_loso.py`) must
  be loaded with `keras.models.load_model(path, safe_mode=False)`.
- `make_manuscript_figures.py` needs `cartopy` for the map panel; Figures 1, 8 and S7
  use MATLAB (R2024a).

---

## Data availability

The seismic data were downloaded from the EarthScope Consortium Data Management Center
with network code `OO`. The continuously updated earthquake catalogue is at
<https://axialdd.ldeo.columbia.edu>. The composite focal-mechanism catalogue and polarity
determinations produced in this study are at
<http://axial.ocean.washington.edu/FocalMechanisms.html>. Input data are not stored in
this repository because of their size. Focal mechanisms are computed with SKHASH
(Skoumal et al. 2024; <https://code.usgs.gov/esc/SKHASH>).

---

## References

- Chakraborty, M., et al. (2022). PolarCAP–A deep learning approach for first motion polarity classification of earthquake waveforms. *Artif. Intell. Geosci.*, 3, 46–52. doi:10.1016/j.aiig.2022.08.001
- Chen, Y., et al. (2024). Deep learning for P-wave first-motion polarity determination and its application in focal mechanism inversion. *IEEE Trans. Geosci. Remote Sens.*, 62, 1–11. doi:10.1109/TGRS.2024.3407060
- Messuti, G., et al. (2023). CFM: a convolutional neural network for first-motion polarity classification of seismic records in volcanic and tectonic areas. *Front. Earth Sci.*, 11, 1223686. doi:10.3389/feart.2023.1223686
- Skoumal, R. J., Hardebeck, J. L., & Shearer, P. M. (2024). SKHASH: a Python package for computing earthquake focal mechanisms. *Seismol. Res. Lett.*, 95(4), 2519–2526. doi:10.1785/0220230329
- Wang, K., et al. (2024). Real-time detection of volcanic unrest and eruption at Axial Seamount using machine learning. *Seismol. Res. Lett.*, 95(5), 2651–2662. doi:10.1785/0220240086
- Zhao, M., et al. (2023). DiTingMotion: a deep-learning first-motion-polarity classifier and its application to focal mechanism inversion. *Front. Earth Sci.*, 11, 1103914. doi:10.3389/feart.2023.1103914

---

## License

The code and trained models in this repository are released under the [MIT License](LICENSE).
