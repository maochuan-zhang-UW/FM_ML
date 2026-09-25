# AxialPolCap

**Adapting Deep-Learning P-Wave Polarity Classification to Ocean-Bottom Seismometers for Near-Real-Time Focal Mechanism Monitoring at Axial Seamount**

Maochuan Zhang¹\*, Marine Denolle², William S. D. Wilcock¹, Felix Waldhauser³, Kaiwen Wang⁴, Maya Tolstoy¹, Yen Joe Tan⁵

¹ School of Oceanography, University of Washington, Seattle, WA, USA
² Department of Earth and Space Sciences, University of Washington, Seattle, WA, USA
³ Lamont-Doherty Earth Observatory, Columbia University, Palisades, NY, USA
⁴ Institute of Geology and Geophysics, Chinese Academy of Sciences, Beijing, China
⁵ Department of Earth and Environmental Sciences, The Chinese University of Hong Kong, Hong Kong S.A.R., China

\* Corresponding author: mczhang8@uw.edu

Manuscript in preparation for *Seismological Research Letters*.

---

## Overview

AxialPolCap is a P-wave first-motion polarity classifier retrained from the PolarCAP
architecture (Chakraborty et al., 2022) on ocean-bottom seismometer data from the OOI
Regional Cabled Array at Axial Seamount. This repository holds the code that builds the
training data, benchmarks four published classifiers against a cross-correlation (CC)
method, trains and evaluates AxialPolCap, applies it to the 2015–2021 catalog, and
generates the manuscript figures.

Headline results:

| Quantity | Value |
|---|---|
| AxialPolCap, held-out test partition, no timing perturbation | ~98.5% |
| AxialPolCap, σ = 0.01 s shift applied in training and testing | ~96% |
| Four published DL models on the synthetic benchmark | 82–89% |
| Cross-correlation on the synthetic benchmark | ~92% |
| Leave-one-station-out, unshifted | 96.5–99.6% |
| Agreement with CC, 2015–2021 catalog (templates excluded) | 82.3% of 230,565 picks |
| Agreement for events with ≥5 confident polarities | 83.2% of 167,230 picks |
| Median Kagan angle, CC vs AxialPolCap composite mechanisms | ~13° (3,145 clusters) |

Seven stations are used throughout: AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3, AXID1
(`AS1`, `AS2`, `CC1`, `EC1`, `EC2`, `EC3`, `ID1` in code).

---

## Model

A hybrid autoencoder plus classifier with a shared encoder:

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

Training: Adam, learning rate 0.001, batch size 256, up to 40 epochs, early stopping on
validation loss with a patience of 5. The augmented set is split 80/10/10 into training,
validation and test partitions, stratified by polarity; the validation partition is used
for early stopping and model selection.

> **Known limitation.** The 80/10/10 split is applied to augmented waveforms rather than
> to parent templates, so different noisy realizations (and the sign-reversed copy) of a
> template can fall in different partitions. Accuracies on this partition therefore
> characterize robustness to noise and timing error rather than generalization to unseen
> earthquakes. This is stated in the manuscript and a template-grouped split is planned.

---

## Manuscript figures

Final figures are in `03-figs/`. Numbering follows the manuscript; the
`FigureNN_python.png` files use an earlier internal numbering and are kept for history.

| Manuscript | File | Produced by |
|---|---|---|
| Figure 1 | `SRL_Figure01.png` | `Figure01_background.m`, `Figure01_panelCD_prep.m`, `figure_01_standalone.py` |
| Figure 2 | `SRL_Figure02.png` | `make_manuscript_figures.py --figures 2` |
| Figure 3 | `SRL_Figure03.png` | `make_manuscript_figures.py --figures 4` |
| Figure 4 | `SRL_Figure04.png` | `make_manuscript_figures.py --figures 6` |
| Figure 5 | `SRL_Figure05.png` | `make_manuscript_figures.py --figures 7` |
| Figure 6 | `SRL_Figure06.png` | `make_manuscript_figures.py --figures 8` |
| Figure 7 | `SRL_Figure07.png` | `make_manuscript_figures.py --figures 9` |
| Figure 8 | `SRL_Figure08.png` | `Figure08_mechanism_comparison.m` |
| Figure 9 | `SRL_Figure09.png` | schematic |
| Figure S1 | `SRL_FigureS1.png` | `make_manuscript_figures.py` (confusion matrices) |
| Figure S2 | `SRL_FigureS2.png` | `make_manuscript_figures.py --figures 5` |
| Figure S3 | `SRL_FigureS3.png` | `make_manuscript_figures.py --figures 16` |
| Figure S4 | `SRL_FigureS4.png` | `plot_figure11_2015_2021.py` |
| Figure S5 | `SRL_FigureS5_snr.png` | `plot_figureS5_snr_map.py` |
| Figure S6 | `SRL_FigureS5.png` | `Figure09_kagan_angle.py` / `FigureS5_kagan_2022_2026.m` |
| Table S2 | `TableS1_benchmark_accuracy.md` | `make_manuscript_figures.py` |

Two commands worth recording, because their defaults differ from the published figures:

```bash
# Figure S4: spatial disagreement between AxialPolCap and CC
python 01-scripts/plot_figure11_2015_2021.py \
    --grid-m 200 --min-per-cell 150 --ratio-max 30 --exclude-templates \
    --output 03-figs/SRL_FigureS4.png

# Figure S5: spatial variation of P-wave SNR, on the same grid and cell minimum
python 01-scripts/plot_figureS5_snr_map.py \
    --grid-m 200 --min-per-cell 150 \
    --output 03-figs/SRL_FigureS5_snr.png
```

---

## Repository layout

```
01-scripts/
  data_preparation/      build training and evaluation datasets (NPY and HDF5 variants)
  benchmark/             evaluate the PolarCAP baseline
  training/              train AxialPolCap; leave-one-station-out; transfer learning
  evaluation/            evaluate LOSO and transfer-learning models
  application/           apply the model to the 2015-2021 catalog
  build_skhash_input_*.py, build_hash_input_*.py, compare_skhash_cc_ml.py
                         focal-mechanism inputs and CC/ML mechanism comparison
  plot_figure11_2015_2021.py, plot_figureS4_2022_2026.py,
  plot_figureS5_snr_map.py, plot_disagreement_vs_snr.py
                         supplementary figures
  *.m                    MATLAB scripts for Figures 1, 8 and the Kagan histogram
  make_manuscript_figures.py
                         most manuscript figures; --figures N regenerates a subset
03-figs/                 figure outputs
06-models/               trained Keras models, LOSO models, training histories
04-logs/                 run logs
```

Not tracked: `02-data/` and `07-files/` (input waveforms, catalogs, bathymetry and
velocity models), the vendored third-party model repositories under `01-scripts/`, and
the manuscript sources. See **Data availability**.

---

## Environment

```bash
conda activate tf_macos      # Python 3.11, TensorFlow 2.13.1, Keras 2.13.1
```

Run every script from the repository root:

```bash
python 01-scripts/<subdir>/<script>.py
```

Notes:

- The TensorFlow Metal plugin crashes during `model.predict` and fine-tuning on macOS.
  Inference and fine-tuning scripts call `tf.config.set_visible_devices([], 'GPU')`; keep
  that pattern in any new script.
- Models whose outputs are wrapped in named `Lambda` layers (from `train_loso.py`) must be
  loaded with `keras.models.load_model(path, safe_mode=False)`.
- `make_manuscript_figures.py` needs `cartopy` for the map panel.

---

## Data availability

Seismic data are from the EarthScope Consortium Data Management Center, network code
`OO`. The earthquake catalog of Wang et al. (2024) is at
<https://axialdd.ldeo.columbia.edu>. The composite focal-mechanism catalog and polarity
determinations produced here are at
<http://axial.ocean.washington.edu/FocalMechanisms.html>. Input data are not stored in
this repository because of their size.

---

## References

- Chakraborty, M., et al. (2022). PolarCAP: a deep learning approach for first motion polarity classification of earthquake waveforms. *Artificial Intelligence in Geosciences*, 3, 46–52. doi:10.1016/j.aiig.2022.08.001
- Chen, Y., et al. (2024). Deep learning for P-wave first-motion polarity determination and its application in focal mechanism inversion. *IEEE TGRS*, 62, 1–11. doi:10.1109/TGRS.2024.3407060
- Messuti, G., et al. (2023). CFM: a convolutional neural network for first-motion polarity classification of seismic records in volcanic and tectonic areas. *Frontiers in Earth Science*, 11, 1223686. doi:10.3389/feart.2023.1223686
- Skoumal, R. J., Hardebeck, J. L., and Shearer, P. M. (2024). SKHASH: a Python package for computing earthquake focal mechanisms. *SRL*, 95(4), 2519–2526. doi:10.1785/0220230329
- Wang, K., et al. (2024). Real-time detection of volcanic unrest and eruption at Axial Seamount using machine learning. *SRL*, 95(5), 2651–2662. doi:10.1785/0220240086
- Zhao, M., et al. (2023). DiTingMotion: a deep-learning first-motion-polarity classifier and its application to focal mechanism inversion. *Frontiers in Earth Science*, 11, 1103914. doi:10.3389/feart.2023.1103914
