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
Regional Cabled Array at Axial Seamount. This repository holds the complete workflow
behind the paper, in Python, as one numbered pipeline: it builds the training and
test data from the template waveforms and real seafloor noise, trains and evaluates the
models, applies the final model to the 2015–2021 catalogue, computes composite focal
mechanisms from catalogue (CC) and AxialPolCap polarities with SKHASH, and draws the
manuscript figures. It also holds the trained model weights.

Headline results (station averages):

| Quantity | Value |
|---|---|
| AxialPolCap, σ = 0.01 s arrival-time shift in training and testing | ~96% |
| Model trained from scratch, unshifted training and test data (Fig. 4) | 98.0% |
| Transfer learning (PolarCAP, classifier frozen, encoder retrained), unshifted (Fig. 4) | 97.8% |
| Four published deep-learning models on the same test data, unshifted (Fig. 3) | 82–89% |
| Cross-correlation on the same test data (Fig. 3) | ~92% |
| Leave-one-station-out, unshifted (Fig. 4) | 94.8–99.6% |
| Agreement with CC, 2015–2021 catalogue, templates excluded (Section 4.1) | 84.4% of 229,574 picks |
| Median Kagan angle, CC vs AxialPolCap composite mechanisms, both quality A–B (Section 4.1) | ~15° (1,152 clusters; 38° over all 3,611 paired) |

Seven stations are used throughout: AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3, AXID1
(`AS1`, `AS2`, `CC1`, `EC1`, `EC2`, `EC3`, `ID1` in code).

---

## Pipeline

Run every step from the repository root. Step 0 checks the input data, steps 1–3 build
the data, step 4 trains, steps 5–7 apply the final model to the catalogue (Section 4.1).

**Getting the data.** Download the data archive (see **Data availability**) and unpack it
in the repository root. It holds `02-data/raw/` (templates, noise library, catalogue
windows, CC clusters, station geometry), `02-data/archive/` (the augmented waveforms
used for the paper) and the map files in `07-files/`: 50 files, 1.8 GB, listed with
their SHA-256 in `data_manifest.csv`. Then run `python 01-scripts/00_check_data.py`.
The results archive holds the outputs of steps 5–8 (listed in `results_manifest.csv`;
check with `--manifest results_manifest.csv`), so the Section 4 figures can be drawn
without rerunning the model or SKHASH.

| Step | Script | Does | Output |
|---|---|---|---|
| 0 | `00_check_data.py` | checks every input file against `data_manifest.csv` | |
| 1 | `01_prepare_inputs.py` | converts the templates, noise library and observed SNR from MATLAB `.mat` to HDF5 | `02-data/prepared/inputs.h5` |
| 2 | `02_augment.py` | adds real seafloor noise to every template at a target SNR (lognormal fit to the catalogue, or uniform) | `02-data/augmented/*.h5` |
| 3 | `03_build_datasets.py` | time-shifts the P arrival, crops 64 samples, normalizes, splits 80/10/10 | `02-data/K_aug/<set>/*.npy` |
| 4 | `04_train.py` (all runs: `04_run_all_training.sh`) | trains one strategy and evaluates it per station on the three test sets | `06-models/retrain_20261122/` |
| 5 | `05_apply_to_catalog.py` | classifies the 2015–2021 catalogue windows with `AxialPolCap_final` | predictions `.mat` |
| 6 | `06_compare_with_cc.py` | agreement with CC polarities (Section 4.1) | `02-data/section41_agreement.json` |
| – | `benchmark/score_benchmark.py` | accuracy and confusion matrices of four published classifiers (Fig. 3, S1, Table S2) | `03-figs/benchmark_confusion_matrix.csv` |
| 7 | `07_focal_mechanisms.py` | composite mechanisms from CC and from AxialPolCap polarities with SKHASH, Kagan angles | `02-data/focal_mechanisms_2015_2021/mechanisms.csv` |
| 8 | `08_export_results.py` | exports the AxialPolCap polarities as a CSV, one row per pick | `02-data/results/AxialPolCap_polarities_2015_2021.csv` |

The commands that rebuild every dataset of the paper:

```bash
python 01-scripts/01_prepare_inputs.py
# augmentation: original-SNR, high-SNR and low-SNR training sets, and the test set
python 01-scripts/02_augment.py --split train --snr lognormal --target 15000 --out 02-data/augmented/train_org.h5
python 01-scripts/02_augment.py --split train --snr uniform --snr-range 10 30 --target 15000 --out 02-data/augmented/train_step.h5
python 01-scripts/02_augment.py --split train --snr uniform --snr-range 0 10 --target 15000 --out 02-data/augmented/train_step010.h5
python 01-scripts/02_augment.py --split test --snr lognormal --target 5000 --out 02-data/augmented/test.h5
# training sets (sigma in samples of 0.01 s)
python 01-scripts/03_build_datasets.py --input 02-data/augmented/train_org.h5 --mode trainvaltest --out 02-data/K_aug/TMSF_Tra_000
python 01-scripts/03_build_datasets.py --input 02-data/augmented/train_org.h5 --mode trainvaltest --sigma-samples 1 --seed 1 --out 02-data/K_aug/TMSF_Tra_001
python 01-scripts/03_build_datasets.py --input 02-data/augmented/train_org.h5 --mode trainvaltest --sigma-samples 2 --seed 2 --out 02-data/K_aug/TMSF_Tra_002s
python 01-scripts/03_build_datasets.py --input 02-data/augmented/train_step.h5 --mode trainvaltest --out 02-data/K_aug/STEP
python 01-scripts/03_build_datasets.py --input 02-data/augmented/train_step010.h5 --mode trainvaltest --out 02-data/K_aug/STEP010
# test sets: unshifted, sigma = 0.01 s, sigma = 0.02 s
python 01-scripts/03_build_datasets.py --input 02-data/augmented/test.h5 --mode test --out 02-data/K_aug/Val
python 01-scripts/03_build_datasets.py --input 02-data/augmented/test.h5 --mode test --sigma-samples 1 --seed 1 --out 02-data/K_aug/TMSF_Val_001
python 01-scripts/03_build_datasets.py --input 02-data/augmented/test.h5 --mode test --sigma-samples 2 --seed 2 --out 02-data/K_aug/TMSF_Val
```

| Set | Augmented input | Shift σ | Used for |
|---|---|---|---|
| `TMSF_Tra_000` | original SNR | 0 | Figs 4, 5 (original SNR), 6, S3 |
| `TMSF_Tra_001` | original SNR | 0.01 s | Fig. 7, final model |
| `TMSF_Tra_002s` | original SNR | 0.02 s | Fig. 7 |
| `STEP` | 10–30 dB | 0 | Fig. 5, high SNR |
| `STEP010` | 0–10 dB | 0 | Fig. 5, low SNR |
| `Val`, `TMSF_Val_001`, `TMSF_Val` | test templates, original SNR | 0, 0.01 s, 0.02 s | test data of every accuracy in Section 3 |

**Archived datasets.** Augmentation is random, so a rerun gives statistically equivalent
but not identical waveforms. The augmented waveforms used for the paper were generated
with MATLAB (the original scripts are in the git history before this restructuring) and
are in `02-data/archive/` of the data archive. `03_build_datasets.py` reads them too and
rebuilds the paper's datasets exactly, e.g.
`--input 02-data/archive/Val --suffix _V_add.mat --mode test --out 02-data/K_aug/Val`
(`Org`, `STEP`, `STEP010` with `--suffix _T_add.mat`, `_T_add_ST.mat`, `_T_add_ST.mat`).

**Checks of the Python port** (`01-scripts/checks/`):
- `03_build_datasets.py` on the archived augmentation reproduces `Val` and `TMSF_Tra_000`
  exactly and `STEP` to float32 rounding;
- `02_augment.py` against the archived augmentation (`compare_augmentation.py`): identical
  waveform counts and polarity balance for every station and set, median SNR within
  0.2 dB, two-sample Kolmogorov–Smirnov D ≤ 0.021;
- a model trained on Python-augmented data (`06-models/checks/`) scores 97.4 / 96.5 /
  93.1% on the unshifted / 0.01 s / 0.02 s test sets, against 97.4–97.6 / 96.7–96.8 /
  93.4–93.5% for the three seeds trained on the archived data;
- `benchmark/score_benchmark.py` reproduces every benchmark accuracy and confusion
  matrix of Fig. 3, Fig. S1 and Table S2 exactly from the stored model predictions
  (`benchmark/provenance/` holds the scripts that ran the four models);
- `06_compare_with_cc.py` reproduces every Section 4.1 statistic of the earlier
  manuscript version exactly when given that version's predictions.

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
| `06-models/checks/python_augmentation_scratch_s2.json` | scores of a model trained on Python-augmented data (pipeline check) |

Run names follow `<training set>_<strategy>_s<seed>[_loso<station>]`. The strategies are
`scratch`, `freezecls` (classifier frozen), `finetune` and `probe` (linear probing). The
training sets are listed under **Pipeline**.

Load with `keras.models.load_model(path, compile=False)`; polarity probabilities are
`model.predict(X)[1]` (class 0 = negative, 1 = positive), with each 64-sample window
max-normalized first.

To pick polarities for a new catalogue with the final model:

```bash
python 01-scripts/05_apply_to_catalog.py --input <catalogue.mat> --output <predictions.mat>
```

The input holds a struct array `Felix` with 64-sample windows `W_{STA}` centred on the P
pick. Each `Po_{STA}` is returned as `[catalogue polarity, prediction, confidence, entropy]`;
the prediction is 0 where the confidence is below `--confidence-threshold` (default 0.8).
Section 4 retains predictions with confidence ≥ 0.95.

---

## Manuscript figures

All figures are drawn in Python by the scripts in `01-scripts/figures/`.

| Manuscript | File in `03-figs/` | Produced by |
|---|---|---|
| Figure 1 | `GJI_Figure01.png` | `fig01_background.py` |
| Figure 2 | `GJI_Figure02.png` | `fig02_03_S1_S4.py` |
| Figure 3 | `GJI_Figure03.png` | `fig02_03_S1_S4.py` |
| Figures 4–7, S3 | `GJI_Figure04.png` – `GJI_Figure07.png`, `GJI_FigureS3.png` | `fig04_07_S3_training_strategies.py` |
| Figure 8 | `GJI_Figure08.png` | `fig08_mechanisms.py` |
| Figure 9 | `GJI_Figure09.png` | schematic, drawn separately |
| Figure S1 | `GJI_FigureS1.png` | `fig02_03_S1_S4.py` |
| Figure S2 | `GJI_FigureS2.jpg` | architecture diagram, modified from Chakraborty et al. (2022) |
| Figure S4 | `GJI_FigureS4.png` | `fig02_03_S1_S4.py` |
| Figure S5 | `GJI_FigureS5.png` | `figS5_disagreement_map.py --exclude-templates` |
| Figure S6 | `GJI_FigureS6.png` | `figS6_snr_map.py` |
| Figure S7 | `GJI_FigureS7.png` | `figS7_kagan_histogram.py` |
| Table S2 | `TableS1_benchmark_accuracy.md` | `fig02_03_S1_S4.py` |

Figs 3 and S1 and Table S2 read `03-figs/benchmark_confusion_matrix.csv`, written by
`benchmark/score_benchmark.py`; run that first. The cross-correlation row is a constant:
its inputs are not in the data archive (see `benchmark/provenance/README.md`).

Each script writes its `GJI_*` file into `03-figs/`. The values plotted in Figs 4–7 are
in `03-figs/training_strategy_numbers.json`.

---

## Repository layout

```
01-scripts/
  00_check_data.py ... 08_export_results.py     the pipeline, in order
  04_run_all_training.sh   the 45 training runs behind Figs 4-7 and S3
  mechanism_tools.py       P/T/B axes, fault type (Table S4), Kagan angle, SKHASH output
  figures/                 one script per manuscript figure (see table above)
  checks/                  checks of the Python port against the archived MATLAB data
  benchmark/               benchmark scoring; provenance/ holds the scripts that ran the four models
03-figs/                   figure outputs (GJI_* = manuscript versions)
06-models/                 trained Keras models and per-run results
```

Not tracked: `02-data/` and `07-files/` (inputs, from the data archive, and
intermediate files), the pretrained PolarCAP weights, and the manuscript sources.

---

## Environment

Everything runs in Python 3.10 with TensorFlow 2.16.2 and Keras 3.11.3
(`environment.yml`); no MATLAB is needed:

```bash
conda env create -f environment.yml
conda activate axialpolcap
```

Step 7 also needs SKHASH (Skoumal et al. 2024), cloned from
<https://code.usgs.gov/esc/SKHASH> and passed with `--skhash <path>/SKHASH/SKHASH.py`.
The transfer-learning runs of step 4 need the pretrained `PolarCAP.h5` of Chakraborty et
al. (2022) in `06-models/` (or set `POLARCAP_H5`).

Notes:

- The TensorFlow Metal plugin crashes during `model.predict` and fine-tuning on macOS.
  Inference and training scripts call `tf.config.set_visible_devices([], 'GPU')`.
- SKHASH's pure-Python grid search needs 1–3 GB per worker for composites with many
  polarities; `07_focal_mechanisms.py` uses four workers.

---

## Data availability

The seismic data were downloaded from the EarthScope Consortium Data Management Center
with network code `OO`. The continuously updated earthquake catalogue is at
<https://axialdd.ldeo.columbia.edu>. The composite focal-mechanism catalogue and polarity
determinations produced in this study are at
<http://axial.ocean.washington.edu/FocalMechanisms.html>. The input data of this
repository (`data_manifest.csv`) and the results of steps 5–8 (`results_manifest.csv`,
including the AxialPolCap polarities and the focal mechanisms) are archived on Zenodo:
DOI to be added on release.
Focal mechanisms are computed with SKHASH (Skoumal et al. 2024;
<https://code.usgs.gov/esc/SKHASH>).

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
