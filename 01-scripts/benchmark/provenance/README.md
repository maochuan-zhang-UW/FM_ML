# Benchmark provenance

These are the scripts, as run, that applied the four published polarity
classifiers zero-shot to the Axial benchmark set (Fig. 3, Fig. S1, Table S2).
Each was run inside the clone of its model's own repository, with that
model's environment and pretrained weights, and added its predictions to the
benchmark `.mat` file in turn:

| Order | Script | Model and source | Field written |
|---|---|---|---|
| 1 | `1_ditingmotion_predict.py` | DiTingMotion (Zhao et al. 2023), DiTing-FOCALFLOW repository, `models/DiTingMotionJul.hdf5` | `PoML_W_{STA}` |
| 2 | `2_cfm_predict.py` | CFM (Messuti et al. 2023), `Network_CFM/CFM_with_timeshift.hdf5` | `CFM_W_{STA}` |
| 3 | `3_eqpolarity_predict.py` | EQPolarity (Chen et al. 2024), `models/best_weigths_Binary_Texas_Transfer10.h5` | `EQP_{STA}` |
| 4 | `4_polarcap_predict.py` | PolarCAP (Chakraborty et al. 2022), SAIPy `saved_models/PolarCAP.h5` | `PolCAP_{STA}` |

The result, `A_wave_dB15_DT_CFM_EQP_PolCAP.mat`, is in the data archive
(`02-data/raw/benchmark/`). `../score_benchmark.py` computes every number of
Fig. 3, Fig. S1 and Table S2 for the four models from it and reproduces the
published values exactly.

DiTingMotion was run on per-station files (`N_Po{STA}.mat`) and returns no
prediction for part of the set, so it is scored on 8,329 picks against 17,128
for the other three models.

The cross-correlation row of Fig. 3 and Table S2 comes from
`cc_baseline_H_compare_Man_CC.m` (MATLAB), which compares CC polarities with
the manual polarities on a separate set of about 1,000 events per station
(`D_man/D_manual_{STA}_CCPo.mat`). Those inputs are not in the data archive,
so the CC values are kept as constants in `figures/fig02_03_S1_S4.py`; its
average is a plain mean over stations.

The scripts keep the absolute paths of the original runs. The third-party
models are not redistributed here; SAIPy (PolarCAP) is GPL-3.0.
