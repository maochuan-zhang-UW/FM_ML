# Figure Manifest — AxialPolCap (MZhang_FM_ML_Final)

Generated: 2026-03-22
Source script: `01-scripts/make_manuscript_figures.py`
Output directory: `03-figs/`
Target: SRL (Seismological Research Letters), 300 DPI

---

| Fig | Status | Width (px) | Height (px) | DPI | Size (KB) | Caption summary |
|-----|--------|-----------|------------|-----|----------|----------------|
| 1  | ✅ PASS | 2643 | 1879 | 300 | 1170 | Bathymetric map of Axial Seamount: caldera, fissures, lava flows, OBS station locations |
| 2  | ✅ PASS | 5294 | 3034 | 300 | 2005 | (a) Template waveform examples at 7 stations; (b) noise waveforms from 2016–2017 |
| 3  | ✅ PASS | 1982 | 1643 | 300 | 295  | Examples of augmented waveforms (black) vs. template (red) across SNR range |
| 4  | ✅ PASS | 2784 | 1643 | 300 | 107  | Benchmark: station-wise polarity accuracy for DiTingMotion, CFM, EQPolarity, PolarCAP vs. CC |
| 5  | ✅ PASS | 2850 | 1446 | 300 | 163  | Autoencoder architecture (encoder + decoder + classifier head) |
| 6  | ✅ PASS | 2551 | 1643 | 300 | 93   | Accuracy comparison: unified model vs. LOSO vs. transfer learning across stations |
| 7  | ✅ PASS | 3034 | 2734 | 300 | 184  | SNR impact on accuracy: (a) newly trained model; (b) transfer learning model |
| 8  | ✅ PASS | 3034 | 2734 | 300 | 193  | P-wave timing shift sensitivity: newly trained vs. transfer learning model |
| 9  | ✅ PASS | 3034 | 2734 | 300 | 185  | Accuracy for models trained with σ=0.01 s and 0.02 s time shifts, tested at σ=0.01 s |
| 10 | ✅ PASS | 5734 | 1474 | 300 | 1012 | Random examples of conflicting CC vs. ML polarities per station (up to 10 per station) |
| 11 | ✅ PASS | 3334 | 2723 | 300 | 1089 | Spatial distribution of CC/ML polarity agreement (gray) vs. conflict (blue) per station |
| 12 | ✅ PASS | 3591 | 2590 | 300 | 1030 | Focal mechanism comparison: CC polarities (a–c) vs. AxialPolCap polarities (d–f) |
| 13 | ✅ PASS | 3631 | 1534 | 300 | 452  | Kagan angle histogram (a) and spatial distribution (b) for CC vs. AxialPolCap mechanisms |
| 14 | ✅ PASS | 3315 | 1446 | 300 | 199  | Schematic of the real-time focal mechanism pipeline |

---

## Validation Summary

- **Total figures:** 14 / 14
- **Passed:** 14
- **Failed:** 0
- **DPI:** All figures at 299.999 ≈ 300 DPI ✅
- **Empty data:** None detected ✅
- **Missing files:** None ✅

## Notes

- All figures generated 2026-03-20 by `make_manuscript_figures.py`
- Figure 2 is the largest (2.0 MB, 5294×3034 px) — check journal file size limits if submitting electronically
- Figure 6 is the smallest (93 KB) — confirm it is legible at print size (check at 100% zoom)
- Figures 1–2 have `@` extended attributes (macOS) — no impact on content
