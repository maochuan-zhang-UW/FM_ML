# Manuscript Structural Fixes — Design Spec

**Date:** 2026-03-24
**File:** `MZhang_FM_ML_Final.docx`
**Issues addressed:** S2, S3, S4 from manuscript review

---

## Issue S3 — New §3.3 "Selected Model Configuration"

**Location:** New subsection inserted at end of Section 3, immediately before the "4 Model application to Axial Seamount..." heading. Uses **Heading 2** style, matching §3.1 and §3.2.

**Performance figure source:** The ~96% figure is from the all-station model (original SNR + σ = 0.02 s augmentation) evaluated on the held-out test set with σ = 0.01 s shift, as reported in Section 3.2 / Figure 9. Use the same dataset description as Section 3.2.

**Figure 6 confirmed:** Figure 6 caption reads "Accuracy (%) comparison of three training strategies... (2) Accuracy of seven models leave-one-station-out (LOSO) training..." — correct figure for the LOSO citation.

**Full draft text:**

> Based on the systematic experiments described above, we selected a final model — referred to as AxialPolCap — using the following configuration: trained from scratch on merged data from all seven stations, using the original catalog SNR distribution, with a σ = 0.02 s P-wave time-shift augmentation applied during training. This configuration achieves the best balance between overall accuracy and robustness to P-wave pick timing uncertainty. Evaluated on the held-out test set with a σ = 0.01 s time shift, AxialPolCap achieves an average accuracy of approximately 96% across all stations, outperforming both the unaugmented newly trained model and all transfer-learning variants. Leave-one-station-out cross-validation of this configuration confirms that the model generalizes across stations without station-specific retraining, achieving 96–98% accuracy on unshifted waveforms for each held-out station (Figure 6). We apply this model to the full 2015–2021 Axial Seamount earthquake catalog in Section 4.

**Constraint:** No other content in Section 3 changes.

---

## Issue S2 — Section 4 Split into §4.1 and §4.2

Both subheadings use **Heading 2** style, matching §3.1–§3.3.

**§4.1 "Application to the 2015–2021 Axial Seamount Catalog"**

- Heading inserted immediately before the existing sentence: "We apply the optimized AxialPolCap model to seismic waveforms recorded at Axial Seamount..."
- All content from that sentence through the Figure 13 caption is §4.1 — unchanged.
- §4.1 ends after: "Figure 13. The histogram (a) and spatial distribution (b) of Kagan angle for focal mechanisms derived from CC and AxialPolcap polarities."

**§4.2 "Real-Time Monitoring Framework"**

- Heading inserted immediately before the sentence beginning: "Building on the validated AxialPolCap polarity pipeline, we have integrated a real-time focal mechanism estimation capability..."
- Figure 14 is already the last element of Section 4 (immediately after the real-time paragraph) — it stays in place. No figure renumbering required.
- Existing single paragraph is kept as Paragraph 1. Two new paragraphs are added after it.

**Full draft text for §4.2 (all three paragraphs):**

*Paragraph 1 — keep existing text as-is:*
> Building on the validated AxialPolCap polarity pipeline, we have integrated a real-time focal mechanism estimation capability into the existing Axial Seamount monitoring infrastructure. The near-real-time focal mechanism results are publicly accessible through the Axial Seamount Earthquake Catalog web portal (http://axial.ocean.washington.edu/), which provides hourly-updated earthquake detections and HYPOINVERSE locations for the Ocean Observatories Initiative (OOI) Regional Cabled Array. As new seismic data are acquired and processed, AxialPolCap automatically classifies P-wave first-motion polarities for each station-event pair, and composite focal mechanisms are computed and appended to the online catalog (Figure 14). This operational pipeline enables continuous, near-real-time monitoring of stress and faulting geometry at Axial Seamount, providing the community with timely access to source mechanism information during periods of heightened volcanic or seismic activity.

*Paragraph 2 — new, pipeline steps:*
> The pipeline operates as follows. Each hour, vertical-component waveforms are extracted for newly detected earthquakes using catalog P-wave arrival times and preprocessed identically to the training data: demeaned, detrended, bandpass filtered between 3 and 20 Hz, resampled to 100 Hz, and normalized by maximum absolute amplitude. AxialPolCap then classifies the first-motion polarity for each station–event pair, retaining only predictions with confidence ≥ 0.8 and entropy ≤ 0.2. The retained polarities are grouped into event clusters using hierarchical clustering within each hourly batch, and composite polarities are computed for each cluster. These are passed to SKHASH (Skoumal et al., 2024) for focal mechanism inversion, and the resulting solutions are appended to the public catalog within the same hourly update cycle.

*Paragraph 3 — new, operational context:*
> The pipeline has been running continuously since March 2026 and requires no manual intervention between updates. By providing probabilistic polarity estimates and confidence-weighted inputs to SKHASH, the framework supports robust focal mechanism determination even for small-magnitude events with limited station coverage. This capability is particularly valuable during periods of elevated seismicity, such as pre-eruptive swarms, when rapid, automated source characterization can inform real-time hazard assessment and guide field response at active submarine volcanic systems.

---

## Issue S4 — LOSO Sentence in Abstract

**Location:** Insert as a new sentence immediately after the sentence ending "...both exceeding the ~90% inter-analyst consistency reported for manual picking (Hardebeck & Shearer, 2002)."

**Exact text to insert:**
> Leave-one-station-out cross-validation, in which the model is trained on six stations and tested on the seventh in turn, yields 96–98% accuracy on unshifted waveforms across all held-out stations, confirming that the model generalizes without station-specific retraining.

**Source:** Confirmed by author from Figure 6 LOSO results (all seven held-out stations).

**Constraint:** No other abstract changes.

---

## Execution order

1. **S4** — Add LOSO sentence to Abstract. Isolated, no dependencies.
2. **S3** — Insert §3.3 heading + draft paragraph. Word heading styles auto-update section numbering; no manual renumbering needed.
3. **S2** — Insert §4.1 and §4.2 headings; add two new paragraphs after existing §4.2 Paragraph 1.
