# Manuscript All-Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply all reviewed edits to `MZhang_FM_ML_Final.docx` in a single comprehensive pass — covering previously identified typo/reference fixes, the abstract metric clarification (S1), and three structural additions (S2 §4.1/§4.2, S3 §3.3, S4 LOSO abstract sentence).

**Architecture:** All edits are direct string operations on `word/document.xml` inside the .docx ZIP. All changes are applied sequentially to a single in-memory XML string, then written back once. This avoids the state-loss problem caused by multiple separate write operations.

**Tech Stack:** Python 3, `zipfile` (stdlib). No third-party packages required.

---

## Chunk 1: Pre-flight verification

### Task 1: Verify all anchor strings are present and reachable

- [ ] **Step 1: Run pre-flight checks**

```python
# save as preflight.py, run: python3 preflight.py
import zipfile

with zipfile.ZipFile('MZhang_FM_ML_Final.docx', 'r') as z:
    xml = z.read('word/document.xml').decode('utf-8')

anchors = {
    # Typo fixes
    'kno  wledge':        'kno  wledge',
    'atda':               'are more consistent atda each station',
    'double space':       'For those that  disagreed',
    # Reference fixes
    'Arnulf 2014':        'Arnulf, Harding, A. J.',
    'Dziak':              'Dziak, &amp; Fox, C. G.',
    'Ross':               'Ross, Meier, M. A., &amp; Hauksson, E.',
    'Wilcock 2016':       'Wilcock, Dziak, R. P.',
    'Inprep':             'Zhang et al., Inprep',
    'XXX':                'We thank XXX that helped improve the manuscript, Ian Stone for thorough and constructive discussions.',
    # Abstract S1 (metric clarification) — old text to replace
    'S1 old text':        'When evaluated on test data with larger time shifts, AxialPolCap achieves approximately 96% accuracy compared to 80-85% for previous deep learning models, outperforming the manually assigned polarity accuracy of 90% (Hardebeck &amp; Shearer, 2002) and cross-correlation-derived accuracy of 92%. Application of our methods to continuous seismic data recorded at Axial Seamount between 2015 and 2021 yields a consistent catalog of P-wave polarities for determining focal mechanisms, facilitating interpretation of recent volcanic activity at this active submarine volcano.',
    # Note: F1 (LOSO insertion anchor) is NOT pre-flight checked — it only exists
    # after Section E's replacement runs. Pre-flight only checks the old text (E's source).
    # Section 3.2 closing paragraph
    'S3 anchor':          'We apply this optimized model to continuous seismic data recorded at Axial Seamount between 2015 and 2021',
    # Section 4 split anchors
    'S2 §4.1 anchor':     'We apply the optimized AxialPolCap model to seismic waveforms recorded at Axial Seamount between January 2015',
    'S2 §4.2 anchor':     'Building on the validated AxialPolCap polarity pipeline, we have integrated a real-time focal mechanism',
    'S2 Fig14 anchor':    'Figure 14. Schematic overview of the real-time focal mechanism pipeline.',
}

all_ok = True
for label, anchor in anchors.items():
    found = anchor in xml
    if not found:
        all_ok = False
    print(f"{'✓' if found else '✗ MISSING':<12} {label}")

print()
print("All anchors found." if all_ok else "STOP — fix missing anchors before proceeding.")
```

Expected: all lines show `✓`. If any show `✗ MISSING`, stop and investigate before running the main script.

---

## Chunk 2: Apply all edits in a single pass

### Task 2: Run the comprehensive edit script

- [ ] **Step 1: Create a timestamped backup**

```bash
cp MZhang_FM_ML_Final.docx "MZhang_FM_ML_Final_before_alledits_$(date +%Y%m%d_%H%M%S).docx"
echo "Backup created."
```

- [ ] **Step 2: Run the comprehensive edit script**

```python
# save as apply_all_edits.py, run: python3 apply_all_edits.py
import zipfile, shutil

DOCX = 'MZhang_FM_ML_Final.docx'
TMP  = '_alledits_tmp.docx'

# ─── helpers ───────────────────────────────────────────────────────────────
def heading_xml(para_id, text):
    return (
        f'<w:p w14:paraId="{para_id}">'
        '<w:pPr><w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>'
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        '<w:b/><w:bCs/><w:highlight w:val="white"/></w:rPr></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        f'<w:b/><w:bCs/><w:highlight w:val="white"/></w:rPr><w:t>{text}</w:t></w:r></w:p>'
    )

def body_xml(para_id, text):
    return (
        f'<w:p w14:paraId="{para_id}">'
        '<w:pPr><w:spacing w:after="0" w:line="276" w:lineRule="auto"/><w:jc w:val="both"/>'
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        '<w:highlight w:val="white"/></w:rPr></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        '<w:highlight w:val="white"/></w:rPr>'
        f'<w:t xml:space="preserve">{text}</w:t></w:r></w:p>'
    )

# ─── load ──────────────────────────────────────────────────────────────────
with zipfile.ZipFile(DOCX, 'r') as z:
    all_files = z.namelist()
    xml = z.read('word/document.xml').decode('utf-8')

# ══════════════════════════════════════════════════════════════════════════
# SECTION A — Typo & formatting fixes
# ══════════════════════════════════════════════════════════════════════════

# A1. kno  wledge → knowledge
assert 'kno  wledge' in xml
xml = xml.replace('kno  wledge', 'knowledge')
print("A1 ✓  knowledge")

# A2. atda → at
assert 'atda each station' in xml
xml = xml.replace('atda each station', 'at each station')
print("A2 ✓  atda")

# A3. double space before "disagreed"
assert 'For those that  disagreed' in xml
xml = xml.replace('For those that  disagreed', 'For those that disagreed')
print("A3 ✓  double space")

# ══════════════════════════════════════════════════════════════════════════
# SECTION B — Reference fixes (missing first-author initials)
# ══════════════════════════════════════════════════════════════════════════

# B1. Arnulf 2014
assert 'Arnulf, Harding, A. J.' in xml
xml = xml.replace('Arnulf, Harding, A. J.', 'Arnulf, A. F., Harding, A. J.')
print("B1 ✓  Arnulf 2014")

# B2. Dziak 1999
assert 'Dziak, &amp; Fox, C. G.' in xml
xml = xml.replace('Dziak, &amp; Fox, C. G.', 'Dziak, R. P., &amp; Fox, C. G.')
print("B2 ✓  Dziak")

# B3. Ross 2018
assert 'Ross, Meier, M. A., &amp; Hauksson, E.' in xml
xml = xml.replace('Ross, Meier, M. A., &amp; Hauksson, E.', 'Ross, Z. E., Meier, M. A., &amp; Hauksson, E.')
print("B3 ✓  Ross")

# B4. Wilcock 2016
assert 'Wilcock, Dziak, R. P.' in xml
xml = xml.replace('Wilcock, Dziak, R. P.', 'Wilcock, W. S. D., Dziak, R. P.')
print("B4 ✓  Wilcock 2016")

# ══════════════════════════════════════════════════════════════════════════
# SECTION C — Citation format fix
# ══════════════════════════════════════════════════════════════════════════

# C1. Inprep → in prep.
assert 'Zhang et al., Inprep' in xml
xml = xml.replace('Zhang et al., Inprep', 'Zhang et al., in prep.')
print("C1 ✓  Inprep → in prep.")

# ══════════════════════════════════════════════════════════════════════════
# SECTION D — Acknowledgments placeholder
# ══════════════════════════════════════════════════════════════════════════

# D1. XXX → Ian Stone (merge duplicate mention)
OLD_ACK = 'We thank XXX that helped improve the manuscript, Ian Stone for thorough and constructive discussions.'
NEW_ACK = 'We thank Ian Stone for thorough and constructive discussions and for helping to improve the manuscript.'
assert OLD_ACK in xml
xml = xml.replace(OLD_ACK, NEW_ACK)
print("D1 ✓  XXX → Ian Stone")

# ══════════════════════════════════════════════════════════════════════════
# SECTION E — Abstract S1: distinguish test-set accuracy from catalog agreement
# ══════════════════════════════════════════════════════════════════════════

OLD_ABS = (
    'When evaluated on test data with larger time shifts, AxialPolCap achieves approximately 96% accuracy '
    'compared to 80-85% for previous deep learning models, outperforming the manually assigned polarity accuracy '
    'of 90% (Hardebeck &amp; Shearer, 2002) and cross-correlation-derived accuracy of 92%. Application of our '
    'methods to continuous seismic data recorded at Axial Seamount between 2015 and 2021 yields a consistent '
    'catalog of P-wave polarities for determining focal mechanisms, facilitating interpretation of recent '
    'volcanic activity at this active submarine volcano.'
)
NEW_ABS = (
    'On a held-out synthetic test set with realistic P-wave timing perturbations, AxialPolCap achieves '
    'approximately 96% polarity accuracy, compared to 80\u201385% for existing deep-learning models and 92% '
    'for cross-correlation, both exceeding the ~90% inter-analyst consistency reported for manual picking '
    '(Hardebeck &amp; Shearer, 2002). When applied to the independent 2015\u20132021 Axial Seamount catalog '
    '\u2014 under realistic noise and pick-timing conditions \u2014 AxialPolCap polarities agree with '
    'cross-correlation-derived polarities for ~88% of picks, and the resulting focal mechanisms match '
    'cross-correlation-based solutions with a median Kagan angle of ~13\u00b0, supporting the reliability '
    'of the approach for large-scale and real-time applications.'
)
assert OLD_ABS in xml
xml = xml.replace(OLD_ABS, NEW_ABS)
print("E1 ✓  Abstract S1 metric clarification")

# ══════════════════════════════════════════════════════════════════════════
# SECTION F — Abstract S4: add LOSO sentence
# (inserted after the now-present Hardebeck & Shearer 2002 citation in the new abstract)
# ══════════════════════════════════════════════════════════════════════════

LOSO_ANCHOR_OLD = 'Hardebeck &amp; Shearer, 2002). When applied to the independent'
LOSO_ANCHOR_NEW = (
    'Hardebeck &amp; Shearer, 2002). '
    'Leave-one-station-out cross-validation, in which the model is trained on six stations '
    'and tested on the seventh in turn, yields 96\u201398% accuracy on unshifted waveforms '
    'across all held-out stations, confirming that the model generalizes without '
    'station-specific retraining. '
    'When applied to the independent'
)
assert LOSO_ANCHOR_OLD in xml, "LOSO anchor not found — check that E1 ran first"
xml = xml.replace(LOSO_ANCHOR_OLD, LOSO_ANCHOR_NEW)
print("F1 ✓  Abstract S4 LOSO sentence")

# ══════════════════════════════════════════════════════════════════════════
# SECTION G — S3: Insert §3.3 "Selected Model Configuration"
# (after Section 3.2 closing paragraph)
# ══════════════════════════════════════════════════════════════════════════

S33_HEADING = heading_xml('AA110001', '3.3 Selected Model Configuration')

S33_BODY_TEXT = (
    'Based on the systematic experiments described above, we selected a final model \u2014 referred to as '
    'AxialPolCap \u2014 using the following configuration: trained from scratch on merged data from all seven '
    'stations, using the original catalog SNR distribution, with a \u03c3\u202f=\u202f0.02\u202fs P-wave '
    'time-shift augmentation applied during training. This configuration achieves the best balance between '
    'overall accuracy and robustness to P-wave pick timing uncertainty. Evaluated on the held-out test set '
    'with a \u03c3\u202f=\u202f0.01\u202fs time shift, AxialPolCap achieves an average accuracy of '
    'approximately 96% across all stations, outperforming both the unaugmented newly trained model and all '
    'transfer-learning variants. Leave-one-station-out cross-validation of this configuration confirms that '
    'the model generalizes across stations without station-specific retraining, achieving 96\u201398% accuracy '
    'on unshifted waveforms for each held-out station (Figure\u202f6). We apply this model to the full '
    '2015\u20132021 Axial Seamount earthquake catalog in Section\u202f4.'
)
S33_BODY = body_xml('AA110002', S33_BODY_TEXT)

S33_ANCHOR = 'We apply this optimized model to continuous seismic data recorded at Axial Seamount between 2015 and 2021'
assert S33_ANCHOR in xml
idx = xml.find(S33_ANCHOR)
para_end = xml.find('</w:p>', idx) + 6
assert para_end > 6, "Paragraph end not found"
xml = xml[:para_end] + S33_HEADING + S33_BODY + xml[para_end:]
print("G1 ✓  §3.3 heading and paragraph inserted")

# ══════════════════════════════════════════════════════════════════════════
# SECTION H — S2: Split Section 4 into §4.1 and §4.2
# ══════════════════════════════════════════════════════════════════════════

H41 = heading_xml('AA220001', '4.1 Application to the 2015\u20132021 Axial Seamount Catalog')
H42 = heading_xml('AA220002', '4.2 Real-Time Monitoring Framework')

P2_TEXT = (
    'The pipeline operates as follows. Each hour, vertical-component waveforms are extracted for newly '
    'detected earthquakes using catalog P-wave arrival times and preprocessed identically to the training data: '
    'demeaned, detrended, bandpass filtered between 3 and 20\u202fHz, resampled to 100\u202fHz, and normalized '
    'by maximum absolute amplitude. AxialPolCap then classifies the first-motion polarity for each '
    'station\u2013event pair, retaining only predictions with confidence \u2265\u202f0.8 and entropy '
    '\u2264\u202f0.2. The retained polarities are grouped into event clusters using hierarchical clustering '
    'within each hourly batch, and composite polarities are computed for each cluster. These are passed to '
    'SKHASH (Skoumal et al., 2024) for focal mechanism inversion, and the resulting solutions are appended '
    'to the public catalog within the same hourly update cycle.'
)
P3_TEXT = (
    'The pipeline has been running continuously since March 2026 and requires no manual intervention between '
    'updates. By providing probabilistic polarity estimates and confidence-weighted inputs to SKHASH, the '
    'framework supports robust focal mechanism determination even for small-magnitude events with limited '
    'station coverage. This capability is particularly valuable during periods of elevated seismicity, such '
    'as pre-eruptive swarms, when rapid, automated source characterization can inform real-time hazard '
    'assessment and guide field response at active submarine volcanic systems.'
)
P2 = body_xml('AA220003', P2_TEXT)
P3 = body_xml('AA220004', P3_TEXT)

# H1: Insert §4.1 heading before Section 4 opening paragraph
A41 = 'We apply the optimized AxialPolCap model to seismic waveforms recorded at Axial Seamount between January 2015'
assert A41 in xml
idx = xml.find(A41)
para_start = xml.rfind('<w:p ', 0, idx)
assert para_start != -1
xml = xml[:para_start] + H41 + xml[para_start:]
print("H1 ✓  §4.1 heading inserted")

# H2: Insert §4.2 heading before real-time paragraph (re-search after H1 shifted positions)
A42 = 'Building on the validated AxialPolCap polarity pipeline, we have integrated a real-time focal mechanism'
assert A42 in xml
idx = xml.find(A42)
para_start = xml.rfind('<w:p ', 0, idx)
assert para_start != -1
xml = xml[:para_start] + H42 + xml[para_start:]
print("H2 ✓  §4.2 heading inserted")

# H3: Append P2 and P3 after Figure 14 caption paragraph
A_FIG14 = 'Figure 14. Schematic overview of the real-time focal mechanism pipeline.'
assert A_FIG14 in xml
idx = xml.find(A_FIG14)
para_end = xml.find('</w:p>', idx) + 6
assert para_end > 6
xml = xml[:para_end] + P2 + P3 + xml[para_end:]
print("H3 ✓  §4.2 paragraphs 2 and 3 appended")

# ─── write ─────────────────────────────────────────────────────────────────
with zipfile.ZipFile(DOCX, 'r') as zin, \
     zipfile.ZipFile(TMP, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in all_files:
        zout.writestr(item,
            xml.encode('utf-8') if item == 'word/document.xml' else zin.read(item))

shutil.move(TMP, DOCX)
print("\nAll edits written to", DOCX)
```

- [ ] **Step 3: Verify all edits applied correctly**

```python
# save as verify_all.py, run: python3 verify_all.py
import zipfile
from xml.etree import ElementTree as ET

ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
with zipfile.ZipFile('MZhang_FM_ML_Final.docx', 'r') as z:
    xml = z.read('word/document.xml').decode('utf-8')
    root = ET.parse(z.open('word/document.xml')).getroot()

paras = []
for p in root.iter(ns + 'p'):
    t = ''.join(n.text for r in p.iter(ns + 'r') for n in r.iter(ns + 't') if n.text)
    if t.strip(): paras.append(t.strip())
full = '\n'.join(paras)

checks = {
    # Typo fixes
    'A1 knowledge':           ('kno  wledge' not in xml,     'kno  wledge' not in full),
    'A2 atda gone':           ('atda each station' not in xml, True),
    'A3 double space gone':   ('that  disagreed' not in xml,  True),
    # References
    'B1 Arnulf A.F.':         ('Arnulf, A. F., Harding' in xml, True),
    'B2 Dziak R.P.':          ('Dziak, R. P., &amp; Fox' in xml, True),
    'B3 Ross Z.E.':           ('Ross, Z. E., Meier' in xml, True),
    'B4 Wilcock W.S.D.':      ('Wilcock, W. S. D., Dziak' in xml, True),
    # Citation
    'C1 in prep.':            ('Inprep' not in xml, True),
    # Acknowledgments
    'D1 Ian Stone':           ('We thank Ian Stone for thorough' in xml, True),
    'D1 XXX gone':            ('We thank XXX' not in xml, True),
    # Abstract S1
    'E1 new abstract':        ('held-out synthetic test set' in full, True),
    'E1 old abstract gone':   ('When evaluated on test data with larger time shifts' not in full, True),
    # Abstract S4
    'F1 LOSO sentence':       ('Leave-one-station-out cross-validation, in which the model' in full, True),
    # Section 3.3
    'G1 §3.3 heading':        ('3.3 Selected Model Configuration' in full, True),
    'G1 §3.3 body':           ('trained from scratch on merged data from all seven stations' in full, True),
    # Section 4 split
    'H1 §4.1 heading':        ('4.1 Application to the 2015' in full, True),
    'H2 §4.2 heading':        ('4.2 Real-Time Monitoring Framework' in full, True),
    'H3 pipeline para':       ('The pipeline operates as follows' in full, True),
    'H3 operational para':    ('The pipeline has been running continuously since March 2026' in full, True),
}

all_ok = True
for label, (check_xml, _) in checks.items():
    ok = check_xml
    if not ok: all_ok = False
    print(f"{'✓' if ok else '✗ FAILED':<12} {label}")

print()
print("All checks passed." if all_ok else "FAILURES DETECTED — do not commit.")
```

Expected: all lines show `✓`.

- [ ] **Step 4: Commit**

```bash
git add MZhang_FM_ML_Final.docx
git commit -m "manuscript: apply all reviewed edits (typos, refs, S1/S2/S3/S4)"
```
