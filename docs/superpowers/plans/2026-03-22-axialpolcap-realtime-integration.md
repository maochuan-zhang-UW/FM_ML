# AxialPolCap Real-Time Focal Mechanism Integration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the trained AxialPolCap model into the real-time Axial Seamount focal mechanism pipeline as an automatic P-wave polarity picker, replacing manual picks.

**Architecture:** A self-contained `polarity_picker.py` module wraps the Keras model and exposes a single function `pick_polarity(waveform, p_sample)` that returns polarity, confidence, and entropy. The existing real-time pipeline calls this module at the step where it currently obtains polarity picks, then feeds results into HASH as before.

**Tech Stack:** Python 3.11, TensorFlow 2.13.1 / Keras 2.13.1, NumPy, SciPy — conda env `tf-2.14.0`

---

## Pre-work: Inspect the Real-Time Pipeline (do this first, on the other Mac)

Before writing any integration code, answer these questions by reading the pipeline source:

1. **What language/framework?** Python script, MATLAB, or mixed?
2. **Where do polarity picks enter?** Find the variable/file/function that currently supplies polarity values to HASH. That is the integration point.
3. **What is the waveform window?** How many samples? Sample rate? Is the P-arrival index already known at that point?
4. **What does HASH expect?** Polarity as `+1/-1`? Integer `0/1`? A specific file format (e.g., `.pol` text file)?
5. **Is the pipeline continuous (daemon) or event-triggered?** This determines whether the model should be loaded once at startup or per-event.

Document your answers as comments in the integration script (Task 3).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `01-scripts/application/polarity_picker.py` | **Create** | Self-contained module: load model, preprocess waveform, return polarity/confidence/entropy |
| `01-scripts/application/apply_to_catalog.py` | Reference only | Shows the existing inference pattern — do not modify |
| `06-models/PolarPicker_h5_20260308_154910.keras` | Use | Trained AxialPolCap model (already in git) |
| `<pipeline>/polarity_integration.py` | **Create** (on other Mac) | Thin adapter that calls `polarity_picker.py` and converts output to whatever HASH expects |

---

## Chunk 1: Self-Contained Polarity Picker Module

### Task 1: Clone repo and verify environment on the other Mac

**On the other Mac (terminal):**

- [ ] **Step 1: Clone the repo**

```bash
cd ~  # or wherever you keep projects
git clone https://github.com/maochuan-zhang-UW/FM_ML.git
cd FM_ML
```

- [ ] **Step 2: Create / activate the conda environment**

```bash
conda activate tf-2.14.0
# If the env doesn't exist yet on this machine:
# conda create -n tf-2.14.0 python=3.11
# conda activate tf-2.14.0
# pip install tensorflow==2.13.1 scipy numpy scikit-learn
```

- [ ] **Step 3: Verify the model loads**

```python
# Run interactively: python
from tensorflow import keras
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')   # avoid Metal crash on macOS
model = keras.models.load_model(
    "06-models/PolarPicker_h5_20260308_154910.keras"
)
model.summary()
# Should show encoder + decoder + classifier heads, ~64-sample input
```

Expected: model summary prints without error.

- [ ] **Step 4: Commit nothing** — this is setup only.

---

### Task 2: Write `polarity_picker.py`

**File:** `01-scripts/application/polarity_picker.py`

This module is the single source of truth for AxialPolCap inference. It must be importable by any script on this machine.

- [ ] **Step 1: Write the module**

```python
"""
polarity_picker.py
------------------
Self-contained AxialPolCap polarity picker.

Usage
-----
from polarity_picker import PolarityPicker

picker = PolarityPicker("06-models/PolarPicker_h5_20260308_154910.keras")
result = picker.pick(waveform_200samp, p_sample_index)
# result = {"polarity": 1, "confidence": 0.97, "entropy": 0.08}
"""

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")   # avoid Metal crash on macOS

from tensorflow import keras

WINDOW = 64          # samples expected by the model
HALF   = WINDOW // 2  # 32 samples each side of P-arrival


class PolarityPicker:
    """Load once, call many times."""

    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path, safe_mode=False)

    # ------------------------------------------------------------------
    def pick(self, waveform: np.ndarray, p_sample: int) -> dict:
        """
        Parameters
        ----------
        waveform : 1-D float array, any length (≥64 samples), 200 Hz
        p_sample : integer index of P-arrival within waveform

        Returns
        -------
        dict with keys: polarity (+1 or -1), confidence (0-1), entropy (≥0)
        Returns None if the window cannot be extracted.
        """
        start = p_sample - HALF
        end   = p_sample + HALF
        if start < 0 or end > len(waveform):
            return None

        window = waveform[start:end].astype(np.float32)
        mx = np.max(np.abs(window))
        if mx == 0:
            return None
        window /= mx                                # max-normalize

        X = window.reshape(1, WINDOW, 1)            # (1, 64, 1)
        y_raw  = self.model.predict(X, verbose=0)
        y_prob = y_raw[1] if isinstance(y_raw, (list, tuple)) else y_raw
        y_prob = np.asarray(y_prob).flatten()       # shape (2,)

        polarity   = 1 if np.argmax(y_prob) == 1 else -1
        confidence = float(np.max(y_prob))
        entropy    = float(-np.sum(y_prob * np.log(y_prob + 1e-12)))

        return {"polarity": polarity, "confidence": confidence, "entropy": entropy}

    # ------------------------------------------------------------------
    def pick_batch(self, windows: np.ndarray) -> list[dict]:
        """
        Parameters
        ----------
        windows : float array of shape (N, 64) — already cropped & normalized

        Returns
        -------
        list of N dicts
        """
        X = windows.reshape(-1, WINDOW, 1).astype(np.float32)
        y_raw  = self.model.predict(X, verbose=0)
        y_prob = y_raw[1] if isinstance(y_raw, (list, tuple)) else y_raw
        y_prob = np.asarray(y_prob)                 # (N, 2)

        results = []
        for p in y_prob:
            polarity   = 1 if np.argmax(p) == 1 else -1
            confidence = float(np.max(p))
            entropy    = float(-np.sum(p * np.log(p + 1e-12)))
            results.append({"polarity": polarity, "confidence": confidence, "entropy": entropy})
        return results
```

- [ ] **Step 2: Quick smoke test (no test framework needed)**

```bash
conda activate tf-2.14.0
cd FM_ML
python - <<'EOF'
import numpy as np
import sys
sys.path.insert(0, "01-scripts/application")
from polarity_picker import PolarityPicker

picker = PolarityPicker("06-models/PolarPicker_h5_20260308_154910.keras")

# synthetic sine wave, P-arrival at sample 100
wave = np.sin(np.linspace(0, 4*np.pi, 200)).astype(np.float32)
result = picker.pick(wave, p_sample=100)
print("Result:", result)
assert result is not None
assert result["polarity"] in (-1, 1)
assert 0.0 <= result["confidence"] <= 1.0
print("PASS")
EOF
```

Expected output: `Result: {'polarity': ..., 'confidence': ..., 'entropy': ...}` then `PASS`.

- [ ] **Step 3: Commit**

```bash
git add 01-scripts/application/polarity_picker.py
git commit -m "feat: add self-contained PolarityPicker module for real-time use"
```

---

## Chunk 2: Pipeline Integration (on the other Mac)

> **Note:** Steps in this chunk require reading the actual real-time pipeline code on the other Mac first (see Pre-work above). The exact file path and variable names below are placeholders — replace with real ones after inspection.

### Task 3: Write the pipeline adapter

**File:** `<pipeline_dir>/polarity_integration.py`

This is the thin glue layer between the real-time pipeline and `PolarityPicker`. It should not contain any ML logic — just call the picker and reformat the output.

- [ ] **Step 1: Identify the integration point in the pipeline**

Find the function/script where polarity picks are currently assigned. Look for:
- Variables named `polarity`, `first_motion`, `pol`, `Po_*`
- File writes to `.pol` or similar HASH input files
- Any call to a manual-pick lookup or cross-correlation polarity function

- [ ] **Step 2: Write the adapter**

```python
"""
polarity_integration.py
-----------------------
Adapter between the real-time Axial pipeline and AxialPolCap.
Replace <FM_ML_ROOT> with the actual path on this machine.
"""
import sys
FM_ML_ROOT = "/path/to/FM_ML"          # ← set this
sys.path.insert(0, f"{FM_ML_ROOT}/01-scripts/application")

from polarity_picker import PolarityPicker

# Load once at module import (model is ~300 KB, fast to load)
_picker = PolarityPicker(f"{FM_ML_ROOT}/06-models/PolarPicker_h5_20260308_154910.keras")


def get_polarity(waveform, p_sample, confidence_threshold=0.70):
    """
    Drop-in replacement for whatever currently provides polarity picks.

    Parameters
    ----------
    waveform          : np.ndarray, shape (N,), 200 Hz, Z-component
    p_sample          : int, index of P-arrival in waveform
    confidence_threshold : float, picks below this are returned as 0 (undecided)

    Returns
    -------
    int : +1, -1, or 0 (low-confidence / undecided)
    """
    result = _picker.pick(waveform, p_sample)
    if result is None:
        return 0
    if result["confidence"] < confidence_threshold:
        return 0
    return result["polarity"]
```

- [ ] **Step 3: Replace the old polarity call in the pipeline**

Find the line(s) that currently assign polarity (e.g., from a lookup table or cross-correlation). Replace with:

```python
from polarity_integration import get_polarity
# ...
pol = get_polarity(waveform_z, p_sample_index)
```

- [ ] **Step 4: Validate on a known historical event**

Pick an event from the catalog where you know the manual polarity. Run the pipeline on that event and confirm the DL prediction matches.

```python
# Example check (adapt to your pipeline's data format):
expected = +1   # from catalog
pol = get_polarity(waveform, p_sample)
print(f"Expected: {expected}, Got: {pol}")
assert pol == expected, "Mismatch on validation event"
```

- [ ] **Step 5: Commit the adapter**

```bash
git add polarity_integration.py
git commit -m "feat: wire AxialPolCap polarity picker into real-time FM pipeline"
```

---

## Chunk 3: Confidence Threshold Tuning & Monitoring

### Task 4: Choose the confidence threshold

The `confidence_threshold=0.70` default in Task 3 is a starting point. Too low → noisy picks contaminate HASH; too high → too few picks per event → HASH cannot compute a mechanism.

- [ ] **Step 1: Run on a batch of historical events with known polarities**

Use `apply_to_catalog.py` (already in the repo) as a reference. It outputs `[GT, Prediction, Confidence, Entropy]` per pick.

- [ ] **Step 2: Plot accuracy vs. threshold**

```python
import numpy as np
import matplotlib.pyplot as plt

# Load saved predictions (from apply_to_catalog output)
# confidences, correct = ...  (build from Po field column 2 == column 0)

thresholds = np.linspace(0.5, 0.99, 50)
accs, coverages = [], []
for t in thresholds:
    mask = confidences >= t
    accs.append((correct[mask]).mean() if mask.sum() > 0 else np.nan)
    coverages.append(mask.mean())

plt.plot(thresholds, accs, label="Accuracy")
plt.plot(thresholds, coverages, label="Coverage (fraction kept)")
plt.xlabel("Confidence threshold")
plt.legend()
plt.savefig("03-figs/threshold_curve.png")
```

- [ ] **Step 3: Pick threshold** — target ≥95% accuracy while keeping ≥70% coverage. Update `confidence_threshold` in `polarity_integration.py`.

- [ ] **Step 4: Commit**

```bash
git add 03-figs/threshold_curve.png
git commit -m "docs: add confidence threshold analysis curve"
```

---

## Rollout Notes

| Concern | Mitigation |
|---------|-----------|
| macOS Metal GPU crash | `tf.config.set_visible_devices([], 'GPU')` already in `polarity_picker.py` |
| Model not found on new machine | Set `FM_ML_ROOT` in `polarity_integration.py`; verify path before running |
| HASH needs ≥3 polarity picks | Set threshold conservatively (0.60–0.70) until you have statistics |
| Real-time latency | Model is ~300 KB; single-pick inference < 50 ms on CPU; batch per event at once |
| Model variant (Lambda layers) | `safe_mode=False` already in `PolarityPicker.__init__` |

---

## Open Questions (resolve before Task 3)

1. Does the real-time pipeline run as a Python daemon, a cron job, or a MATLAB script?
2. What exact format does HASH expect for polarity input? (`+`/`-` characters in a text file? Integer codes?)
3. Is a waveform Z-component and P-sample index already available at the integration point, or does preprocessing need to be added?
4. Is there a test/replay mode in the pipeline that lets you run it on historical data without triggering live outputs?
