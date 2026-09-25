#!/usr/bin/env python3
"""Apply PolarPicker_h5_20260308_154910 to the 2015-2021 Axial catalog."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_fm5_ml")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "06-models" / "PolarPicker_h5_20260308_154910.keras"
DEFAULT_INPUT = Path(
    "/Users/mcZhang/Documents/GitHub/FM_ML_bk/02-data/A_wave_2015_2022CC.mat"
)
DEFAULT_OUTPUT = ROOT / "02-data" / "A_wave_2015_2021_h5_conf80_predictions.mat"
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]


def norm(x: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(x), axis=1, keepdims=True)
    max_val[max_val == 0] = 1
    return x / max_val


def as_events(raw) -> np.ndarray:
    return np.atleast_1d(raw).ravel()


def get_manual_polarity(event, station: str) -> int:
    value = np.asarray(getattr(event, f"Po_{station}", []), dtype=float).ravel()
    if value.size == 0:
        return 0
    return int(np.sign(value[0]))


def collect_station_waveforms(events: np.ndarray, station: str):
    waveforms = []
    indices = []
    manual = []
    w_key = f"W_{station}"

    for i, event in enumerate(events):
        if not hasattr(event, w_key):
            continue
        waveform = np.asarray(getattr(event, w_key), dtype=float).ravel()
        if waveform.size != 64:
            continue
        waveforms.append(waveform)
        indices.append(i)
        manual.append(get_manual_polarity(event, station))

    if not waveforms:
        return np.empty((0, 64, 1), dtype=np.float32), [], []

    x = np.asarray(waveforms, dtype=np.float32).reshape((-1, 64, 1))
    return norm(x), indices, manual


def apply_predictions(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    confidence_threshold: float,
    batch_size: int,
) -> None:
    tf.config.set_visible_devices([], "GPU")

    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, safe_mode=False)

    print(f"Loading catalog: {input_path}")
    data = loadmat(str(input_path), squeeze_me=True, struct_as_record=False)
    events = as_events(data["Felix"])
    print(f"Events: {len(events)}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")

    total_waveforms = 0
    retained = 0
    manual_nonzero = 0
    correct_retained = 0

    for station in STATIONS:
        x, indices, manual = collect_station_waveforms(events, station)
        if x.size == 0:
            print(f"{station}: no valid 64-sample waveforms")
            continue

        y_raw = model.predict(x, batch_size=batch_size, verbose=0)
        y_prob = y_raw[1] if isinstance(y_raw, (list, tuple)) else y_raw
        y_bin = np.argmax(y_prob, axis=1)
        y_pred = np.where(y_bin == 0, -1, 1)
        confidence = np.max(y_prob, axis=1)
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-12), axis=1)

        station_retained = 0
        station_manual_nonzero = 0
        station_correct_retained = 0

        for event_idx, gt, pred, conf, ent in zip(indices, manual, y_pred, confidence, entropy):
            pred_conf80 = int(pred) if conf >= confidence_threshold else 0
            setattr(
                events[event_idx],
                f"Po_{station}",
                np.array([gt, pred_conf80, conf, ent], dtype=np.float32),
            )
            station_retained += int(pred_conf80 != 0)
            station_manual_nonzero += int(gt != 0)
            station_correct_retained += int(gt != 0 and pred_conf80 != 0 and gt == pred_conf80)

        total_waveforms += len(indices)
        retained += station_retained
        manual_nonzero += station_manual_nonzero
        correct_retained += station_correct_retained

        accuracy = station_correct_retained / station_retained if station_retained else np.nan
        print(
            f"{station}: waveforms={len(indices)}, retained={station_retained}, "
            f"manual_nonzero={station_manual_nonzero}, retained_accuracy={accuracy:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_path), {"Felix": events}, format="5", do_compression=True)
    print(f"Saved: {output_path}")
    print(f"Total station waveforms: {total_waveforms}")
    print(f"Retained ML polarities >= threshold: {retained}")
    print(f"Manual nonzero polarities: {manual_nonzero}")
    total_accuracy = correct_retained / retained if retained else np.nan
    print(f"Retained accuracy where manual polarity exists: {total_accuracy:.4f}")
    print("Po field format: [manual polarity, ML polarity conf>=threshold else 0, confidence, entropy]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    apply_predictions(
        model_path=Path(args.model),
        input_path=Path(args.input),
        output_path=Path(args.output),
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
