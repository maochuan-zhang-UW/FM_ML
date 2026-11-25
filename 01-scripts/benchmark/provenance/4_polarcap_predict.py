#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the official pretrained PolarCAP model (SAIPy, Chakraborty et al. 2022/2024)
zero-shot on the 7-station Axial Seamount benchmark dataset, following the same
per-station .mat chaining convention as DiTing-FOCALFLOW / CFM / eqpolarity.

PolarCAP preprocessing (confirmed from saipy/data/base.py::get_polarcap_data and
saipy/models/polarcap.py in this folder's cloned SAIPy repo):
  - 64-sample window centered on the P arrival: data[p_arr-32 : p_arr+32]
  - The P arrival in this project's raw 200-sample/200 Hz window is at index
    ~100 (verified empirically: mean positive- vs negative-polarity waveforms,
    normalized per-trace, are ~0 and indistinguishable up to index ~99, then
    sharply diverge with opposite sign starting at index 100-101, consistent
    across AS1/CC1/EC2). An earlier assumption of index 64 (inferred from
    Stone_polarPicker's own window choice) was wrong and left PolarCAP looking
    at almost pure pre-arrival noise -- this is what actually matters.
  - Normalize by per-trace max absolute value (polarcap.py's own norm()).
  - model.predict returns [decoder_output, class_probs]; argmax(class_probs)
    gives 0=Negative(down), 1=Positive(up).
"""
import os
import sys
import argparse

import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saipy.models.polarcap import PolarCAP

parser = argparse.ArgumentParser(
    description='Run official pretrained PolarCAP on .mat file waveforms')
parser.add_argument(
    '--wdir',
    type=str,
    default='/Users/mczhang/Documents/GitHub/FM5_ML/',
    help='Base directory containing 02-data/F_ML/A_wave_dB15_DT_CFM_EQP_modified.mat')
parser.add_argument(
    '--model',
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saipy', 'saved_models') + os.sep,
    help='Directory containing PolarCAP.h5')
args = parser.parse_args()

mat_file_path = os.path.join(args.wdir, '02-data', 'F_ML', 'A_wave_dB15_DT_CFM_EQP_modified.mat')
output_mat_file = os.path.join(args.wdir, '02-data', 'F_ML', 'A_wave_dB15_DT_CFM_EQP_PolCAP.mat')

if not os.path.isfile(mat_file_path):
    print(f"Error: .mat file not found at {mat_file_path}")
    sys.exit(1)

try:
    polarcap = PolarCAP(args.model)
except Exception as e:
    print(f"Error loading PolarCAP model from {args.model}: {e}")
    sys.exit(1)

waveform_fields = ['W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1']
P_ARR = 100  # P arrival sample index within the 200-sample/200 Hz raw window

mat_data = loadmat(mat_file_path)
felix_data = mat_data['Felix'][0]
num_events = len(felix_data)
print(f"Found {num_events} events in the file")

results = {field: np.zeros(num_events, dtype=np.int8) for field in waveform_fields}

counter = 0
for i in range(num_events):
    for field in waveform_fields:
        waveform = np.array(felix_data[i][field]).flatten()

        if len(waveform) < 200:
            continue

        w64 = waveform[P_ARR - 32:P_ARR + 32].astype(np.float32)

        if w64.size != 64 or np.max(np.abs(w64)) == 0:
            continue

        X = w64.reshape(1, 64, 1)
        try:
            preds = polarcap.predict(X)
        except Exception as e:
            print(f"Event {i + 1} {field}: prediction failed: {e}")
            continue

        label, _prob = preds[0]
        results[field][i] = 1 if label == 'Positive' else -1

    counter += 1
    if counter % 500 == 0:
        print(f'Processed {counter} events')

print("Saving results...")
new_names = [f'PolCAP_{f[2:]}' for f in waveform_fields]
new_cols = [results[f] for f in waveform_fields]
new_felix = rfn.append_fields(
    felix_data, new_names, data=new_cols,
    dtypes=['i1'] * len(new_names), usemask=False, asrecarray=False)

mat_data['Felix'] = new_felix.reshape(1, -1)
savemat(output_mat_file, mat_data, long_field_names=True)
print(f'Results saved to {output_mat_file}')
