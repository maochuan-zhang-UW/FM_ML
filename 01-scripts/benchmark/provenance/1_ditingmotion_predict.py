#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on 2025-09-19
@author: mczhang
Process all station .mat files (N_PoAS1.mat, N_PoAS2.mat, …), predict polarity,
and save separately as N_PoAS1_DT.mat, etc.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import warnings
import sys
from tensorflow.keras import backend as K
from scipy.io import loadmat, savemat
warnings.filterwarnings("ignore")

# Clear Keras session
try:
    K.clear_session()
except Exception as e:
    print(f"Error clearing Keras session: {e}")
    sys.exit(1)

# -----------------------
# Parse command-line arguments
# -----------------------
parser = argparse.ArgumentParser(
    description='Predict polarity for all N_Po*.mat station files'
)
parser.add_argument(
    '--wdir',
    type=str,
    default='/Users/mczhang/Documents/GitHub/FM5_ML/02-data/N_Po/',
    help='Directory containing N_Po*.mat files'
)
parser.add_argument(
    '--model',
    type=str,
    default='./models/DiTingMotionJul.hdf5',
    help='Path to the CNN model file'
)
args = parser.parse_args()

# Verify model file exists
if not os.path.isfile(args.model):
    print(f"Error: Model file not found at {args.model}")
    sys.exit(1)

# -----------------------
# Load the CNN model
# -----------------------
try:
    motion_model = tf.keras.models.load_model(args.model, compile=False)
except Exception as e:
    print(f"Error loading model from {args.model}: {e}")
    sys.exit(1)

# -----------------------
# Station files list
# -----------------------
station_files = ['N_PoAS1.mat','N_PoAS2.mat','N_PoCC1.mat',
                 'N_PoEC1.mat','N_PoEC2.mat','N_PoEC3.mat','N_PoID1.mat']

# Waveform fields
waveform_fields = ['W_AS1','W_AS2','W_CC1','W_EC1','W_EC2','W_EC3','W_ID1']

# -----------------------
# Process each station file
# -----------------------
for fname in station_files:
    mat_file_path = os.path.join(args.wdir, fname)
    output_mat_file = mat_file_path.replace('.mat', '_DT.mat')

    if not os.path.isfile(mat_file_path):
        print(f"Skipping missing file: {mat_file_path}")
        continue

    print(f"\nProcessing file: {mat_file_path}")

    try:
        mat_data = loadmat(mat_file_path)
        felix_data = mat_data['Felix'][0]  # MATLAB stores as 1xN array
        num_events = len(felix_data)
        print(f"  Found {num_events} events")

        # Prepare result arrays
        poml_results = {field: np.zeros(num_events, dtype=np.int8) for field in waveform_fields}

        counter = 0
        for i in range(num_events):
            for field in waveform_fields:
                if field not in felix_data.dtype.names:
                    continue

                waveform = np.array(felix_data[i][field]).flatten()

                if len(waveform) < 142:  # Need at least 142 points
                    continue

                # Input preparation
                motion_input = np.zeros([1, 128, 2])
                motion_input[0, :, 0] = waveform[36:164]

                if np.max(motion_input[0, :, 0]) == 0:
                    continue

                motion_input[0, :, 0] -= np.mean(motion_input[0, :, 0])
                norm_factor = np.std(motion_input[0, :, 0])
                if norm_factor == 0:
                    continue
                motion_input[0, :, 0] /= norm_factor

                diff_data = np.diff(motion_input[0, 64:, 0])
                motion_input[0, 65:, 1] = np.sign(diff_data[:])

                # Predict
                pred_res = motion_model.predict(motion_input, verbose=0)
                pred_fmp = (pred_res['T0D0'] + pred_res['T0D1'] +
                            pred_res['T0D2'] + pred_res['T0D3']) / 4

                if np.argmax(pred_fmp[0, :]) == 1:
                    poml_results[field][i] = 1   # Down
                elif np.argmax(pred_fmp[0, :]) == 0:
                    poml_results[field][i] = -1  # Up
                else:
                    poml_results[field][i] = 0   # Unknown

            counter += 1
            if counter % 100 == 0:
                print(f"  Processed {counter} / {num_events} events")

        # -----------------------
        # Save results back
        # -----------------------
        new_dtype = list(felix_data.dtype.descr)
        for field in waveform_fields:
            new_dtype.append((f'PoML_{field}', np.int8))

        new_felix = np.zeros(num_events, dtype=new_dtype)
        for field in felix_data.dtype.names:
            new_felix[field] = felix_data[field]
        for field in waveform_fields:
            new_felix[f'PoML_{field}'] = poml_results[field]

        mat_data['Felix'] = new_felix.reshape(1, -1)
        savemat(output_mat_file, mat_data, long_field_names=True)
        print(f"  Results saved to {output_mat_file}")

    except Exception as e:
        print(f"Error processing {mat_file_path}: {e}")
        continue
