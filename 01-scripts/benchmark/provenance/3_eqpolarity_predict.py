#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:25:48 2025

@author: mczhang
"""

# Modified script to process seismic waveform data from MATLAB .mat file
import os
import argparse
import tensorflow as tf
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
from tensorflow.keras import backend as K
import warnings
import sys
from eqpolarity.utils import construct_model

# Clear Keras session
try:
    K.clear_session()
except Exception as e:
    print(f"Error clearing Keras session: {e}")
    sys.exit(1)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Process seismic waveforms from .mat file for polarity prediction using EQpolarity model')
parser.add_argument(
    '--wdir',
    type=str,
    default='/Users/mczhang/Documents/GitHub/FM5_ML/',
    help='Base directory containing 02-data/A_wave/A_wave_dB15.mat')
parser.add_argument(
    '--model',
    type=str,
    default='./models/best_weigths_Binary_Texas_Transfer10.h5',
    help='Path to the EQpolarity model weights file')
args = parser.parse_args()

# Construct file paths
mat_file_path = os.path.join(args.wdir, '02-data', 'F_ML', 'A_wave_dB15_DT_CFM_struct.mat')
output_mat_file = os.path.join(args.wdir, '02-data', 'F_ML', 'A_wave_dB15_DT_CFM_EQP.mat')
model_path = args.model

# Verify files exist
if not os.path.isfile(mat_file_path):
    print(f"Error: .mat file not found at {mat_file_path}")
    sys.exit(1)
if not os.path.isfile(model_path):
    print(f"Error: Model weights file not found at {model_path}")
    sys.exit(1)

# Load the EQpolarity model
try:
    input_shape = (600, 1)
    model = construct_model(input_shape)
    model.load_weights(model_path)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    sys.exit(1)

# Create plots directory if it doesn't exist
plots_dir = os.path.join(args.wdir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# List of all waveform fields to process
waveform_fields = ['W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1']

# Load the .mat file
try:
    mat_data = loadmat(mat_file_path)
    felix_data = mat_data['Felix'][0]  # MATLAB struct array is stored as 1xN array
    
    num_events = len(felix_data)
    print(f"Found {num_events} events in the file")
    
    # Initialize arrays for new polarity field (EQP[station])
    poml_results = {field: np.zeros(num_events, dtype=np.int8) for field in waveform_fields}
    
    counter = 0
    for i in range(num_events):
        try:
            # Process each waveform field
            for field in waveform_fields:
                # Extract waveform (200 points at 200 Hz)
                waveform = felix_data[i][field]
                
                # Convert to 1D array regardless of original shape
                waveform = np.array(waveform).flatten()
                
                # Verify we have enough samples
                if len(waveform) < 200:
                   # print(f"Event {i+1} {field}: Insufficient samples ({len(waveform)}), skipping")
                    continue
                
                # Resample all waveforms from 200 to 600 samples
                waveform = signal.resample(waveform, 600)
                
                # Prepare input for the model
                motion_input = np.zeros([1, 600, 1])
                motion_input[0, :, 0] = waveform
                
                if np.max(np.abs(motion_input[0, :, 0])) == 0:
                    print(f"Event {i+1} {field}: Zero amplitude waveform, skipping")
                    continue
                
                # Preprocess waveform (demean, normalize)
                motion_input[0, :, 0] -= np.mean(motion_input[0, :, 0])
                norm_factor = np.std(motion_input[0, :, 0])
                
                if norm_factor == 0:
                    print(f"Event {i+1} {field}: Zero standard deviation, skipping")
                    continue
                
                motion_input[0, :, 0] /= norm_factor
                
                # Predict polarity
                # NOTE: the original version of this loop (batch_size=1024 and
                # in-place boolean-mask thresholding on the predict() output,
                # `out`) produced predictions that scored near/below chance for
                # several stations when run over the full 6801-event/7-station
                # set, despite looking correct on small subsets -- exact root
                # cause not fully isolated (batch_size mismatch with the
                # single-sample input and/or mutating `out` in place across
                # ~47000 repeated predict() calls are both suspects). This
                # batch_size=1 + no-in-place-mutation version was validated
                # against ground truth on the full dataset (see
                # 01-scripts/I_CC_MLs_V2_comapreall.m) and should be kept.
                out = model.predict(motion_input, batch_size=1, verbose=0)
                prob = float(out.flatten()[0])
                pred = -1 if prob >= 0.5 else 1

                # Assign polarity
                poml_results[field][i] = pred
                
            counter += 1
            if counter % 100 == 0:
                print(f'Processed {counter} events')
        
        except Exception as e:
            print(f"Error processing event {i+1}: {e}")
            continue

except Exception as e:
    print(f"Error reading .mat file {mat_file_path}: {e}")
    sys.exit(1)

# Save results back to .mat file
try:
    # NOTE: manually rebuilding the dtype via felix_data.dtype[field_name] (as
    # the original version of this block did) can raise
    # "TypeError: 'NoneType' object is not iterable" from numpy for this
    # struct's field dtypes, depending on numpy version. append_fields is the
    # robust way to add columns to an existing structured array.
    from numpy.lib import recfunctions as rfn

    new_names = [f'EQP_{field[2:]}' for field in waveform_fields]
    new_cols = [poml_results[field] for field in waveform_fields]
    new_felix = rfn.append_fields(
        felix_data, new_names, data=new_cols,
        dtypes=['i1'] * len(new_names), usemask=False, asrecarray=False)

    # Update the mat_data structure
    mat_data['Felix'] = new_felix.reshape(1, -1)  # Reshape to MATLAB's 1xN format
    
    # Save the modified data
    savemat(output_mat_file, mat_data, long_field_names=True)
    print(f'Results saved to {output_mat_file}')
    
except Exception as e:
    print(f"Error saving .mat file {output_mat_file}: {str(e)}")
    sys.exit(1)