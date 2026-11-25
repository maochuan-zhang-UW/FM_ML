import os
import argparse
import tensorflow as tf
import numpy as np
import sys
from scipy.io import loadmat, savemat

# Create argument parser
parser = argparse.ArgumentParser(
    description='Predict polarity from .mat file waveforms using CFM model'
)
parser.add_argument(
    '--wdir',
    type=str,
    default='/Users/mczhang/Documents/GitHub/FM5_ML/',
    help='Base directory containing 02-data/A_wave/A_wave_dB15.mat'
)
parser.add_argument(
    '--model',
    type=str,
    default='./Network_CFM/CFM_with_timeshift.hdf5',
    help='Path to the CFM model file'
)

# Filter out Jupyter-specific arguments
valid_args = [arg for arg in sys.argv if not arg.startswith('--f=')]

# Parse arguments
args = parser.parse_args(valid_args[1:])  # Skip the first argument (script name or kernel launcher)

# File paths
mat_file_path = os.path.join(args.wdir, '02-data', 'F_ML', 'A_wave_dB15_DT.mat')
output_mat_file = os.path.join(args.wdir, '02-data', 'F_ML', 'A_wave_dB15_DT_CFM.mat')
model_path = args.model

# Verify input files
if not os.path.isfile(mat_file_path):
    print(f"Error: .mat file not found at {mat_file_path}")
    sys.exit(1)
if not os.path.isfile(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

print(f"Input .mat file: {mat_file_path}")
print(f"Output .mat file: {output_mat_file}")
print(f"Model file: {model_path}")

# Load CFM model
try:
    cfm_model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    sys.exit(1)

# Load Felix data
try:
    mat_data = loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    Felix = mat_data['Felix']

    # Standardize to array
    if isinstance(Felix, np.ndarray):
        felix_data = Felix
        num_events = len(felix_data)
    else:
        felix_data = np.array([Felix])  # Convert single struct to 1-element array
        num_events = 1

    print(f"Found {num_events} events in the file")
except Exception as e:
    print(f"Error reading .mat file {mat_file_path}: {e}")
    sys.exit(1)

# Stations to process
waveform_fields = ['W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1']

# Initialize arrays for results
cfm_results = {field: np.zeros(num_events, dtype=np.int8) for field in waveform_fields}

# Process each event
counter = 0
for i in range(num_events):
    try:
        for field in waveform_fields:
            if not hasattr(felix_data[i], field):
                continue

            waveform = getattr(felix_data[i], field)
            waveform = np.array(waveform).flatten()

            if len(waveform) < 181:  # Need at least 180 samples
                continue

            # Extract 160 samples (20–180)
            wf_input = waveform[20:180]

            if np.max(np.abs(wf_input)) == 0:
                continue

            # Demean + normalize
            wf_input = wf_input - np.mean(wf_input)
            wf_input = wf_input / (np.max(np.abs(wf_input)) + 1e-6)

            # Reshape to (1, 160) for model
            wf_input = wf_input.reshape(1, -1)

            # Predict polarity
            pred = cfm_model.predict(wf_input, verbose=0)
            pred_val = pred.flatten()[0]

            if pred_val >= 0.5:
                cfm_results[field][i] = 1   # Upward
            else:
                cfm_results[field][i] = -1  # Downward

        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} events")

    except Exception as e:
        print(f"Error processing event {i+1}: {e}")
        continue

# Add CFM results to Felix structure
for i in range(num_events):
    for field in waveform_fields:
        cfm_field = f'CFM_{field}'  # Create field name like CFM_W_AS1
        setattr(felix_data[i], cfm_field, cfm_results[field][i])

# Update the mat_data structure
    mat_data['Felix'] = felix_data.reshape(1, -1)  # Reshape to MATLAB's 1xN format
# Save the updated Felix structure to a .mat file
try:
    savemat(output_mat_file, {'Felix': mat_data}, appendmat=False)
    print(f"Results saved to {output_mat_file}")
except Exception as e:
    print(f"Error saving .mat file {output_mat_file}: {e}")
    sys.exit(1)