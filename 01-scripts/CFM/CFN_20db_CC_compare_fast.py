import os
import argparse
import tensorflow as tf
import numpy as np
import sys
from scipy.io import loadmat, savemat

# -----------------------
# Argument parser
# -----------------------
parser = argparse.ArgumentParser(
    description='Predict polarity for multiple .mat files (N_PoAS1.mat, N_PoAS2.mat, …)'
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
    default='./Network_CFM/CFM.hdf5',
    help='Path to the CFM model file'
)
args = parser.parse_args([a for a in sys.argv[1:] if not a.startswith('--f=')])

# -----------------------
# Configure TensorFlow to use all CPU cores
# -----------------------
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

# -----------------------
# Load model
# -----------------------
try:
    cfm_model = tf.keras.models.load_model(args.model, compile=False)
except Exception as e:
    print(f"Error loading model from {args.model}: {e}")
    sys.exit(1)

# -----------------------
# Station files
# -----------------------
station_files = ['N_PoAS1.mat','N_PoAS2.mat','N_PoCC1.mat',
                 'N_PoEC1.mat','N_PoEC2.mat','N_PoEC3.mat','N_PoID1.mat']

waveform_fields = ['W_AS1','W_AS2','W_CC1','W_EC1','W_EC2','W_EC3','W_ID1']

# -----------------------
# Process each file
# -----------------------
for fname in station_files:
    mat_path = os.path.join(args.wdir, fname)
    out_path = mat_path.replace('.mat', '_CFM.mat')

    if not os.path.isfile(mat_path):
        print(f"Skipping missing file: {mat_path}")
        continue

    print(f"\nProcessing file: {mat_path}")

    # Load Felix struct
    mat_data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    Felix = mat_data['Felix']
    if not isinstance(Felix, np.ndarray):
        felix_data = np.array([Felix])
    else:
        felix_data = Felix
    num_events = len(felix_data)
    print(f"Found {num_events} events")

    # Initialize prediction results
    cfm_results = {field: np.zeros(num_events, dtype=np.int8) for field in waveform_fields}

    # -----------------------
    # Collect all valid waveforms for batch prediction
    # -----------------------
    batch_inputs = []
    batch_indices = []
    batch_fields = []

    for i in range(num_events):
        for field in waveform_fields:
            if not hasattr(felix_data[i], field):
                continue
            waveform = np.array(getattr(felix_data[i], field)).flatten()
            if len(waveform) < 181 or np.max(np.abs(waveform)) == 0:
                continue

            wf_input = waveform[20:180]
            wf_input = wf_input - np.mean(wf_input)
            wf_input = wf_input / (np.max(np.abs(wf_input)) + 1e-6)

            batch_inputs.append(wf_input)
            batch_indices.append(i)
            batch_fields.append(field)

        if (i + 1) % 100 == 0:
            print(f"  Collected {i+1} / {num_events} events")

    # Convert to NumPy for batch prediction
    if len(batch_inputs) > 0:
        batch_inputs = np.array(batch_inputs)
        preds = cfm_model.predict(batch_inputs, batch_size=512, verbose=1)

        # Assign results back
        for pred_val, idx, field in zip(preds.flatten(), batch_indices, batch_fields):
            cfm_results[field][idx] = 1 if pred_val >= 0.5 else -1

    # Attach predictions to Felix
    for i in range(num_events):
        for field in waveform_fields:
            cfm_field = f'CFM_{field}'
            setattr(felix_data[i], cfm_field, cfm_results[field][i])

    # Save results
    try:
        savemat(out_path, {'Felix': felix_data.reshape(1,-1)}, appendmat=False, do_compression=True)
        print(f"Saved predictions to {out_path}")
    except Exception as e:
        print(f"Error saving {out_path}: {e}")
