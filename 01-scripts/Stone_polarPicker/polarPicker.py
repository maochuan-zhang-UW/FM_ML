import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Flatten, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report
from helpers import plot_classification_learning_curves, plot_confusion_matrix

# set variables
drop_rate = 0.3
learn_rate = 0.001
classes = ["Negative","Positive"]
#classes = ["Negative","Positive","Undecided"]

# -----------------------------------------------------------------------------
# init logging
# -----------------------------------------------------------------------------

use_local_log = True
local_log_fname = './logs/polarPicker_learning.log'
# otherwise read output from output file with name specified in SLURM script

if use_local_log: # else 
    old_stdout = sys.stdout
    log_file = open(local_log_fname, 'w') # can do this or use output file from SLURM
    sys.stdout = log_file

# -----------------------------------------------------------------------------
# read in training, validating, and testing datasets
# -----------------------------------------------------------------------------

pdat = "/caldera/projects/usgs/hazards/ehp/istone/tallgrass_ml/data_dir/"
xin = np.load(pdat+"polarity_training_timeseries_undecided.npy")
yin = np.load(pdat+"polarity_training_polarities_undecided.npy")
vxin = np.load(pdat+"polarity_validating_timeseries_undecided.npy")
vy = np.load(pdat+"polarity_validating_polarities_undecided.npy")
txin = np.load(pdat+"polarity_testing_timeseries_undecided.npy")
ty = np.load(pdat+"polarity_testing_polarities_undecided.npy")

yin = to_categorical(yin)
vy = to_categorical(vy)
ty = to_categorical(ty)

def norm(X):
    max_val = np.max(abs(X),axis=1)
    X_norm = X.copy()
    for i in range(X.shape[0]):
        X_norm[i] = X_norm[i]/max_val[i]

    return X_norm

# -----------------------------------------------------------------------------
# get info on cpus, gpus
# -----------------------------------------------------------------------------

print(f'Using TensorFlow version {tf.__version__}')
# ? print(f'Using Keras version {tf.keras.__version__}')

cpus = tf.config.list_logical_devices("CPU")
gpus = tf.config.list_logical_devices("GPU")
print(f'Running on {len(cpus)} CPU(s)')
print(f'Running on {len(gpus)} GPU(s)')
if len(gpus) > 0:
    strategy = tf.distribute.MirroredStrategy(gpus)  # this works for 1 to multiple GPUs; but must create vars differently
    print('Using mirrored strategy to run on multiple GPUs on a single node')
else:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on CPU or single GPU')

# -----------------------------------------------------------------------------
# define model
# -----------------------------------------------------------------------------

def build_polarPicker(drop_rate=drop_rate):
    """
    Define an autoencoder system
    """
    
    # encoder (uses timeseries as input)
    encoder = Sequential()
    encoder.add(Input(shape = (64,1)))
    encoder.add(Conv1D(32, 32, activation = 'relu', padding = 'same'))
    encoder.add(Dropout(drop_rate))
    encoder.add(BatchNormalization())
    encoder.add(MaxPooling1D(2, padding='same'))
    encoder.add(Conv1D(8, 16, activation = 'relu', padding = 'same'))
    encoder.add(BatchNormalization())
    encoder.add(MaxPooling1D(2, padding='same'))

    # decoder (uses encoder as input)
    decoder = Sequential()
    decoder.add(Input(shape = (16,8)))
    decoder.add(Conv1D(8, 16, activation = 'tanh', padding = 'same'))
    decoder.add(BatchNormalization())
    decoder.add(UpSampling1D(2))
    decoder.add(Conv1D(32, 32, activation = 'relu', padding = 'same'))
    decoder.add(BatchNormalization())
    decoder.add(UpSampling1D(2))
    decoder.add(Conv1D(1, 64, padding = 'same', activation = 'tanh'))

    # probability estimate (uses encoder as input)
    prob = Sequential()
    prob.add(Input(shape = (16,8)))
    prob.add(Flatten())
    prob.add(Dense(2, activation = 'softmax'))
#    prob.add(Dense(3, activation = 'softmax'))

    return encoder,decoder,prob

# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

encoder,decoder,prob = build_polarPicker()
encoder.summary()
decoder.summary()
prob.summary()

X = Input(shape = (64,1))
enc = encoder(X)
dec = decoder(enc)
p = prob(enc)

model = Model(inputs=X,outputs=[dec,p])
#model = Model(inputs=X,outputs=p)

# loss is based on dec and p, using mean squared error and huber loss, respectively.
# dec loss is weighted 1, and p loss is weighted 200 (since we care more about p)
hub = tf.keras.losses.Huber(delta=0.5, name='huber_loss')

model.compile(optimizer = keras.optimizers.Adam(learning_rate=learn_rate), \
    loss=['mse', hub], loss_weights = [1,200],metrics = ['mse','acc'])
#model.compile(optimizer = keras.optimizers.Adam(learning_rate=learn_rate), \
#    loss=[hub],metrics = ['mse','acc'])

# -----------------------------------------------------------------------------
# Train model
# -----------------------------------------------------------------------------

# Training options:
# Stop the training early if no improvement after 15 epochs
# Reduce the learning rate by a factor of 10 if no improvement after 10 epochs.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,min_lr=1e-6)

# since model loss is determined by both the decoded signal and the estimated polarity,
# the output targets are the input xin and the polarity labels y.
history = model.fit(x=xin,y=[xin,yin],validation_data=[vxin,[vxin,vy]],epochs=120,callbacks=[early_stop,reduce_lr])
#history = model.fit(x=xin,y=yin,validation_data=[vxin,vy],epochs=120,callbacks=[early_stop,reduce_lr])

history_df = pd.DataFrame(history.history)
history_df.to_csv('./tmp/polarPicker_learning-history.csv')

print('\n# BEGIN polarPicker training history description')
print('Training history description:')
history_df.describe() # TO-DO not printing to log file
print('# END polarPicker training history description')

print('\n# BEGIN polarPicker training history table')
print('Training history description:')
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 5,
                       ):
    print(history_df)

plot_classification_learning_curves(history_df, fig_fname='./figs/polarPicker-learning_history.png')

# -----------------------------------------------------------------------------
# Save model
# -----------------------------------------------------------------------------

model.save("polarityModel.keras")

# -----------------------------------------------------------------------------
# Evaluate model
# -----------------------------------------------------------------------------

# validation data

# --- testing dataset
# BEGIN evaluation on testing dataset
#test_loss, test_acc = model.evaluate(txin,[txin,ty],verbose=False)
y_pred = model.predict(txin)
pred_idx = np.argmax(y_pred[1], axis=1)

true_idx = np.argmax(ty.astype(int),axis=1)

true_lab = [classes[i] for i in true_idx]
pred_lab = [classes[i] for i in pred_idx]

# confusion matrix
cmat = confusion_matrix(true_lab, pred_lab, labels=classes, sample_weight=None, normalize=None)
plot_confusion_matrix(cmat, classes, fig_fname='./figs/polarPicker-conf_mat_testing.png')
# metrics
test_rpt = classification_report(true_lab, pred_lab)

print('\n# BEGIN evaluation:testing dataset')
print('\nEvaluation on testing dataset:')
#print(f'\nTest dataset accuracy: {test_acc}')
print('Confusion Matrix')
print(cmat)
print('Classification Report:')
print(test_rpt)
print('# END evaluation:testing dataset')

# -----------------------------------------------------------------------------
# finalize
# -----------------------------------------------------------------------------

# ------- set back to stdout --------------------------------------------------

if use_local_log:
    print('\nDONE!!!')
    sys.stdout = old_stdout
    log_file.close()

print('\nDONE!!!')
