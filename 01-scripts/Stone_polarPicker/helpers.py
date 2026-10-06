import math
import numpy as np
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay

import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_classification_learning_curves(history_df, fig_fname=None):
    acc = history_df['sequential_2_acc'].tolist()
    n_epochs = len(acc)
    val_acc = history_df['val_sequential_2_acc'].tolist()
    loss = history_df['sequential_2_loss'].tolist()
    val_loss = history_df['val_sequential_2_loss'].tolist()
    epochs_range = range(n_epochs)
    #
    min_val_loss_idx = np.argmin(val_loss)
    # accuracy
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot(epochs_range[min_val_loss_idx], val_acc[min_val_loss_idx], color='red', marker='o', markersize=15., alpha=0.3)
    plt.legend(loc='upper left', frameon=False)
    #plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    # loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.plot(epochs_range[min_val_loss_idx], val_loss[min_val_loss_idx], color='red', marker='o', markersize=15., alpha=0.3)
    plt.legend(loc='upper right', frameon=False)
    #plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # acc vs loss
    plt.subplot(1, 3, 3)
    plt.plot(loss, acc, 'o', label='Training', alpha=0.5)
    plt.plot(val_loss, val_acc, 'o', label='Validation', alpha=0.5)
    plt.plot(val_loss[min_val_loss_idx], val_acc[min_val_loss_idx], color='red', marker='o', markersize=15., alpha=0.3)
    plt.legend(loc='upper right', frameon=False)
    #plt.title('Training and Validation Accuracy vs Loss')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.grid(True)
    #
    if fig_fname is not None:
        # TO-DO: check path
        plt.savefig(fig_fname)
    else:
        plt.show()
# END plot classification learning history

def plot_confusion_matrix(cmat, classes, cmap='plasma', fig_fname=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    pmat = ConfusionMatrixDisplay(cmat, display_labels=classes)
    pmat.plot(cmap=cmap, ax=ax)
    ax.grid(False)
    if fig_fname is not None:
        # TO-DO: check path
        plt.savefig(fig_fname)
    else:
        plt.show()
