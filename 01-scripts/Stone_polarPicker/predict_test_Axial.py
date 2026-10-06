#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 08:44:14 2025

@author: mczhang
"""
import numpy as np
import pandas as pd
from predict import polarPredict

# 1) load your 7-channel waveforms
X7 = np.load("./data/Felix_waveforms_64.npy")   # shape (N, 7, 64)

# 2) initialize predictor
polarpred = polarPredict(
    mode="predict",
    model="./models/polarityModel_20240803.keras"
)

all_preds = []
for i in range(X7.shape[0]):
    # --- pick channel 0 (W_AS1); if you want AS2, use 1, etc. ---
    x_raw = X7[i, 0, :]              # shape (64,)
    
    # --- reshape to (1, 64, 1) ---
    x_i = x_raw.reshape(1, 64, 1)
    
    # --- predict on single trace ---
    pred_i = polarpred.predict(x_i)  # returns [(label, prob)]
    all_preds.append(pred_i[0])

# 3) save CSV
df = pd.DataFrame(all_preds, columns=["prediction","confidence"])
df.to_csv(
    "Felix_AS1_predictions_one_by_one.csv",
    index=False,
    header=False
)

