#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:04:14 2025

@author: mczhang
"""

from tensorflow.keras.models import load_model

model = load_model('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/model_fm_best.hdf5', compile=False)
model.summary()
