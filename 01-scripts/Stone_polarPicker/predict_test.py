import numpy as np
import pandas as pd
from predict import polarPredict

X = np.load("./data/Felix_waveforms_64.npy")
polarpred = polarPredict(mode="predict",model="./models/polarityModel_20240803.keras")

predictions =polarpred.predict(X)

olist = pd.DataFrame(predictions)
olist.to_csv('Felix_waveforms_64_predictions_undecided_newdownload.csv',index=False,header=False)
