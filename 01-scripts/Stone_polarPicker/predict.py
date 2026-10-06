import numpy as np
import pandas as pd
import tensorflow.keras as keras
import numpy as np

# class polarPredict:
#     def __init__(self, mode="predict", model="polarityModel.keras"):
#         self.model = keras.models.load_model(model)
#         # ... 

#     def predict(self, X):
#         X = Norm(X)
#         preds = self.model.predict(X)

#         # DEBUG: see shape and example logits
#         print("  ▶ model.predict returned:", preds.shape)   # (N, C)
#         print("  ▶ example row:", preds[0])

#         # your old mapping
#         # polarity = ["Negative","Positive"]
#         # …


def Norm(X):

    maxi = np.max(abs(X),axis=1)
    X_ret = X.copy()
    for i in range(X.shape[0]):
        X_ret[i] = X_ret[i]/maxi[i]
    
    return X_ret

class polarPredict:
    def __init__(self,mode="predict",**kwargs):
        if mode == "predict":
            mod_name = kwargs["model"]
            self.model = keras.models.load_model(mod_name)

    def predict(self,X):
        model = self.model

        y_pred = model(Norm(X))
        
        pol_predict = np.argmax(y_pred[1],axis=1)
        pred_prob = np.max(y_pred[1],axis=1)
        predictions = []
        #polarity = ['Negative','Positive']
        polarity = ['Negative','Positive','Undecided']

        for pol, prob in zip(pol_predict,pred_prob):
            predictions.append((polarity[pol],prob))

        return predictions
