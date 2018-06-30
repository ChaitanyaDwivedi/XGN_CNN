import pickle
data = pickle.load(open("../Data/PCA_GREYC_2X.pickle", "rb"))

from sklearn.model_selection import train_test_split
import numpy as np

X = np.array(data['data'])
Y = np.array(data['labels'])
#X = X.reshape(X.shape[0], 1, 31, 1)

n = X.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
print(X_train.shape)

import xgboost as xgb
#To load XGB model
model = xgb.Booster({'nthread': 4}) # init model
model.load_model("../XGB_models/GREYC1.model") # load data

xg_test = xgb.DMatrix(X_test.astype(float), label = y_test.astype(float))
xg_valid = xgb.DMatrix(X_valid.astype(float), label = y_valid.astype(float))

pred_test = model.predict(xg_test)
pred_valid = model.predict(xg_valid)

print(pred_test.shape, pred_valid.shape)

pickle.dump({"valid": pred_valid, "test": pred_test}, open("./xgboostPredictionsGreyC.pickle", "wb"))