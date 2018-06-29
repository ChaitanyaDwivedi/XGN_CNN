# XGN_CNN
Using XGB: </br>
import xgboost as xgb  
#To load XGB model  
model = xgb.Booster({'nthread': 4})  # init model  
model.load_model('model_name')  # load data


Make a data object before passing it to xgb model:  
xg_test = xgb.DMatrix(X_test, label = y_test) #y is to be one_hot encoded  
pred = model.predict(xg_test)  

