from pydoc import importfile
from pyexpat import features

import pandas as pd
import numpy as np
import shap
from seaborn import heatmap
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import json



#till 21 shits that dael with importing the feature engineering function
import importlib.util

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# Loading the data
preprocess_train_data_path = "data/feature_engineered_data.csv"
dfcpy = pd.read_csv(preprocess_train_data_path)
X = dfcpy.drop(columns=["SalePrice"])
y = dfcpy["SalePrice"]


# setting the data
X_train, X_test , y_train , y_test = train_test_split(X,y , test_size=0.2, random_state=420)
kf = KFold(n_splits=5, random_state=420 , shuffle=True)

#defineing the model
xgb_model = XGBRegressor(random_state=420 , objective="reg:squarederror",eval_metric="rmse")


#model parameters to iterate over
param_grid = {
    "n_estimators" : [50,100,200,400] ,
    "learning_rate" : [0.05,0.1,0.3],
    "gamma" : [0,1,2],
    "max_depth" : [4,6,5,8]
}

#fitting and finding the best model
grid_search = GridSearchCV(xgb_model, param_grid, cv=kf,scoring="neg_mean_squared_error",verbose=2,n_jobs=-1)
grid_search.fit(X_train,y_train)
best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_

#save the best parameters in json file
with open("data/first_xgb_param.json","w") as f:
    json.dump(best_parameters,f,indent=4)
print("Best parameters saved to best_xgb_params.json")


#predict
predictions = best_model.predict(X_test)
#evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(best_parameters)


# SHAP values analysis
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

# Plot summary
shap.summary_plot(shap_values, X_test)

print("Hello world")