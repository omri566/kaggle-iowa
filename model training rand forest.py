import json

import sklearn.ensemble
import xgboost as xgb
import numpy as np
import pandas as pd
from numba.core.datamodel.old_models import ListModel
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
import random
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Load your preprocessed dataset
df = pd.read_csv("data/feature_engineered_data.csv")
print(len(df))
# Split features & target variable
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
X_train, X_test , y_train , y_test = train_test_split(X,y , test_size=0.2, random_state=420)

# Set up 5-fold Cross-Validation (since data < 1500 rows)
kf = KFold(n_splits=5, shuffle=True, random_state=420)

param_grid = {
    "n_estimators" : [100,150,200,250] ,
    "max_depth" : [None,5,10,20,40] ,
    "max_features" : [1.0,"sqrt", "log2",1.0] ,
}
with open("model training/rf_model_best_params.json") as f:
    best_params = f

rf_model = RandomForestRegressor(n_jobs=-1,random_state=420,criterion ="squared_error",**best_params)

#grid search in comment so it wont run every time
"""
grid_search = GridSearchCV(rf_model, param_grid, cv=kf,scoring="neg_mean_squared_error",verbose=2,n_jobs=-1)
grid_search.fit(X_train,y_train)
best_model = grid_search.best_estimator_
best_param = grid_search.best_params_
with open("model training/rf_model_best_params.json", "w") as f:
    json.dump(best_param, f, indent=4)
    
# Print RMSE for each hyperparameter combination
print("\nRMSE for each hyperparameter combination:")
for i, params in enumerate(grid_search.cv_results_['params']):
    rmse = np.sqrt(-grid_search.cv_results_['mean_test_score'][i])  # Convert negative MSE to RMSE
    print(f"Params: {params} -> RMSE: {rmse:.4f}")

# Best model
print("\nBest Parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))
"""""
