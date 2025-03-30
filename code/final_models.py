import json
import os
from math import gamma

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 📌 Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, "data", "feature_engineered_data.csv")

# 📌 Load dataset
df = pd.read_csv(data_path)
print(f"Dataset size: {len(df)}")

# 📌 Split features & target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# 📌 Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 Log-transform the target variable
y_train_log = np.log1p(y_train)  # log(1 + y)
y_test_log = np.log1p(y_test)


# 📌 Evaluate Models using the log of y on Test Data
def evaluate_log_model(model, model_name):
    y_pred_log = model.predict(X_test_scaled)  # Predictions in log scale
    y_pred = np.expm1(y_pred_log)  # Inverse transform to get real values
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n📌 True RMSE on Test Set for {model_name}: {rmse:.4f}")

# 📌 Evaluate Models on Test Data
def evaluate_model(model, model_name,X_test_data=None):
    if X_test_data is None:
        X_test_data = X_test_scaled  # Default remains scaled data
    y_pred = model.predict(X_test_data)  # Predictions in log scale
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print(f"\n📌 True RMSE on Test Set for {model_name}: {rmse:.4f}")


#train and test linear ridge model

#train and test xgb model
xgb_model = XGBRegressor(eval_metric="rmse",n_jobs=-1,random_state=420,objective="reg:squarederror",colsample_bytree=0.6, gamma=0,learning_rate=0.05,max_depth=5,n_estimators=500,reg_alpha=1, reg_lambda=10, subsample=0.6)
xgb_model.fit(X_train,y_train)
xgb_model_scaled = XGBRegressor(eval_metric="rmse",n_jobs=-1,random_state=420,objective="reg:squarederror",gamma=1,learning_rate=0.093,max_depth=3,n_estimators=181)
xgb_model_scaled.fit(X_train_scaled,y_train_log)
evaluate_model(xgb_model,"XGB",X_test)
evaluate_log_model(xgb_model_scaled,"XGB_SCALED")
