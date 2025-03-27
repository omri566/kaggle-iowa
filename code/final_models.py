import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# 📌 Cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=420)

# 📌 Evaluate Models using the log of y on Test Data
def evaluate_log_model(model, model_name):
    y_pred_log = model.predict(X_test_scaled)  # Predictions in log scale
    y_pred = np.expm1(y_pred_log)  # Inverse transform to get real values

# 📌 Evaluate Models on Test Data
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test_scaled)  # Predictions in log scale







linear_reg_model = Ridge(alpha="1")

