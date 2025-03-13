import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

# Load your preprocessed dataset
df = pd.read_csv("C:/Users/itayp/PycharmProjects/housing prices Kaggle/housing prices train DF_clean.csv")

# Split features & target variable
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
# Set up 5-fold Cross-Validation (since data < 1500 rows)
kf = KFold(n_splits=5, shuffle=True, random_state=420)

# Initialize XGBoost model with default parameters
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=420,
    n_estimators=1000,
    n_jobs = 4,
    learning_rate = 0.05,
    max_depth = 6
)

# Perform Cross-Validation & Compute RMSE
cv_rmse = np.sqrt(-cross_val_score(xgb_model, X, y, cv=kf, scoring="neg_mean_squared_error"))

# Print Results
print(f"Cross-Validation RMSE Scores: {cv_rmse}")
print(f"Mean RMSE: {cv_rmse.mean():.4f}")
print(f"Standard Deviation RMSE: {cv_rmse.std():.4f}")
