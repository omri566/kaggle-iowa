import os
import json
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

# 📌 Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, "data", "feature_engineered_data.csv")
best_param_file = os.path.join(project_dir, "data", "best_xgb_params.json")

# 📌 Load dataset
df = pd.read_csv(data_path)
print(f"📌 Dataset size: {len(df)}")

# 📌 Split features & target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# 📌 Cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=420)

# 📌 Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

# 📌 Initialize XGBoost model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42
)

# 📌 Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=kf,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# 📌 Extract Best Parameters & Save to JSON
best_params = grid_search.best_params_
with open(best_param_file, "w") as f:
    json.dump(best_params, f, indent=4)

print(f"\n✅ Best Parameters Found: {best_params}")

# 📌 Train Final Model using Best Parameters
best_xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    **best_params
)
best_xgb_model.fit(X_train, y_train)

# 📌 Cross-Validation RMSE
cv_rmse = -cross_val_score(
    best_xgb_model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
)

# 📌 Evaluate on Test Set
y_pred = best_xgb_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 📌 Print Results
print("\n📌 Cross-Validation RMSE Scores:", cv_rmse)
print(f"📌 Mean RMSE: {cv_rmse.mean():.4f}")
print(f"📌 Standard Deviation RMSE: {cv_rmse.std():.4f}")
print(f"\n📌 Final Test RMSE: {test_rmse:.4f}")
print("📌 Baseline RMSE (before feature engineering): 21361.6215")
