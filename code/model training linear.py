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

# 📌 Hyperparameter grids
param_grid_ridge = {"alpha": [1, 10, 100]}
param_grid_lasso = {"alpha": [1, 10, 100], "max_iter": [5000, 10000, 20000], "selection": ["cyclic", "random"]}
param_grid_enet = {
    "alpha": [1, 10, 100],
    "l1_ratio": [0.1, 0.5, 0.9],  # Elastic Net mixes Lasso (L1) and Ridge (L2)
    "max_iter": [5000, 10000, 20000]
}


# 📌 Train & Tune Models
def train_and_tune(model, param_grid, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_log)  # Train using log(y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 📌 Save best parameters
    with open(os.path.join(project_dir, f"model_training/{model_name}_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # 📌 Print RMSE for each hyperparameter combination
    print(f"\n🔹 RMSE for {model_name} (cross-validation):")
    for i, params in enumerate(grid_search.cv_results_["params"]):
        rmse = np.sqrt(-grid_search.cv_results_["mean_test_score"][i])  # Convert negative MSE to RMSE
        print(f"Params: {params} -> RMSE: {rmse:.4f}")

    print(f"\n✅ Best Parameters for {model_name}: {best_params}")
    print(f"✅ Best RMSE (log scale): {np.sqrt(-grid_search.best_score_):.4f}")

    return best_model


# 📌 Train all models
best_ridge = train_and_tune(Ridge(), param_grid_ridge, "ridge")
#best_lasso = train_and_tune(Lasso(), param_grid_lasso, "lasso")
#best_enet = train_and_tune(ElasticNet(), param_grid_enet, "elasticnet")


# 📌 Evaluate Models on Test Data
def evaluate_model(model, model_name):
    y_pred_log = model.predict(X_test_scaled)  # Predictions in log scale
    y_pred = np.expm1(y_pred_log)  # Inverse transform to get real values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n📌 True RMSE on Test Set for {model_name}: {rmse:.4f}")


evaluate_model(best_ridge, "Ridge")
#evaluate_model(best_lasso, "Lasso")
#evaluate_model(best_enet, "Elastic Net")
