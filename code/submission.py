import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from structured_preprocessing import preprocess_housing_data, CATEGORY_MAPPING, NUMERIC_MAPPING, ordinal_mappings
from feature_engineering_model_testing import feature_enggeniring_pipeline
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# … (same imports and directory setup) …
test_data_path = os.path.join(project_dir, "data", "test.csv")
train_data_path = os.path.join(project_dir, "data", "train.csv")
# 1. Load and preprocess
df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

pp_train = preprocess_housing_data(df_train, is_train=True)
X_train = feature_enggeniring_pipeline(pp_train, is_train=True)
y_train = np.log1p(X_train.pop("SalePrice"))

pp_test = preprocess_housing_data(df_test, is_train=False)
X_test = feature_enggeniring_pipeline(pp_test, is_train=False)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
# 2. Scale features properly
scaler = StandardScaler()
scaler.fit(X_train)


X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. (Optional) Quick CV to check performance
ridge = Ridge(alpha=100)
scores = cross_val_score(ridge, X_train_s, y_train,
                         cv=5,
                         scoring="neg_root_mean_squared_error")
rmse = -scores
print(f"CV RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}")

# 4. Retrain on all data
ridge.fit(X_train_s, y_train)

# 5. Predict on test set and back‑transform
y_pred_log = ridge.predict(X_test_s)
y_pred = np.expm1(y_pred_log)

# 6. Save submission
submission = pd.DataFrame({
    "Id": df_test["Id"],
    "SalePrice": y_pred
})
submission.to_csv(os.path.join(project_dir, "submission.csv"), index=False)
print("✅ Submission saved to submission.csv")
