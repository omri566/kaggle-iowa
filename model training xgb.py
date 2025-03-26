import json
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import random
from sklearn.metrics import mean_squared_error

# Load your preprocessed dataset
df = pd.read_csv("data/feature_engineered_data.csv")
print(len(df))
# Split features & target variable
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test , y_train , y_test = train_test_split(X,y , test_size=0.2, random_state=420)

# Set up 5-fold Cross-Validation (since data < 1500 rows)
kf = KFold(n_splits=5, shuffle=True, random_state=420)

#function to generate random parameters to fine tune
def get_random_params(base_params):
    """
    Generates a slightly modified version of the given hyperparameters.

    - Increases/decreases integer parameters by ±20%.
    - Adjusts float parameters by ±10-20%.

    Parameters:
    - base_params (dict): Original estimated parameters.

    Returns:
    - dict: Randomly adjusted parameters.
    """
    new_params = {}

    for key, value in base_params.items():
        if isinstance(value, int):  # Integer parameters (e.g., max_depth, n_estimators)
            change = max(1, int(value * random.uniform(0.8, 1.2)))  # ±20%
            new_params[key] = change

        elif isinstance(value, float):  # Float parameters (e.g., learning_rate)
            change = round(value * random.uniform(0.8, 1.2), 4)  # ±10-20%, rounded to 4 decimals
            new_params[key] = change

        else:
            new_params[key] = value  # If it's something else, keep it unchanged

    return new_params

# Load parameters from JSON
with open("data/first_xgb_param.json", "r") as f:
    first_params = json.load(f)

#this functions get parmeters to tweek and num_trials and return the best parameters and save it to json
def fine_tune(params,num_trials):
    best_rmse = float("inf")
    best_params = None

    for i in range(num_trials):
        #generating parameters and initializing the model
        new_params = get_random_params(params)
        model = xgb.XGBRegressor(random_state=42,objective="reg:squarederror",eval_metric="rmse", **new_params)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"Trial {i+1}: RMSE = {rmse}, Params = {new_params}")

        # Save the best parameters
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = new_params

    # Save the best fine-tuned parameters
    with open("data/best_xgb_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\nBest parameters saved to data/best_xgb_params.json with RMSE = {best_rmse}")
    return best_params

#fine_tune(first_params,50)

with open("data/best_xgb_params.json","r") as f:
    best_params = json.load(f)


# Initialize XGBoost model with default parameters
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=400,
    **best_params
)

print(best_params)

# Perform Cross-Validation & Compute RMSE
cv_rmse = -cross_val_score(xgb.XGBRegressor(objective="reg:squarederror",random_state=400,**best_params),
                           X, y, cv=5, scoring="neg_root_mean_squared_error")



# Print Results
print(f"Cross-Validation RMSE Scores: {cv_rmse}")
print(f"Mean RMSE: {cv_rmse.mean():.4f}")
print(f"Standard Deviation RMSE: {cv_rmse.std():.4f}")
print("21361.6215 is the rmse of the pre data")
