import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



preprocess_train_data_path = "data/pre_processed_data.csv"


worst_43  = [
    "MSZoning_RM",
    "Story_Count",
    "LotConfig_Inside",
    "BsmtHalfBath",
    "3SsnPorch",
    "LowQualFinSF",
    "Exterior2nd_Plywood",
    "MasVnrType_BrkCmn",
    "HouseStyle_1Story",
    "Exterior2nd_Wd Sdng",
    "MasVnrType_None",
    "GarageType_Modified_Other",
    "Exterior2nd_BrkFace",
    "MiscVal",
    "BldgType_TwnhsE",
    "PavedDrive",
    "Exterior1st_CemntBd",
    "GarageType_Modified_BuiltIn",
    "Exterior1st_Other",
    "Foundation_CBlock",
    "Age_Category",
    "GarageType_Modified_2Types",
    "MSZoning_RH",
    "Electrical",
    "HouseStyle_2Story",
    "Exterior1st_WdShing",
    "Exterior2nd_Wd Shng",
    "RoofStyle_Other",
    "BldgType_Twnhs",
    "LotConfig_FR3",
    "PUD_Flag",
    "Exterior2nd_Stucco",
    "Foundation_Slab",
    "Exterior2nd_CmentBd",
    "Exterior2nd_AsbShng",
    "Exterior1st_Stucco",
    "Exterior1st_AsbShng",
    "HouseStyle_SFoyer",
    "Exterior2nd_Other",
    "Foundation_Other",
    "HouseStyle_2.5Fin",
    "HouseStyle_1.5Unf",
    "BldgType_2fmCon",
    "HouseStyle_2.5Unf"
]


def feature_enggeniring_pipeline(df_input):
    df = df_input.copy()

    # General Drops
    df = df.drop(columns=["PoolArea", "PoolQC"])
    df = df.drop(columns=["KitchenAbvGr"])

    # Bathroom features
    df["bathrooms_final"] = (df["BsmtFullBath"] + df["FullBath"] +
                             0.5 * (df["BsmtHalfBath"] + df["HalfBath"]))

    # Garage features
    df["Garage_final"] = df["GarageFinish"] * df["GarageCars"] * df["GarageArea"]

    # Size features
    df["total_house_area"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["qual*total_house_area"] = df["OverallQual"] * df["total_house_area"]

    # Semi-built area
    df["semi_built_area"] = (df['WoodDeckSF'] +
                             df['OpenPorchSF'] +
                             df["3SsnPorch"] +
                             df["ScreenPorch"])

    # Basement features
    df["BsmtUnitsGrade"] = (df["BsmtFinSF1"] * df["BsmtFinType1"] +
                            df["BsmtFinType2"] * df["BsmtFinSF2"])
    df["BsmtGrade"] = df["BsmtQual"] * df["TotalBsmtSF"] * df["BsmtQual"]


    # Electrical infrastructure score
    df["elec_inf"] = df["HeatingQC"] + df["CentralAir"] + df["Electrical"]

    # GrLivArea room ratio (overwrites previous rooms_to_size)
    df["BedroomAbvGr"] = df["BedroomAbvGr"].replace({0: 1})
    df["rooms_to_size"] = df["GrLivArea"] / df["BedroomAbvGr"]

    # House to basement ratio with log
    df["house_to_bsmt_ratio"] = np.log((df["TotalBsmtSF"] / df["total_house_area"]).replace({0: 1}))

    # Land score with log
    df["land_score"] = np.log(
        (df["LandSlope"] + (4 - df["LotShape"]) + (4 - df["LandContour"])) * df["LotArea"])
    df = df.drop(columns= worst_43)
    df.to_csv("feature_engineered_data.csv")

def model_feature_diagnostics(model, X_df, y, sort_by="mean_abs_shap", ascending=False):
    """
    Computes SHAP values, correlation, and MI for all features in a model.

    Args:
        model: Trained model (e.g., XGBoostRegressor)
        X_df (pd.DataFrame): Features used for training
        y (pd.Series): Target variable
        sort_by (str): Column to sort the output DataFrame by ('mean_abs_shap', 'correlation', 'mutual_info')
        ascending (bool): Sort order

    Returns:
        pd.DataFrame: Feature diagnostics DataFrame
    """
    # SHAP values
    explainer = shap.Explainer(model, X_df)
    shap_values = explainer(X_df)

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = pd.Series(data=abs(shap_values.values).mean(axis=0),
                              index=X_df.columns, name="mean_abs_shap")

    # Correlation with y
    correlation = X_df.corrwith(y).rename("correlation")

    # Mutual Information with y
    mi = pd.Series(mutual_info_regression(X_df, y, discrete_features='auto'),
                   index=X_df.columns, name="mutual_info")

    # Combine all into a DataFrame
    diagnostics_df = pd.concat([mean_abs_shap, correlation, mi], axis=1)

    # Sort
    diagnostics_df = diagnostics_df.sort_values(by=sort_by, ascending=ascending)

    return diagnostics_df

feature_enggeniring_pipeline(pd.read_csv(preprocess_train_data_path))
df = pd.read_csv("feature_engineered_data.csv")
X_raw = df.drop(columns=['SalePrice'])
y_raw = df['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=420)
from xgboost import XGBRegressor
# dummy model creation for testing
model = XGBRegressor(n_estimators=300, random_state=420, gamma=1, learning_rate=0.0545, n_jobs=-1, max_depth = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

#model_feature_diagnostics(model, X_train,y_train).to_csv("feature_diagnostics.csv")
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")