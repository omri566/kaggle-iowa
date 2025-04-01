import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import os
from scipy.stats import zscore
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler


"""
weak_features = [
    "Story_Count", "ExterQual", "PUD_Flag", "Exterior2nd_MetalSd", "Exterior2nd_HdBoard",
    "MasVnrType_BrkFace", "Exterior2nd_VinylSd", "RoofStyle_Gable", "LotConfig_CulDSac",
    "LotConfig_FR2", "Exterior1st_VinylSd", "BldgType_Twnhs", "Exterior1st_Plywood",
    "Exterior2nd_Plywood", "MasVnrType_Stone", "BsmtFullBath", "LotConfig_Inside",
    "HouseStyle_1.5Fin", "Exterior1st_HdBoard", "Exterior1st_MetalSd", "LotConfig_Corner",
    "BldgType_Duplex", "RoofStyle_Hip", "HouseStyle_1Story", "HouseStyle_SLvl",
    "house_to_bsmt_ratio", "GarageType_Modified_BuiltIn", "RoofStyle_Other",
    "Exterior2nd_CmentBd", "MSZoning_C (all)", "Exterior2nd_AsbShng", "HouseStyle_2.5Unf",
    "Exterior1st_AsbShng", "Exterior2nd_Wd Shng", "Exterior2nd_Other",
    "GarageType_Modified_Other", "Exterior1st_Stucco", "Exterior2nd_BrkFace",
    "Exterior1st_CemntBd", "BldgType_TwnhsE", "LotConfig_FR3", "Foundation_Other",
    "Exterior1st_Other", "Exterior1st_WdShing", "Foundation_Slab",
    "GarageType_Modified_2Types", "MSZoning_RH", "BldgType_2fmCon", "HouseStyle_1.5Unf",
    "MasVnrType_BrkCmn", "Exterior2nd_Stucco", "HouseStyle_2.5Fin", "HouseStyle_SFoyer",
    "LandSlope", "ExterCond", "GarageCond", "GarageQual", "LandContour", "BsmtCond",
    "BsmtHalfBath", "LowQualFinSF", "Electrical", "BsmtFinType2", "BsmtFinSF2",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "Fence", "MiscVal", "Functional"
]
"""

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up to the project directory
project_dir = os.path.dirname(script_dir)

# Construct the path to the 'data' folder
data_path = os.path.join(project_dir, "data")

preprocess_train_data_path =  data_path +"/pre_processed_data.csv"


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


"""def detect_and_clip_outliers(df, feature, factor=1.5):
    # Calculate Q1, Q3, and IQR
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Check if there are any outliers in the feature
    outliers_exist = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).any()
    if outliers_exist:
        print(
            f"Outliers detected in '{feature}'. Capping values at [{lower_bound:.2f}, {upper_bound:.2f}]. Lowest value found: {df[feature].min():.2f}")
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

    return df
"""


def detect_and_clip_outliers(df, target_col = "SalePrice", factor=1.5):
    """
    Clips outliers in numeric features using the IQR method,
    skipping binary features and the specified target column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column to exclude.
        factor (float): Multiplier for IQR to define outlier bounds.

    Returns:
        pd.DataFrame: The DataFrame with capped outliers.
    """
    df = df.copy()  # Avoid modifying original DataFrame

    # Identify numeric features, excluding the target
    numeric_features = df.select_dtypes(include=["float", "int"]).columns
    numeric_features = [col for col in numeric_features if col != target_col]

    # Identify binary features (e.g., one-hot encoded columns)
    binary_features = [col for col in numeric_features if df[col].dropna().nunique() == 2]

    # Filter to features that are neither binary nor the target
    features_to_clip = [col for col in numeric_features if col not in binary_features]

    for feature in features_to_clip:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Avoid negative lower bounds if the feature should be non-negative
        lower_bound = max(lower_bound, 0)

        outliers_exist = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).any()
        if outliers_exist:
            print(
                f"Outliers detected in '{feature}'. "
                f"Capping values at [{lower_bound:.2f}, {upper_bound:.2f}]. "
                f"Lowest value found: {df[feature].min():.2f}, "
                f"Highest value found: {df[feature].max():.2f}"
            )
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        else:
            print(f"No outliers detected in '{feature}'.")

    # Log skipped binary features
    for binary_col in binary_features:
        print(f"⏭️ Skipping binary feature '{binary_col}' (likely one-hot encoded).")

    print(f"\n✅ Outlier capping complete. {len(features_to_clip)} features clipped.")
    return df


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
    df["BedroomAbvGr"] = (df["BedroomAbvGr"]+1)
    df["rooms_to_size"] = df["GrLivArea"] / df["BedroomAbvGr"]

    # House to basement ratio with log
    df["house_to_bsmt_ratio"] = (df["TotalBsmtSF"] / df["total_house_area"])

    # Land score with log
    df["land_score"] = np.log(
        (df["LandSlope"] + (5 - df["LotShape"]) + (5 - df["LandContour"])) * df["LotArea"])
    df = df.drop(columns= worst_43)
    #used_features = ["LotShape","LandContour","BedroomAbvGr","BsmtFinType2","bsmtfin","3SsnPorch",]
    #df = df.drop(columns=used_features,errors="ignore")

    df = detect_and_clip_outliers(df)

    #exporting the df to data folder, ensuring smooth operation after changing the pipeline

    df.to_csv(data_path+"/feature_engineered_data.csv")


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
df = pd.read_csv(data_path+"/feature_engineered_data.csv")
X_raw = df.drop(columns=['SalePrice'])
y_raw = df['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=420)


# dummy model creation for testing
# 1 - XGB

from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, random_state=420, gamma=1, learning_rate=0.0545, n_jobs=-1, max_depth = 6)
model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_val)



#model_feature_diagnostics(model, X_train,y_train).to_csv("feature_diagnostics.csv")
rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"xbg Validation RMSE: {rmse:.2f}")