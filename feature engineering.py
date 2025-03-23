import pandas as pd
import numpy as np
from seaborn import heatmap
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns



preprocess_train_data_path = "data/pre_processed_data.csv"

df = pd.read_csv(preprocess_train_data_path)




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

    return df




   """
    #drops, not yet completed
    df = df.drop(columns = ["BsmtFullBath","FullBath","BsmtHalfBath","HalfBath"])
    df = df.drop(columns = ["GarageType","GarageFinish","GarageArea","GarageQual","GarageCars","GarageCond"])
    df = df.drop(columns = ["GrLivArea", "TotalBsmtSF", "total_house_area"])
    df = df .drop(columns = ["WoodDeckSF", "OpenPorchSF", "3SsnPorch", "ScreenPorch","EnclosedPorch"])
    df  = df.drop(columns = ['BsmtQual','BsmtExposure','BsmtCond','BsmtFinType1','BsmtFinType2',"TotalBsmtSF", "BsmtFinSF2","BsmtFinSF1","BsmtUnfSF"])
"""


