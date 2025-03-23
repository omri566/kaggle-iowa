import pandas as pd
import numpy as np
from seaborn import heatmap
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns



preprocess_train_data_path = "data/pre_processed_data.csv"

df = pd.read_csv(preprocess_train_data_path)




def feature_enggeniring_pipeline(df):

    #General Drops
    df = df.drop(columns=["PoolArea","PoolQC"]) #drops because low mi and corr at any reassonable combination



    #bathroom features
    df["bathrooms_final"] = (df["BsmtFullBath"] + df["FullBath"] +
                             0.5 * (df["BsmtHalfBath"] + df["HalfBath"]))
    df = df.drop(columns = ["BsmtFullBath","FullBath","BsmtHalfBath","HalfBath"])


    #Garage features
    df["Garage_final"] = (df["GarageFinish"] * df["GarageCars"] *
                          df["GarageArea"])  # GarageArea * GarageCars * GarageFinish
    df = df.drop(columns = ["GarageType","GarageFinish","GarageArea","GarageQual","GarageCars","GarageCond"])


    #size features
    df["total_house_area"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["qual*total_house_area"] = (df["OverallQual"] *
                                   df["total_house_area"])  # need to decide if not to stay as 2 different features
    df = df.drop(columns = ["GrLivArea", "TotalBsmtSF", "total_house_area"])
    # garden built area, deck porch, pergula, etc combined into 1 feature and than dropped - note that enclosed porch was harming the relation to y in both parameters and therefore was simply dropped
    df["semi_built_area"] = (df['WoodDeckSF'] +
                             df['OpenPorchSF'] +
                             df["3SsnPorch"] +
                                df["ScreenPorch"])
    df = df .drop(columns = ["WoodDeckSF", "OpenPorchSF", "3SsnPorch", "ScreenPorch","EnclosedPorch"])
    #kitchen features were tested, number of kitchen was irrevent and transformations gave poor results, decided to drop
    df = df.drop(columns = ["KitchenAbvGr"])
    #bsmt size quality product, BsmtGrade is from general basemnet features and BsmtUnitsGrade relates to different parts in it, they are similar in context and have 0.58 corr with each other
    df["BsmtUnitsGrade"] = df["BsmtFinSF1"] * df["BsmtFinType1"] + df["BsmtFinType2"] * df["BsmtFinSF2"]
    df["BsmtGrade"] = df["BsmtQual"] * df["TotalBsmtSF"] *df["BsmtQual"]
    df  = df.drop(columns = ['BsmtQual','BsmtExposure','BsmtCond','BsmtFinType1','BsmtFinType2',"TotalBsmtSF", "BsmtFinSF2","BsmtFinSF1","BsmtUnfSF"])
    #room to size ratio, first chaging 0 rooms to 1, there are only 6 instances and 0 doesnt seem correct anyway
    df["BedroomAbvGr"] = df["BedroomAbvGr"].replace({0: 1})
    df["rooms_to_size"] = df["total_house_area"] / df["BedroomAbvGr"]
    #land cumulative score
    df["elec_inf"] = df["HeatingQC"] + df["CentralAir"] + df["Electrical"]
    #non basement area divided by none basement rooms, not including bathrooms
    df["rooms_to_size"] = df["GrLivArea"] / (df["BedroomAbvGr"])
    #total basement divided by total house(including bsmt) replaced zero with 1 for log purposes
    df["house_to_bsmt_ratio"] = np.log((df["TotalBsmtSF"] / df["total_house_area"]).replace({0: 1}))