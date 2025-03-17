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
    df = df.dtop(columns = ["GarageType","GarageFinish","GarageArea","GarageQual","GarageCars","GarageCond"])


    #size features
    df["total_house_area"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["qual*total_house_area"] = (df["OverallQual"] *
                                   df["total_house_area"])  # need to decide if not to stay as 2 different features
    df = df.drop(columns = ["GrLivArea", "TotalBsmtSF", "total_house_area"])