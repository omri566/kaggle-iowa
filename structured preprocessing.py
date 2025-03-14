import pandas as pd
pd.set_option('display.max_columns', None) # allows to see all of the columns of a pandas output works similarly for rows
# Embedded JSON mappings (from category_analysis.json and num_analysis.json)

import os

train_data_path = "data/train.csv"
df = pd.read_csv(train_data_path)

#map of each catagorical column with it's preproccesing recomendation
CATEGORY_MAPPING = {
    "LotConfig": {"action": "one-hot","certainty": "sure","comment": "LotConfig has no clear ranking, so we apply one-hot encoding."},
    "MSZoning": {"action": "one-hot", "certainty": "probably", "comment": "No nulls, most values are low and medium density"},
    "Street": {"action": "drop", "certainty": "sure", "comment": "Almost all values are 'Pave', likely drop"},
    "Alley": {"action": "drop", "certainty": "sure", "comment": "94% nulls, should drop"},
    "LotShape": {"action": "one-hot", "certainty": "sure", "comment": "No nulls, might combine IR2 and IR3"},
    "LandContour": {"action": "one-hot", "certainty": "maybe", "comment": "No nulls, over 80% in one category"},
    "Utilities": {"action": "drop", "certainty": "sure", "comment": "No nulls, 99% same category"},
    "Condition1": {"action": "drop", "certainty": "probably", "comment": "86% in a single category, not very interesting"},
    "Condition2": {"action": "drop", "certainty": "sure", "comment": "99% in a single category"},
    "BldgType": {"action": "one-hot", "certainty": "probably", "comment": "Bad distribution but valuable context"},
    "HouseStyle": {"action": "one-hot", "certainty": "sure", "comment": "Important context, consider merging low-frequency categories"},
    "RoofStyle": {"action": "one-hot", "certainty": "sure", "comment": "No nulls, good context"},
    "RoofMatl": {"action": "drop", "certainty": "sure", "comment": "99% in a single category"},
    "Exterior1st": {"action": "one-hot", "certainty": "sure", "comment": "Top 5 categories rule 87% of data"},
    "Exterior2nd": {"action": "one-hot", "certainty": "sure", "comment": "Consider integrating with Exterior1"},
    "MasVnrType": {"action": "one-hot", "certainty": "sure", "comment": "60% nulls, replace NaNs with 'None'"},
    "ExterCond": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "ExterQual": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "Foundation": {"action": "one-hot", "certainty": "sure", "comment": "No nulls, poor distribution"},
    "BsmtQual": {"action": "ordinal", "certainty": "sure", "comment": "Replace nulls with 0, logical order"},
    "BsmtCond": {"action": "ordinal", "certainty": "sure", "comment": "Replace nulls with 0, logical order"},
    "BsmtExposure": {"action": "ordinal", "certainty": "sure", "comment": "Change NaNs to -1 to differentiate"},
    "BsmtFinType1": {"action": "ordinal", "certainty": "sure", "comment": "Set nulls to -1"},
    "BsmtFinType2": {"action": "ordinal", "certainty": "sure", "comment": "Set nulls to -1"},
    "Heating": {"action": "drop", "certainty": "sure", "comment": "99% in a single category"},
    "HeatingQC": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "CentralAir": {"action": "binary", "certainty": "sure", "comment": "No nulls, reshape to 0/1"},
    "Electrical": {"action": "ordinal", "certainty": "sure", "comment": "1 null, logical order"},
    "KitchenQual": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "Functional": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "FireplaceQu": {"action": "ordinal", "certainty": "sure", "comment": "50% nulls, replace with 0"},
    "GarageType_Modified": {"action": "one-hot", "certainty": "maybe", "comment": "81 nulls, merged two low corr to 'other'"},
    "GarageFinish": {"action": "ordinal", "certainty": "sure", "comment": "81 nulls, set nulls to -1"},
    "GarageQual": {"action": "ordinal", "certainty": "sure", "comment": "Set nulls to -1"},
    "GarageCond": {"action": "ordinal", "certainty": "sure", "comment": "Change nulls to 0"},
    "PavedDrive": {"action": "binary", "certainty": "probably", "comment": "Consider merging N and P, then binary"},
    "PoolQC": {"action": "ordinal", "certainty": "probably", "comment": "99% nulls, change to 0"},
    "Fence": {"action": "ordinal", "certainty": "sure", "comment": "80% nulls, set nulls to 0"},
    "MiscFeature": {"action": "drop", "certainty": "probably", "comment": "96% nulls, drop or OH"},
    "SaleType": {"action": "drop", "certainty": "sure", "comment": "Potential data leakage"},
    "SaleCondition": {"action": "drop", "certainty": "sure", "comment": "Potential data leakage"},
    "Age_Category": {"action": "ordinal", "certainty": "sure", "comment": "Derived from MSSubClass to categorize homes by age group"}


}

#each numerical column with it's preprocsessing
NUMERIC_MAPPING = {
    "Id": {
        "action": "keep_as_is",
        "note": ""
    },
    "LotFrontage": {
        "action": "skip_for_now",
        "note": ""
    },
    "LotArea": {
        "action": "impute_with_median",
        "note": "a big difference in lot size would have direct relation to it's price, no tranformations except for nulls"
    },
    "OverallQual": {
        "action": "keep_as_is",
        "note": "an ordinal category ranging from 0-10, looks good"
    },
    "OverallCond": {
        "action": "keep_as_is",
        "note": "keep as it is, similar to overallquality, check for corr"
    },
    "YearBuilt": {
        "action": "keep_as_is",
        "note": "consider dropping and creating a new feature age(curr_year - Original construction date)"
    },
    "YearRemodAdd": {
        "action": "keep_as_is",
        "note": "consider creating a new feature years since renovation current_year - max(remodel date, Original construction date)"
    },
    "MasVnrArea": {
        "action": "impute_with_0",
        "note": "make sure to check that extreme values are possible"
    },
    "BsmtFinSF1": {
        "action": "keep_as_is",
        "note": "check for corr with BsmtFinSF2, make sure that it is 0 for BsmtFinType1 = 0"
    },
    "BsmtFinSF2": {
        "action": "keep_as_is",
        "note": "check for corr with BsmtFinSF1, make sure that it is 0 for BsmtFinType2 = 0"
    },
    "BsmtUnfSF": {
        "action": "n",
        "note": ""
    },
    "TotalBsmtSF": {
        "action": "keep_as_is",
        "note": ""
    },
    "1stFlrSF": {
        "action": "keep_as_is",
        "note": ""
    },
    "2ndFlrSF": {
        "action": "keep_as_is",
        "note": ""
    },
    "LowQualFinSF": {
        "action": "drop_feature",
        "note": "almost only zeros, make sure later"
    },
    "GrLivArea": {
        "action": "keep_as_is",
        "note": ""
    },
    "BsmtFullBath": {
        "action": "keep_as_is",
        "note": "consider joining with other bathroom fields for a new total_bathroom feature(if doesn't exist)"
    },
    "BsmtHalfBath": {
        "action": "keep_as_is",
        "note": "consider joining with BsmtFullBath for total bsmt bathroom"
    },
    "FullBath": {
        "action": "keep_as_is",
        "note": ""
    },
    "HalfBath": {
        "action": "keep_as_is",
        "note": "consider joining with FullBath"
    },
    "BedroomAbvGr": {
        "action": "keep_as_is",
        "note": "can be added with bsmt bedrooms to total house bedrooms, idea: a new feature(has living unit) for houses with more than 1 bedroom and bathroom in bsmt"
    },
    "KitchenAbvGr": {
        "action": "keep_as_is",
        "note": ""
    },
    "TotRmsAbvGrd": {
        "action": "keep_as_is",
        "note": ""
    },
    "GarageYrBlt": {
        "action": "impute_with_0",
        "note": "we could also make sure that every house that doesn't have a gararge will have 0 in this field and if a house does have a grage but has 0 change it to the year built or year modified"
    },
    "GarageCars": {
        "action": "keep_as_is",
        "note": "make sure it correlates with other garage features"
    },
    "GarageArea": {
        "action": "keep_as_is",
        "note": ""
    },
    "WoodDeckSF": {
        "action": "keep_as_is",
        "note": ""
    },
    "OpenPorchSF": {
        "action": "keep_as_is",
        "note": ""
    },
    "EnclosedPorch": {
        "action": "keep_as_is",
        "note": ""
    },
    "3SsnPorch": {
        "action": "keep_as_is",
        "note": ""
    },
    "ScreenPorch": {
        "action": "keep_as_is",
        "note": ""
    },
    "PoolArea": {
        "action": "keep_as_is",
        "note": "make sure it correlates with other pool features"
    },
    "MiscVal": {
        "action": "keep_as_is",
        "note": "consider capping the extreme value"
    },
    "MoSold": {
        "action": "drop_feature",
        "note": "flag for data leak"
    },
    "YrSold": {
        "action": "drop_feature",
        "note": "flag for data leak"
    },
    "SalePrice": {
        "action": "drop_feature",
        "note": "this is the label"
    },
    "Neighborhood": {
        "action": "keep_as_is",
        "certainty": "sure",
        "comment": "No nulls, well distributed, **note** was moved here from the cat mapping after altering the "
                   "values into the mean of the neighborhood saleprice"},
}

# a mapping for all the ordinal features
ordinal_mappings = {
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"None": -1, "No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"None": -1, "Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
    "BsmtFinType2": {"None": -1, "Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Functional": {"Sal": 1, "Sev": 2, "Maj1": 3, "Maj2": 4, "Min1": 5, "Min2": 6, "Mod": 7, "Typ": 8},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Fence": {"None": 0, "MnPrv": 1, "GdPrv": 2, "MnWw": 3, "GdWo": 4},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PavedDrive": {"N": 0, "Y": 1},  # ✅ Only Y/N after replacing "P" with "Y"
    "CentralAir": {"N": 0, "Y": 1},
    "Electrical": {"None":-1,"Mix": 0, "FuseP": 1, "FuseF": 2, "FuseA": 3, "SBrkr": 4},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "OverallQual": {i: i for i in range(1, 11)},  # 1-10 mapping
    "OverallCond": {i: i for i in range(1, 11)},  # 1-10 mapping
    "Age_Category": {"Newer": 2, "Mixed": 1, "Older": 0},
    "GarageFinish": {"None": 0,"Unf": 1,"RFn": 2,"Fin": 3}

}



def preprocess_housing_data(df):
    """
    Preprocesses the housing dataset:
    - Drops unnecessary features
    - Handles missing values
    - Applies categorical transformations (binary, ordinal, one-hot encoding)

    Parameters:
    - df (pd.DataFrame): The input dataset (train or test)

    Returns:
    - df (pd.DataFrame): Preprocessed dataset
    """
    # Extract feature lists
    drop_features = [f for f, details in CATEGORY_MAPPING.items() if details.get("action") == "drop"]
    one_hot_features = [f for f, details in CATEGORY_MAPPING.items() if details.get("action") == "one-hot"]
    ordinal_features = [f for f, details in CATEGORY_MAPPING.items() if details.get("action") == "ordinal"]
    binary_features = [f for f, details in CATEGORY_MAPPING.items() if details.get("action") == "binary"]

    # Drop unnecessary features

    df = df.drop(columns=drop_features, errors='ignore')

    # special Handle missing values
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearRemodAdd"]).fillna(df["YearBuilt"]).fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())

    # Fill categorical missing values
    for col, details in CATEGORY_MAPPING.items():
        if col in df.columns and df[col].isna().sum() > 0:
            if details.get("action") in ["one-hot", "ordinal"]:
                df[col] = df[col].fillna("None")
    #final steps for binary features
    #merging two entries into one after checking for corr and MI

    df["GarageType_Modified"] = df["GarageType"].replace({"CarPort": "Other", "Basment": "Other"}).infer_objects(copy=False)
    df.drop(columns=["GarageType"], inplace=True)
    #merging partial pavement into yes and making the feature binary
    df["PavedDrive"] = df["PavedDrive"].replace({"P": "Y"}).infer_objects(copy=False)
    # Binary encoding
    for col in binary_features:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0})
    #final steps for OH features
    # Merge rare categories in RoofStyle
    df["RoofStyle"] = df["RoofStyle"].replace(
        {"Flat": "Other", "Gambrel": "Other", "Mansard": "Other", "Shed": "Other"}).infer_objects(copy=False)

    # Merge rare categories in Exterior1st & Exterior2nd
    rare_exterior1st = df["Exterior1st"].value_counts()[df["Exterior1st"].value_counts() < 20].index
    rare_exterior2nd = df["Exterior2nd"].value_counts()[df["Exterior2nd"].value_counts() < 20].index

    df["Exterior1st"] = df["Exterior1st"].replace(rare_exterior1st, "Other").infer_objects(copy=False)
    df["Exterior2nd"] = df["Exterior2nd"].replace(rare_exterior2nd, "Other").infer_objects(copy=False)

    # a dictionary containing the mean saleprice per neighborhood(shouldn't cause dataleakage but consider dropping in fine tuning)
    # Hardcoded mean SalePrice per Neighborhood (from train set)
    NEIGHBORHOOD_MEAN_PRICES = {
        "Blmngtn": 194870.88,
        "Blueste": 137500.00,
        "BrDale": 104493.75,
        "BrkSide": 124834.05,
        "ClearCr": 212565.43,
        "CollgCr": 197965.77,
        "Crawfor": 210624.73,
        "Edwards": 128219.70,
        "Gilbert": 192854.51,
        "IDOTRR": 100123.78,
        "MeadowV": 98576.47,
        "Mitchel": 156270.12,
        "NAmes": 145847.08,
        "NPkVill": 142694.44,
        "NWAmes": 189050.07,
        "NoRidge": 335295.32,
        "NridgHt": 316270.62,
        "OldTown": 128225.30,
        "SWISU": 142591.36,
        "Sawyer": 136793.14,
        "SawyerW": 186555.80,
        "Somerst": 225379.84,
        "StoneBr": 310499.00,
        "Timber": 242247.45,
        "Veenker": 238772.73
    }
    # Compute overall mean to handle unknown neighborhoods in test set
    OVERALL_TRAIN_MEAN = sum(NEIGHBORHOOD_MEAN_PRICES.values()) / len(NEIGHBORHOOD_MEAN_PRICES)
    # Apply the precomputed neighborhood mean values
    df["Neighborhood"] = df["Neighborhood"].map(NEIGHBORHOOD_MEAN_PRICES)

    # Handle unseen neighborhoods (test set case)
    df["Neighborhood"] = df["Neighborhood"].fillna(OVERALL_TRAIN_MEAN)
    # ***IMPORTANT*** AFTER THESE CHANGES THE FEATURE WAS NO LONGER A CATEGORICAL AND WAS MANUALLY CHANGED IN THE HARD CODE ABOVE
    #this difficult feature was divided into 3 different ones and than dropped
    # Map MSSubClass to Story_Count (Numerical - Ordinal)
    story_map = {
        20: 1, 30: 1, 40: 1,  # 1-Story (including finished attic)
        45: 1.5, 50: 1.5,  # 1.5-Story
        60: 2, 70: 2,  # 2-Story
        75: 2.5,  # 2.5-Story
        80: 2.5,  # Split/Multi-Level (assumed similar to 2.5)
        85: 2.5,  # Split Foyer (assumed similar to 2.5)
        90: 2,  # Duplex (assuming treated like 2-story)
        120: 1,  # 1-Story PUD
        150: 1.5,  # 1.5-Story PUD
        160: 2,  # 2-Story PUD
        180: 2.5,  # PUD Multi-Level (assumed similar to 2.5-Story)
        190: 2  # 2-Family Conversion (assuming treated like 2-story)
    }

    df["Story_Count"] = df["MSSubClass"].replace(story_map).infer_objects(copy=False)

    # Map MSSubClass to Age_Category (Categorical - Nominal)
    age_map = {
        20: "Newer",  # 1-STORY 1946 & NEWER
        30: "Older",  # 1-STORY 1945 & OLDER
        40: "Mixed",  # 1-STORY W/FINISHED ATTIC ALL AGES
        45: "Mixed",  # 1.5-STORY UNFINISHED ALL AGES
        50: "Mixed",  # 1.5-STORY FINISHED ALL AGES
        60: "Newer",  # 2-STORY 1946 & NEWER
        70: "Older",  # 2-STORY 1945 & OLDER
        75: "Mixed",  # 2.5-STORY ALL AGES
        80: "Mixed",  # SPLIT OR MULTI-LEVEL
        85: "Mixed",  # SPLIT FOYER
        90: "Mixed",  # DUPLEX - ALL STYLES AND AGES
        120: "Newer",  # 1-STORY PUD (Planned Unit Development) - 1946 & NEWER
        150: "Mixed",  # 1.5-STORY PUD - ALL AGES
        160: "Newer",  # 2-STORY PUD - 1946 & NEWER
        180: "Mixed",  # PUD - MULTILEVEL - INCL SPLIT LEVEL/FOYER
        190: "Older"  # 2 FAMILY CONVERSION - ALL STYLES AND AGES
    }

    df["Age_Category"] = df["MSSubClass"].replace(age_map).infer_objects(copy=False)

    # Create PUD_Flag (Binary)
    df["PUD_Flag"] = df["MSSubClass"].apply(lambda x: 1 if x in {120, 150, 160} else 0)

    # Drop the original MSSubClass column
    df.drop(columns=["MSSubClass"], inplace=True)


    #dealing with exterior 1st and 2nd
    threshold = 20
    # Find rare categories for Exterior1st
    rare_exterior1st = df["Exterior1st"].value_counts()[df["Exterior1st"].value_counts() < threshold].index
    df["Exterior1st"] = df["Exterior1st"].replace(rare_exterior1st, "Other").infer_objects(copy=False)

    # Find rare categories for Exterior2nd
    rare_exterior2nd = df["Exterior2nd"].value_counts()[df["Exterior2nd"].value_counts() < threshold].index
    df["Exterior2nd"] = df["Exterior2nd"].replace(rare_exterior2nd, "Other").infer_objects(copy=False)

    # Merge rare categories in RoofStyle
    df["RoofStyle"] = df["RoofStyle"].replace(
        {"Flat": "Other", "Gambrel": "Other", "Mansard": "Other", "Shed": "Other"}).infer_objects(copy=False)

    # Merge rare categories in Foundation
    df["Foundation"] = df["Foundation"].replace({"Stone": "Other", "Wood": "Other"}).infer_objects(copy=False)
    #extract OH features from json
    ohe_features = [f for f, details in CATEGORY_MAPPING.items() if details["action"] == "one-hot"]

    # Apply One-Hot Encoding
    df = pd.get_dummies(df, columns=ohe_features, dtype=float)



    #ordinals
    # Apply the mappings for ordinal features
    #needed this messy loop beacaues pandas gonna remove the silent casting of the replace function so the
    # line below was added to supprese the warrnings
    pd.set_option('future.no_silent_downcasting', True)

    for feature, mapping in ordinal_mappings.items():
        if feature in df.columns:
            # Perform replacement first
            df[feature] = df[feature].replace(mapping)

            # Determine correct type (int if all values are integers, else float)
            dtype = pd.Series(mapping).dtype  # Get the dtype from mapping

            # Convert the column explicitly to prevent FutureWarning
            if dtype == 'int64' or dtype == 'int32':  # Ensure integer casting
                df[feature] = df[feature].astype(int)
            else:
                df[feature] = df[feature].astype(float)  # Use float if necessary

    return df #return preprocessed df without null values and only int and float column's types


test1 = preprocess_housing_data(df)
test1.to_csv("data/pre_processed_data.csv", index=False)
test1.describe(include="all").transpose().to_csv("data/description_pre_processed.csv")





"""
insights about pre processing:
1. after every decision add what ever was made(droping columns, imputing nulls etc) to a main function to keep everything organsied for the test tat preprocessing
2. if working one at a time on features keep track of what was has been done
3. if working one feature at a time make sure to test for unexpected output, even on trivial things
4. after every process on a feature, make sure that there are no nulls
    4.1. ordinal - make sure the mapping is comprehensive and includes all unique values and not just what the description mentions
    4.2. one-hot - make sure that the cardinality is relatively low
5. when mapping what to do with each column use a dictionary with closed format (one-hot,ordinal,drop, etc) to make sure that it can be used wihtout fear of mismatching or missing features
6. keep more information in the notes! it is useful
7. need to find a solution for dynamic updating of the mapping jsons (moving a numerical to catgeorical like we had with MSSubClass, etc)    
8. if you split a feature make sure to add the new features to the relevant jsons or they will not be treated


"""