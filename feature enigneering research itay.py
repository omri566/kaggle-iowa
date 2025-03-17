import pandas as pd
import numpy as np
from seaborn import heatmap
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

#paths
corr_mat_path = "data/correlation_matrix.csv"
preprocess_train_data_path = "data/pre_processed_data.csv"
mi_mat_path = "data/mutual_information_matrix.csv"
#uploading files
df = pd.read_csv(preprocess_train_data_path)
corr_mat = pd.read_csv(corr_mat_path)
mi_mat = pd.read_csv(mi_mat_path)
# Split features & target variable
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

#creating a copy of x for experimenting
working_df = X.copy()


# creating an assesment function that will allow to test a new feature for corr and mi with the label
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def feature_assessment(feature_input, label, full_df=working_df, sort_by='correlation', ascending=False):
    def feature_assessment(feature_input, label, full_df=None, sort_by='correlation', ascending=False):
        """
        Takes a feature vector or multiple features (Series, DataFrame, or list of columns/Series)
        and returns their correlation and mutual information with the label.

        Args:
            feature_input:
                - A pandas Series (single feature),
                - A pandas DataFrame (multiple features), or
                - A list of Series or column names (str).
                  If list of column names, `full_df` must be provided, working_df is set as default.

            label:
                - A pandas Series representing the target feature to assess dependency against.

            full_df:
                - Optional. Required if `feature_input` is a list of column names.
                - A pandas DataFrame containing all available features.
                - It is advised to set a default df

            sort_by:
                - Optional. 'correlation' or 'mutual_info' to sort the output DataFrame.
                - Default is 'correlation'.

            ascending:
                - Boolean. True to sort ascending, False for descending.
                - Default is False (i.e., highest values first).

        Returns:
            pd.DataFrame:
                - Indexed by feature name.
                - Columns:
                    'correlation': Pearson correlation between feature and label.
                    'mutual_info': Estimated mutual information (non-linear dependency).

        Notes:
            - Mutual information is computed using sklearn's `mutual_info_regression`.
            - Handles flexible inputs for convenient exploration and assessment.
        """

    if isinstance(feature_input, list):
        if all(isinstance(item, pd.Series) for item in feature_input):
            feature_df = pd.concat(feature_input, axis=1)
        elif all(isinstance(item, str) for item in feature_input):
            if full_df is None:
                raise ValueError("To use column names, provide 'full_df' with all features.")
            feature_df = full_df[feature_input]
        else:
            raise ValueError("List must contain all Series or all column names (str).")
    #tranforms seris to df
    elif isinstance(feature_input, pd.Series):
        feature_df = feature_input.to_frame()
    # changes nothing just sets into a general parameter
    elif isinstance(feature_input, pd.DataFrame):
        feature_df = feature_input

    else:
        raise ValueError("Input must be Series, DataFrame, or list of Series/column names.")

    # Correlation
    correlation = feature_df.corrwith(label)

    # Mutual Information (ensure numeric values, avoid NaNs)
    mi = mutual_info_regression(feature_df, label)

    # Output DataFrame
    output = pd.DataFrame({
        "feature_name": feature_df.columns,
        "correlation": correlation.values,
        "mutual_info": mi
    }).set_index("feature_name")

    # Sorting
    if sort_by in ['correlation', 'mutual_info']:
        output = output.sort_values(by=sort_by, ascending=ascending)

    return output


#trials for new features
# logical transformations

#total house area, adding the bsmt area and the non basement area
#all the features below will be in the final pipeline
working_df["total_house_area"] = working_df["GrLivArea"] + working_df["TotalBsmtSF"]
working_df["qual*total_house_area"] = working_df["OverallQual"]*working_df["total_house_area"] #need to decide if not to stay as 2 different features
working_df["Garage_final"] = working_df["GarageFinish"]*working_df["GarageCars"]*working_df["GarageArea"] #GarageArea * GarageCars * GarageFinish
working_df["GarageYrBlt"] = working_df["GarageYrBlt"]
working_df["bathrooms_final"] = working_df["BsmtFullBath"] + working_df["FullBath"] + 0.5*(working_df["BsmtHalfBath"] + working_df["HalfBath"])
#till here

working_df["lot_new"] = working_df["LotArea"]+working_df["LotFron"]
working_df["lot_new_2"] = working_df["LotFrontage"]/working_df["LotShape"]

print(feature_assessment(["lot_new","LotFrontage","LotArea","LotShape","lot_new_2"],y))
#below good way to do log feature
#working_df["Garage_finish_size_2"] = working_df["Garage_finish_size_2"].apply(lambda x: np.log(x) if x > 0 else 0)

















#checking mutual info and correlation to y for every attribute
def analyze_feature_importance(X, y):
    """
    Computes correlation and mutual information between each feature and the target variable.

    Parameters:
    - X: Feature dataframe
    - y: Target variable (series)

    Returns:
    - DataFrame with correlation and mutual information scores
    """
    # Compute Pearson correlation
    corr = X.corrwith(y)

    # Compute Mutual Information (MI)
    mi_scores = mutual_info_regression(X, y, discrete_features='auto', random_state=42)

    # Combine results into a DataFrame
    feature_importance = pd.DataFrame({
        "Correlation": corr,
        "Mutual Information": mi_scores
    })

    # Sort by highest MI scores
    feature_importance.sort_values(by="Mutual Information", ascending=False, inplace=True)

    return feature_importance

feature_importance = analyze_feature_importance(X,y)
feature_importance.to_csv("data/features_to_y_info.csv")
# Example Usage:
"""
feature_importance = analyze_feature_importance(X, y)
pd.set_option("display.max_rows", None)  # Show all rows
print(feature_importance)
pd.reset_option("display.max_rows")  # Reset to default after printing
"""

# checking for every x to every x the correlation and mutual info
def analyze_feature_dependencies(X, y, save_as_csv=True):
    """
    Computes and saves two matrices:
    1. Pearson correlation matrix (linear relationships)
    2. Mutual information matrix (non-linear dependencies)

    Parameters:
    - X: Feature DataFrame
    - y: Target variable
    - save_as_csv: Whether to save matrices as CSV files (default=True)

    Returns:
    - corr_matrix: Pearson correlation matrix
    - mi_matrix: Mutual Information matrix
    """

    # Compute Pearson Correlation Matrix
    corr_matrix = X.corr()

    # Compute Mutual Information Matrix
    mi_matrix = pd.DataFrame(index=X.columns, columns=X.columns)
    for feature in X.columns:
        mi_matrix[feature] = mutual_info_regression(X, X[feature], discrete_features='auto')

    # Save matrices as CSV files (optional)
    if save_as_csv:
        corr_matrix.to_csv("correlation_matrix.csv")
        mi_matrix.to_csv("mutual_information_matrix.csv")
        print("✅ Correlation & MI matrices saved as CSV files.")

    return corr_matrix, mi_matrix

# Example Usage:
#analyze_feature_dependencies(X, y)

# Load the saved correlation and MI matrices
#corr_matrix = pd.read_csv("correlation_matrix.csv", index_col=0)
#mi_matrix = pd.read_csv("mutual_information_matrix.csv", index_col=0)


def find_highly_correlated_features(corr_matrix, threshold=0.8):
    """
    Identifies highly correlated features (> threshold) and suggests one to drop.

    Parameters:
    - corr_matrix: DataFrame containing feature correlation values.
    - threshold: Correlation threshold for considering features redundant.

    Returns:
    - DataFrame listing feature pairs, their correlation, and the feature suggested for removal.
    """
    to_drop = []  # Store features to drop
    checked_pairs = set()  # Store checked feature pairs
    correlation_list = []  # Store correlation details

    for feature in corr_matrix.columns:
        for correlated_feature in corr_matrix.index:
            if feature != correlated_feature and (feature, correlated_feature) not in checked_pairs:
                corr_value = corr_matrix.loc[feature, correlated_feature]

                if corr_value > threshold:
                    checked_pairs.add((feature, correlated_feature))

                    # Decide which feature to drop (keep feature with higher MI if available)
                    if correlated_feature in mi_matrix and feature in mi_matrix:
                        if mi_matrix.loc[feature, feature] > mi_matrix.loc[correlated_feature, correlated_feature]:
                            drop_feature = correlated_feature
                        else:
                            drop_feature = feature
                    else:
                        drop_feature = correlated_feature  # Default: drop the second feature

                    # Store results
                    correlation_list.append([feature, correlated_feature, corr_value, drop_feature])
                    to_drop.append(drop_feature)

    # Convert to DataFrame
    drop_df = pd.DataFrame(correlation_list, columns=["Feature 1", "Feature 2", "Correlation", "Suggested Drop"])

    return drop_df

# Find features with high correlation
#drop_df = find_highly_correlated_features(corr_matrix)
#drop_df.to_csv("features_to_drop1.csv", index=False)

# Display result
#print(f" Features to drop saved in 'features_to_drop.csv'.")
#print(drop_df.head())  # Show the first few rows of correlated features
#X["TotalLivableSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] # fair correlation, higher than the two component's but not enough to make up for the loss of flexibility
#X["GarageScore"] = (X["GarageQual"] + X["GarageCond"]) / 2 # low correlation





