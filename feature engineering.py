import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

#### commit commit

df = pd.read_csv("C:/Users/itayp/PycharmProjects/housing prices Kaggle/housing prices train DF_clean.csv")

# Split features & target variable
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]


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
print(analyze_feature_importance(X,y))
# Example Usage:
"""
feature_importance = analyze_feature_importance(X, y)
pd.set_option("display.max_rows", None)  # Show all rows
print(feature_importance)
pd.reset_option("display.max_rows")  # Reset to default after printing
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

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
#corr_matrix, mi_matrix = analyze_feature_dependencies(X, y)


# Load the saved correlation and MI matrices
corr_matrix = pd.read_csv("correlation_matrix.csv", index_col=0)
mi_matrix = pd.read_csv("mutual_information_matrix.csv", index_col=0)


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

