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
working_df["total_house_area"] = working_df["GrLivArea"] + working_df["TotalBsmtSF"]
print(feature_assessment(["total_house_area", "GrLivArea", "TotalBsmtSF"],y))
# total house area has a much higher mi and corr
