# stage 1, loading libraries and data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import json
import os
from pathlib import Path


pd.set_option('display.max_columns', None) # allows to see all of the columns of a pandas output works similarly for rows

df = pd.read_csv("C:/Users/itayp/OneDrive/Documents/Kaggle/hosuing price competition/housing prices train DF.csv")

#creating the categorial and numerical columns lists
cats = df.select_dtypes(include=['object']).columns
nums = df.select_dtypes(include = ['number']).columns

# diving into the categorial fields
#print(df[cats].isnull().sum()[df.isnull().sum() > 0])
#print(df[cats].describe())
#path to description file below
description_file_path = "C:/Users/itayp/OneDrive/Documents/Kaggle/hosuing price competition/data_description.txt"




#the funciton assumes that the entire description file is in the same format
def load_full_descriptions(file_path):
    """
    Reads the data description file and maps column names to their descriptions.
    Also extracts category meanings for categorical variables.
    """
    descriptions = {}
    category_mappings = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    current_col = None
    for line in lines:
        line = line.strip()

        # Detect column names (assumes format: ColumnName: Description)
        if line and ":" in line:
            col_name, desc = line.split(":", 1)
            current_col = col_name.strip()
            descriptions[current_col] = desc.strip()
            category_mappings[current_col] = {}

        # Detect category values (assumes format: Value Meaning)
        elif current_col and re.match(r"^\s*\S+\s+\S+", line):  # Looks for "Key  Meaning"
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, meaning = parts
                category_mappings[current_col][key] = meaning.strip()

    return descriptions, category_mappings


# Load descriptions once
column_descriptions, category_mappings = load_full_descriptions(description_file_path)


def analyze_categorical(df, col):
    """
    Analyzes a categorical column with full data descriptions.
    - Shows missing values, unique categories, and value counts.
    - Fetches the column description and possible category meanings.
    """
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found in the dataset.")
        return

    dtype = df[col].dtype
    if dtype != 'object':
        print(f"⚠️ Column '{col}' is not categorical. Skipping...")
        return

    print(f"\n📊 **Analysis of Categorical Column: {col}**")
    print("-" * 60)

    # Display description if available
    description = column_descriptions.get(col, "No description available.")
    print(f"📝 **Description:** {description}")

    # Missing values
    missing = df[col].isnull().sum()
    missing_percent = (missing / len(df)) * 100
    print(f"❌ Missing Values: {missing} ({missing_percent:.2f}%)")

    # Unique categories
    unique = df[col].nunique()
    print(f"🔹 Unique Categories: {unique}")

    # Most common category
    most_common = df[col].mode()[0] if unique > 0 else None
    print(f"⭐ Most Common Category: {most_common} ({df[col].value_counts().iloc[0]} occurrences)")

    # Value counts (top 5)
    print("\n📌 **Top 5 Categories:**")
    print(df[col].value_counts().head(5))

    # Show category meanings if available
    if col in category_mappings and category_mappings[col]:
        print("\n📖 **Category Meanings:**")
        for key, meaning in category_mappings[col].items():
            print(f"  - {key}: {meaning}")

    # Show sample values
    print("\n🎯 Sample Values:")
    print(df[col].dropna().sample(min(5, len(df)), random_state=42).tolist())
def interactive_categorical_analysis(df, categorical_cols):
    """
    Interactively analyzes categorical features.
    - Allows moving forward/backward, skipping, or quitting.
    - Collects user insights into a dictionary.
    *** SAVE THE JSON OUTPUT MANUALLY OR IT WILL BE GONE ***
    """
    cat_analysis = {}  # Store insights
    index = 0
    num_features = len(categorical_cols)

    while True:
        col = categorical_cols[index]
        print("\n" + "=" * 80)
        print(f"📊 **Feature {index + 1}/{num_features}: {col}**")
        print("=" * 80)

        # Run the feature analysis
        analyze_categorical(df, col)

        # Display previous insight if exists
        if col in cat_analysis:
            print(f"\n📌 **Previous Insight:** {cat_analysis[col]}")

        # User input for insights
        new_insight = input("\n📝 Enter your insight for this feature (or press Enter to keep previous): ").strip()
        if new_insight:
            cat_analysis[col] = new_insight  # Store new insight

        # User navigation options
        print("\n🔹 **Options:**")
        print("   [Enter] → Next Feature")
        print("   [B] → Go Back")
        print("   [S] → Skip This Feature")
        print("   [Q] → Quit Analysis")

        user_input = input("\n👉 Enter your choice: ").strip().lower()

        if user_input == "q":
            print("\n🚀 Exiting interactive analysis. Great job!")
            break
        elif user_input == "b":
            index = max(0, index - 1)  # Move back, but don't go below 0
        elif user_input == "s":
            index = min(num_features - 1, index + 1)  # Skip feature
        else:
            index = min(num_features - 1, index + 1)  # Move to next feature

    print("\n✅ **Final Collected Insights:**")
    for key, value in cat_analysis.items():
        print(f"{key}: {value}")

    return cat_analysis  # Return the dictionary for later use
category_analysis = {
    "MSZoning": {"action": "one-hot", "certainty": "probably", "comment": "No nulls, most values are low and medium density"},
    "Street": {"action": "drop", "certainty": "sure", "comment": "Almost all values are 'Pave', likely drop"},
    "Alley": {"action": "drop", "certainty": "sure", "comment": "94% nulls, should drop"},
    "LotShape": {"action": "one-hot", "certainty": "sure", "comment": "No nulls, might combine IR2 and IR3"},
    "LandContour": {"action": "one-hot", "certainty": "maybe", "comment": "No nulls, over 80% in one category"},
    "Utilities": {"action": "drop", "certainty": "sure", "comment": "No nulls, 99% same category"},
    "LotConfig": {"action": "review", "certainty": "maybe", "comment": "Not sure, come back to this"},
    "LandSlope": {"action": "one-hot", "certainty": "maybe", "comment": "94% gentle slope, context is interesting"},
    "Neighborhood": {"action": "one-hot", "certainty": "sure", "comment": "No nulls, well distributed"},
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
    "BsmtFinType2": {"action": "review", "certainty": "maybe", "comment": "Seems confusing, come back to this"},
    "Heating": {"action": "drop", "certainty": "sure", "comment": "99% in a single category"},
    "HeatingQC": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "CentralAir": {"action": "binary", "certainty": "sure", "comment": "No nulls, reshape to 0/1"},
    "Electrical": {"action": "ordinal", "certainty": "sure", "comment": "1 null, logical order"},
    "KitchenQual": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "Functional": {"action": "ordinal", "certainty": "sure", "comment": "No nulls, logical order"},
    "FireplaceQu": {"action": "ordinal", "certainty": "sure", "comment": "50% nulls, replace with 0"},
    "GarageType": {"action": "binary", "certainty": "maybe", "comment": "81 nulls, importance unclear"},
    "GarageFinish": {"action": "ordinal", "certainty": "sure", "comment": "81 nulls, set nulls to -1"},
    "GarageQual": {"action": "ordinal", "certainty": "sure", "comment": "Set nulls to -1"},
    "GarageCond": {"action": "ordinal", "certainty": "sure", "comment": "Change nulls to 0"},
    "PavedDrive": {"action": "binary", "certainty": "probably", "comment": "Consider merging N and P, then binary"},
    "PoolQC": {"action": "ordinal", "certainty": "probably", "comment": "99% nulls, change to 0"},
    "Fence": {"action": "ordinal", "certainty": "sure", "comment": "80% nulls, set nulls to 0"},
    "MiscFeature": {"action": "drop", "certainty": "probably", "comment": "96% nulls, drop or OH"},
    "SaleType": {"action": "drop", "certainty": "sure", "comment": "Potential data leakage"},
    "SaleCondition": {"action": "drop", "certainty": "sure", "comment": "Potential data leakage"}
}


# saving as json

"""file_path = os.path.join(os.getcwd(), "category_analysis.json")
with open(file_path, "w") as f:
    json.dump(category_analysis, f, indent=4)

if os.path.exists(file_path):
    print(f"✅ File successfully saved at: {file_path}")
else:
    print("❌ File saving failed. Check your path!")

"""# numerical fields research

# 1️⃣ Load Feature Descriptions from File
def load_feature_descriptions(filepath):
    """
    Reads the data description file and returns a dictionary mapping features to descriptions.
    """
    descriptions = {}
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines:
            if ":" in line:  # Assumes format "FeatureName: Description"
                feature, desc = line.split(":", 1)
                descriptions[feature.strip()] = desc.strip()
    return descriptions


# Load descriptions before analysis
feature_descriptions = load_feature_descriptions("data_description.txt")  # Update path if needed


# 2️⃣ Retrieve Feature Descriptions
def get_feature_description(feature_name, description_dict):
    """
    Retrieves the description of a feature from the provided dictionary.
    """
    return description_dict.get(feature_name, "🔍 No description available for this feature.")


# 3️⃣ Load Previous Progress
def load_progress(filename="num_analysis.json"):
    """
    Loads previous numerical feature analysis progress if available.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


# Dictionary to store insights (Load existing progress if available)
num_analysis = load_progress()


# 4️⃣ Save Progress After Each Entry
def save_progress(filename="num_analysis.json"):
    """
    Saves progress to a JSON file to prevent data loss.
    """
    with open(filename, "w") as f:
        json.dump(num_analysis, f, indent=4)
    print("✅ Progress saved successfully!")


# 5️⃣ Analyze a Single Numerical Feature
def analyze_numerical_feature(df, col):
    """
    Analyzes a numerical column:
    - Shows feature description
    - Displays missing values
    - Detects outliers using IQR method
    - Checks skewness
    - Allows user to request a plot
    - Lets user input preprocessing decisions + notes
    - Saves progress after every feature
    """
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not found in the dataset.")
        return

    print("\n" + "=" * 80)
    print(f"📊 **Feature: {col}**")
    print("=" * 80)

    # Display feature description
    description = get_feature_description(col, feature_descriptions)
    print(f"📖 **Feature Description:** {description}\n")

    # Summary statistics
    print(df[col].describe())

    # Missing values
    missing = df[col].isnull().sum()
    print(f"\n❌ Missing Values: {missing} ({(missing / len(df)) * 100:.2f}%)")

    # Detect Outliers using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    print(f"⚠️ Outliers Detected: {outliers}")

    # Skewness
    skewness = df[col].skew()
    print(f"🔄 Skewness: {skewness:.2f} (|Skew| > 1 indicates strong skewness)")

    # Ask user if they want to see a plot
    plot_choice = input("\n📈 Would you like to see a histogram & boxplot? (y/n): ").strip().lower()

    if plot_choice == "y":
        # Plot histogram & boxplot
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        sns.histplot(df[col], bins=30, kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram of {col}")

        # Boxplot
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot of {col}")

        plt.show()

    # User input for actions
    print("\n🔹 **Actions to Consider:**")
    print("   [1] Impute missing values (mean/median/mode)")
    print("   [2] Fill missing values with 0")
    print("   [3] Drop this feature")
    print("   [4] Cap outliers (Winsorization)")
    print("   [5] Apply log transformation")
    print("   [6] Keep as is")
    print("   [S] Skip")

    action = input("\n📝 Enter your decision (or press Enter to skip): ").strip()

    # Allow user to add notes
    note = input("📝 Add additional notes (press Enter to skip): ").strip()

    if action or note:
        num_analysis[col] = {"action": action, "note": note}  # Store both action and notes
        save_progress()  # ✅ Save progress immediately


# 6️⃣ Interactive Analysis for All Numerical Features
def interactive_numerical_analysis(df, numerical_cols):
    """
    Interactively analyzes numerical features.
    - Allows moving forward/backward, skipping, or quitting.
    - Collects user insights into a dictionary.
    - Saves progress after each step.
    """
    index = 0
    num_features = len(numerical_cols)

    while index < num_features:
        col = numerical_cols[index]

        # If feature is already analyzed, show stored result
        if col in num_analysis:
            print(f"\n⏩ Skipping {col} (Already Analyzed)")
        else:
            analyze_numerical_feature(df, col)

        # User navigation options
        print("\n🔹 **Options:**")
        print("   [Enter] → Next Feature")
        print("   [B] → Go Back")
        print("   [S] → Skip This Feature")
        print("   [Q] → Quit Analysis (Progress is Saved)")

        user_input = input("\n👉 Enter your choice: ").strip().lower()

        if user_input == "q":
            print("\n🚀 Exiting interactive analysis. Progress is saved!")
            save_progress()
            break
        elif user_input == "b":
            index = max(0, index - 1)  # Move back, but don't go below 0
        elif user_input == "s":
            index = min(num_features - 1, index + 1)  # Skip feature
        else:
            index = min(num_features - 1, index + 1)  # Move to next feature

    print("\n✅ **Final Collected Insights:**")
    for key, value in num_analysis.items():
        print(f"{key}: {value}")

    return num_analysis  # ✅ Return the dictionary for later use


#num_analysis = interactive_numerical_analysis(df,nums)

##save the mapping
"""
# 🔹 MANUALLY SET THIS PATH: Your existing JSON file
your_existing_json_path = "C:/Users/itayp/PycharmProjects/housing prices Kaggle/num_analysis.json"

# 🔹 MANUALLY SET THIS PATH: Where you want to save the cleaned version
your_cleaned_json_path = "C:/Users/itayp/PycharmProjects/housing prices Kaggle/num_analysis_decoded.json"

# Define the mapping of numeric actions to meaningful labels
action_mapping = {
    "1": "impute_with_median",
    "2": "impute_with_0",
    "3": "drop_feature",
    "4": "cap_extreme_values",
    "5": "log_transform",
    "6": "keep_as_is",
    "s": "skip_for_now",
}

# Load the existing JSON file
with open(your_existing_json_path, "r") as f:
    num_analysis = json.load(f)

# Convert actions from numbers to meaningful labels
for feature, details in num_analysis.items():
    if details["action"] in action_mapping:
        num_analysis[feature]["action"] = action_mapping[details["action"]]

# Save the updated JSON file (overwriting the old one or saving a new version)
with open(your_cleaned_json_path, "w") as f:
    json.dump(num_analysis, f, indent=4)

print(f"✅ JSON file updated successfully! Saved at: {your_cleaned_json_path}")
"""


# categorical mutations
file_path = Path("C:/Users/itayp/PycharmProjects/housing prices Kaggle/category_analysis.json")
with file_path.open("r") as f:
    cat_map = json.load(f)

file_path1 = Path("C:/Users/itayp/PycharmProjects/housing prices Kaggle/num_analysis_decoded.json")
with file_path1.open("r") as f:
    num_map = json.load(f)
def extract_feature_lists(map):
    """
    Extracts lists of categorical features based on preprocessing action.
    Only includes features with "certainty": "sure".

    Parameters:
    - cat_map (dict): A dictionary with categorical feature preprocessing decisions.

    Returns:
    - drop_features (list): Features to drop.
    - one_hot_features (list): Features to one-hot encode.
    - ordinal_features (list): Features to ordinal encode.
    - binary_features (list): Features to binary encode.
    """
    drop_features = []
    one_hot_features = []
    ordinal_features = []
    binary_features = []

    for feature, details in cat_map.items():
        if details.get("certainty") == "sure":
            action = details.get("action")
            if action == "drop":
                drop_features.append(feature)
            elif action == "one-hot":
                one_hot_features.append(feature)
            elif action == "ordinal":
                ordinal_features.append(feature)
            elif action == "binary":
                binary_features.append(feature)

    return drop_features, one_hot_features, ordinal_features, binary_features


drop_features, one_hot_features, ordinal_features, binary_features = extract_feature_lists(cat_map)
#drop the marked as drop features
df = df.drop(columns = drop_features)


#checking what features need permutation to 0 or none
def null_status(df,cat_map,num_map):
    for col in df:
        x = None
        if df[col].isna().sum() > 0:
            x_cat = cat_map.get(col)
            x_num = num_map.get(col)
            x = x_cat if x_cat else x_num
        if x is not None:
            print(x,col,"nulls: ",df[col].isna().sum(),"type: ",df[col].dtype)


features_to_replace = [
    "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature","BsmtFinType2","GarageType"
]

#permutation of missing values to 0 or none
def replace_missing_with_zero_or_none(df, features):
    """
    Replaces missing values in given features:
    - Categorical (object dtype) → 'None'
    - Numerical → 0

    Parameters:
    - df (pd.DataFrame): The dataset
    - features (list): List of features to modify

    Returns:
    - df (pd.DataFrame): Updated DataFrame
    """
    for col in features:
        if col in df.columns:  # Ensure the column exists
            if df[col].dtype == object:
                df[col] = df[col].fillna("None")  # ✅ Now modifies df
            else:
                df[col] = df[col].fillna(0)  # ✅ Now modifies df

    return df

df = replace_missing_with_zero_or_none(df, features_to_replace)

#attedning specific difficult permutations
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df["Electrical"] = df["Electrical"].fillna("SBrkr")# a single null, filled with mode
df.loc[df["GarageType"] == "None", "GarageYrBlt"] = 0 # replacing missing entires of houses with no garage with 0 and house with to the year of renovation or building
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearRemodAdd"]).fillna(df["YearBuilt"])

#analyze_categorical(df, "GarageType")
#analyze_numerical_feature(df,"GarageYrBlt")
#null_status(df,cat_map,num_map)
print(df.isna().sum()[df.isna().sum() > 0])
from sklearn.feature_selection import mutual_info_regression
garage_dummies = pd.get_dummies(df["GarageType"], prefix="GarageType", drop_first=True)

# Merge into the main dataset
garage_corr = garage_dummies.corrwith(df["SalePrice"]).abs().sort_values(ascending=False)

#print(garage_corr)

mi_scores = mutual_info_regression(garage_dummies, df["SalePrice"], discrete_features=True)

# Convert to a readable format
print(df["MasVnrArea"].isna().sum())