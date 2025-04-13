import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# Load dataset
df = pd.read_csv('final/code3data.csv')

### data cleaning and preprocessing ###
df = df.drop(columns=['Date', 'Patient No.', 'Child-Pugh score', 'MELD'])
# df = df.drop(columns=['Date', 'Patient No.'])

# drop columns with NaNs
df.dropna(axis=1, inplace=True)

# Replace string "." with NaN
df.replace(".", np.nan, inplace=True)
df.dropna(inplace=True)

## discretise continuous features
def discretize_and_onehot(df):
    features_to_bin = [
        "Age",
        "Blood transfused in 48 hours (u)",
        "Platelet count (x10^3/uL)",
        "WBC (x10^3/uL)",
        "Hemoglobin (g/L)",
        "INR",
        "Na (mEq/L)",
        "Creatinine (mg/L)",
        "Bilirubin (mg/dL)",
        "ALT (IU/L)",
        "Albumin (g/dL)",
        "Systolic blood pressure (mmHg)",
        "Heart rate (beats/min)",
        "Hospitalization (day)"
    ]

    binned_df = df.copy()
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile') # n_bins=2 as >2 will have too many parameters for riskslim to process

    for col in features_to_bin:
        if col in binned_df.columns:
            reshaped = binned_df[[col]].values
            binned = discretizer.fit_transform(reshaped).astype(int)
            binned_df[col + "_bin"] = binned

    # Drop original continuous columns
    binned_df.drop(columns=features_to_bin, inplace=True)

    # One-hot encode the binned features
    binned_df = pd.get_dummies(binned_df, columns=[f"{col}_bin" for col in features_to_bin], drop_first=False)

    return binned_df

df = discretize_and_onehot(df)

# Binary columns
binary_columns = ["Antibiotic prophylaxis", "HCC", "Ascites", "Hepatic encephalopathy", "Prior SBP", "ICU admission"]
for col in binary_columns:
    df[col] = df[col].str.lower().str.strip().map(lambda x: 1 if x == 'yes' else 0)

# Gender column
df["Sex"] = df["Sex"].str.lower().str.strip().map(lambda x: 1 if x == 'male' else 0)

# One-hot encode categorical columns
one_hot_columns = ["Etiology of cirrhosis", "Etiology of bleeding", "Treatment"]
df = pd.get_dummies(df, columns=one_hot_columns, drop_first=False)

# Convert boolean columns to int
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# clean column names
df.columns = df.columns.str.replace(r'[()\s^/]+', '_', regex=True)
print(df.columns)

# Common feature set: drop target columns
X = df.drop(columns=["Infection_within_14_days", "Releeding_within_14_days", "Mortality_within_42_days"])

# Target labels
targets = {
    "infection": df["Infection_within_14_days"],
    "rebleeding": df["Releeding_within_14_days"],
    "mortality": df["Mortality_within_42_days"]
}

# Loop through each target and generate cleaned train/test CSVs
for label, y in targets.items():
    print(f"Processing: {label}")

    ### Train/test split ###
    X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Convert target features to 1 / -1
    y_train = np.where(np.array(y_train_raw) == "yes", 1, -1).astype(float)
    y_test = np.where(np.array(y_test_raw) == "yes", 1, -1).astype(float)

    # Convert all features to float
    X_train = np.array([[float(x) for x in row] for row in X_train_df.values])
    X_test = np.array([[float(x) for x in row] for row in X_test_df.values])

    # Save as DataFrames
    train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
    train_df[label] = y_train
    test_df = pd.DataFrame(X_test, columns=X_test_df.columns)
    test_df[label] = y_test

    # Save to CSV
    train_df.to_csv(f"final/data_train_{label}.csv", index=False)
    test_df.to_csv(f"final/data_test_{label}.csv", index=False)

    # Print summary
    print(f"Saved data_train_{label}.csv and data_test_{label}.csv")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}\n")
