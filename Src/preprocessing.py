import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# paths
PROCESSED_DATA_PATH = "data/processed/telco.csv"
OUTPUT_DIR = "data/final"


def preprocess_data():
    # create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # fix TotalCharges column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)

    # encode categorical columns
    for col in df.select_dtypes(include="object"):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # split features & target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # save datasets
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    print("✅ Preprocessing completed")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data()
