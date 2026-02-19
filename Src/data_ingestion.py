import pandas as pd
import os

RAW_DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "telco.csv")


def load_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Data ingestion successful")
    print(df.head())

    return df


if __name__ == "__main__":
    load_data()
