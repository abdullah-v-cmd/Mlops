import pandas as pd
import os
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

DATA_DIR = "data/final"
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/features.pkl"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def evaluate_model():
    # load data
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    # drop customerID if exists
    if "customerID" in X_test.columns:
        X_test.drop(columns=["customerID"], inplace=True)

    # load training feature names
    feature_names = joblib.load(FEATURES_PATH)

    # reorder X_test columns to match training
    X_test = X_test[feature_names]

    # load model
    model = joblib.load(MODEL_PATH)

    # predictions
    preds = model.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"{ARTIFACT_DIR}/confusion_matrix.png")
    plt.close()

    # classification report
    report = classification_report(y_test, preds)
    with open(f"{ARTIFACT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # log artifacts to MLflow
    mlflow.set_experiment("Telco-Churn")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_artifact(f"{ARTIFACT_DIR}/confusion_matrix.png")
        mlflow.log_artifact(f"{ARTIFACT_DIR}/classification_report.txt")

    print("✅ Evaluation completed")
    print(report)

if __name__ == "__main__":
    evaluate_model()
