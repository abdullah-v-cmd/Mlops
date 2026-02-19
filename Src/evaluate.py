import pandas as pd
import os
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

DATA_DIR = "data/final"
MODEL_PATH = "models/model.pkl"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def evaluate_model():

    # load data
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    # load model
    model = joblib.load(MODEL_PATH)

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

import json

metrics = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_model()
