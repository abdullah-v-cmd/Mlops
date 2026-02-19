import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# paths
DATA_DIR = "data/final"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    # load data
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv")

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Drop customerID
    for df in [X_train, X_test]:
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)

    # set MLflow experiment
    mlflow.set_experiment("Telco-Churn")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # log model
        mlflow.sklearn.log_model(model, "model")

        print("✅ Training completed")
        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")

        # save local copy
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        import joblib
        joblib.dump(model, model_path)
        print("✅ Model saved locally")


if __name__ == "__main__":
    train_model()
