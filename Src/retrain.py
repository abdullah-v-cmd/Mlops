import os
import subprocess

def retrain_pipeline():
    print("Running full retraining pipeline...")
    subprocess.run(["python", "Src/data_ingestion.py"])
    subprocess.run(["python", "Src/preprocessing.py"])
    subprocess.run(["python", "Src/train.py"])
    subprocess.run(["python", "Src/evaluate.py"])
    print("Retraining complete.")

if __name__ == "__main__":
    retrain_pipeline()
