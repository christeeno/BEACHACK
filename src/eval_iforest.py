
import pandas as pd
import joblib
import os
import numpy as np
from src.real_data import load_real_sample
from src.config import ARTIFACT_DIR

def evaluate_iforest_separation():
    model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    print("Loading model and data...")
    iforest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load just 2 files for quick eval
    df = load_real_sample("csv_output", num_files=2)
    features = [c for c in df.columns if c != '_source_file']
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    print("Scoring data...")
    # decision_function: Average anomaly score of X of the base classifiers.
    # The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
    # The measure of normality of an observation given a tree is the depth of the leaf containing this observation...
    # In scikit-learn IsolationForest:
    # - scores < 0 are anomalies
    # - scores > 0 are normal
    scores = iforest.decision_function(X_scaled)
    preds = iforest.predict(X_scaled)
    
    normals = scores[preds == 1]
    anomalies = scores[preds == -1]
    
    print("\n--- Model Separation Metrics ---")
    print(f"Total Points: {len(scores)}")
    print(f"Normal Count: {len(normals)}")
    print(f"Anomaly Count: {len(anomalies)} ({len(anomalies)/len(scores):.2%})")
    print("-" * 30)
    print(f"Normal Score Mean:   {normals.mean():.4f} (std: {normals.std():.4f})")
    print(f"Anomaly Score Mean:  {anomalies.mean():.4f} (std: {anomalies.std():.4f})")
    print(f"Min Normal Score:    {normals.min():.4f}")
    print(f"Max Anomaly Score:   {anomalies.max():.4f}")
    
    # Separation Gap
    gap = normals.mean() - anomalies.mean()
    print("-" * 30)
    print(f"Statistical Separation (Effect Size): {gap:.4f}")
    print("Interpretation: A larger gap indicates the model finding very distinct anomalies.")

if __name__ == "__main__":
    evaluate_iforest_separation()
