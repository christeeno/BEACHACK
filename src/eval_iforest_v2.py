
import pandas as pd
import joblib
import os
import numpy as np
from src.real_data import load_stratified_data
from src.config import ARTIFACT_DIR

def evaluate_iforest_separation():
    model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
    
    if not os.path.exists(model_path):
        return

    iforest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load stratified sample for evaluation
    df = load_stratified_data("csv_output", num_std_files=2, num_long_files=2)
    features = [c for c in df.columns if c != '_source_file']
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    scores = iforest.decision_function(X_scaled)
    preds = iforest.predict(X_scaled)
    
    normals = scores[preds == 1]
    anomalies = scores[preds == -1]
    
    output = []
    output.append("\n--- Model Separation Metrics (Context-Aware) ---")
    output.append(f"Total Points: {len(scores)}")
    output.append(f"Normal Count: {len(normals)}")
    output.append(f"Anomaly Count: {len(anomalies)} ({len(anomalies)/len(scores):.2%})")
    output.append("-" * 30)
    output.append(f"Normal Score Mean:   {normals.mean():.4f} (std: {normals.std():.4f})")
    try:
        output.append(f"Anomaly Score Mean:  {anomalies.mean():.4f} (std: {anomalies.std():.4f})")
        gap = normals.mean() - anomalies.mean()
        output.append("-" * 30)
        output.append(f"Separation Gap: {gap:.4f}")
    except:
        output.append("No anomalies detected in this sample.")
    
    with open("eval_results.txt", "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    evaluate_iforest_separation()
