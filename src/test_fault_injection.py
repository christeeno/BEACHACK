
import pandas as pd
import numpy as np
import joblib
import os
import glob
from src.config import ARTIFACT_DIR, SENSOR_FEATURES

# Reuse column mapping logic
COLUMN_MAP = {
    'LATP': 'LATP', 'LONP': 'LONP', 'ALT': 'ALT', 'TAS': 'VEL', 'TH': 'TH',
    'N1_1': 'N1', 'N2_1': 'N2', 'EGT_1': 'EGT', 'FF_1': 'FF', 
    'VIB_1': 'VIB', 'VRTG': 'VRTG', 'OIP_1': 'OIL_P', 'OIT_1': 'OIL_T', 
    'FLAP': 'FLAP', 'HYDY': 'HYDY'
}

def inject_faults(df):
    """
    Creates a copy of the dataframe with specific synthetic failures injected.
    Returns: faulted_df, labels (0=Normal, 1=Fault)
    """
    faulted_df = df.copy()
    labels = np.zeros(len(df))
    
    # 1. Engine Overheat (Gradual EGT Drift + Spike)
    # Inject in 2nd quarter of data
    start = int(len(df) * 0.25)
    end = int(len(df) * 0.30)
    # Spike EGT by 4 standard deviations (approx +50-100 deg)
    faulted_df.iloc[start:end, df.columns.get_loc('EGT')] += 100 
    labels[start:end] = 1
    
    # 2. Structural Micro-Fracture (VRTG Noise)
    # Inject in 3rd quarter
    start = int(len(df) * 0.50)
    end = int(len(df) * 0.55)
    noise = np.random.normal(0, 0.5, size=(end-start)) # High variance noise
    faulted_df.iloc[start:end, df.columns.get_loc('VRTG')] += noise
    labels[start:end] = 1
    
    # 3. Fuel Pump Failure (Drop in Flow + RPM drop)
    start = int(len(df) * 0.75)
    end = int(len(df) * 0.80)
    faulted_df.iloc[start:end, df.columns.get_loc('FF')] *= 0.5  # Drop flow by 50%
    faulted_df.iloc[start:end, df.columns.get_loc('N1')] *= 0.8  # Drop RPM
    labels[start:end] = 1
    
    return faulted_df, labels

def run_fault_test():
    print("--- AeroGuard Synthetic Fault Injection Test ---")
    
    model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Direct Load (Skip Scanner)
    print("Loading baseline data (Direct Mode)...")
    files = glob.glob("csv_output/*.csv")
    if not files: 
        print("No data found.")
        return
        
    f = files[0] # Just take the first one
    df = pd.read_csv(f)
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    df = df[[c for c in SENSOR_FEATURES if c in df.columns]]
    df_clean = df.interpolate().fillna(method='bfill').fillna(method='ffill')
    df_clean = df_clean.head(1000) # DEBUG: Limit size
    
    # Ensure it's mostly normal to start with
    X_clean = df_clean[SENSOR_FEATURES].values
    X_clean_scaled = scaler.transform(X_clean)
    baseline_preds = model.predict(X_clean_scaled)
    baseline_anomalies = np.sum(baseline_preds == -1)
    print(f"Baseline (Clean) Anomaly Rate: {baseline_anomalies/len(df_clean):.2%}")
    
    # Inject Faults
    print("Injecting synthetic failures (EGT Surge, VRTG Noise, Fuel Pump Drop)...")
    try:
        df_faulted, labels = inject_faults(df_clean)
        print("Faults injected successfully.")
    except Exception as e:
        print(f"Error during injection: {e}")
        return
    
    # Predict on Faulted Data
    print("Predicting on faulted data...")
    try:
        X_faulted = df_faulted[SENSOR_FEATURES].values
        X_faulted_scaled = scaler.transform(X_faulted)
        preds_faulted = model.predict(X_faulted_scaled) # -1 Anomaly, 1 Normal
        print("Prediction complete.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    
    # Evaluation Metrics
    # True Positive = IS Anomaly (-1) AND WAS Injected (1)
    # Convert iForest pred (-1/1) to (1/0)
    try:
        binary_preds = np.where(preds_faulted == -1, 1, 0)
        
        from sklearn.metrics import recall_score, precision_score, f1_score
        
        recall = recall_score(labels, binary_preds)
        precision = precision_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds)
        
        print("\n--- Test Results ---", flush=True)
        print(f"Total Frames Tested: {len(df_faulted)}", flush=True)
        print(f"Injected Fault Frames: {int(sum(labels))}", flush=True)
        print(f"Detected Fault Frames: {int(sum(binary_preds))}", flush=True)
        print("-" * 30, flush=True)
        print(f"Simulated Recall (Sensitivity): {recall:.4f}", flush=True)
        print(f"Precision: {precision:.4f}", flush=True)
        print(f"F1-Score:  {f1:.4f}", flush=True)
        print("-" * 30, flush=True)
        
        if recall > 0.85:
            print("PASS: Model successfully detects >85% of synthetic critical failures.")
        else:
            print("FAIL: Model missed significant failure signatures.")
            
    except Exception as e:
        print(f"Error computing metrics: {e}", flush=True)

if __name__ == "__main__":
    run_fault_test()
