
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
    faulted_df = df.copy()
    labels = np.zeros(len(df))
    
    # 1. Engine Overheat
    start = int(len(df) * 0.25)
    end = int(len(df) * 0.30)
    faulted_df.iloc[start:end, df.columns.get_loc('EGT')] += 100 
    labels[start:end] = 1
    
    # 2. Structural Micro-Fracture
    start = int(len(df) * 0.50)
    end = int(len(df) * 0.55)
    noise = np.random.normal(0, 0.5, size=(end-start))
    faulted_df.iloc[start:end, df.columns.get_loc('VRTG')] += noise
    labels[start:end] = 1
    
    # 3. Fuel Pump Failure
    start = int(len(df) * 0.75)
    end = int(len(df) * 0.80)
    faulted_df.iloc[start:end, df.columns.get_loc('FF')] *= 0.5
    faulted_df.iloc[start:end, df.columns.get_loc('N1')] *= 0.8
    labels[start:end] = 1
    
    return faulted_df, labels

def run_test():
    try:
        model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
        scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        files = glob.glob("csv_output/*.csv")
        f = files[0]
        df = pd.read_csv(f)
        rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        df_clean = df[[c for c in SENSOR_FEATURES if c in df.columns]]
        df_clean = df_clean.interpolate().fillna(method='bfill').fillna(method='ffill')
        
        # Inject
        df_faulted, labels = inject_faults(df_clean)
        
        # Predict
        X_faulted = df_faulted[SENSOR_FEATURES].values
        X_faulted_scaled = scaler.transform(X_faulted)
        preds_faulted = model.predict(X_faulted_scaled) # -1 Anomaly, 1 Normal
        
        # Manually calc metrics
        # TP: Label=1, Pred=-1
        # FN: Label=1, Pred=1
        tp = np.sum((labels == 1) & (preds_faulted == -1))
        fn = np.sum((labels == 1) & (preds_faulted == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        with open("manual_results.txt", "w") as out:
            out.write(f"Injected: {int(np.sum(labels))}\n")
            out.write(f"Detected (TP): {int(tp)}\n")
            out.write(f"Missed (FN): {int(fn)}\n")
            out.write(f"Simulated Recall: {recall:.4f}\n")
            out.write("Explanation: Direct manual calculation.\n")
            
    except Exception as e:
        with open("manual_results.txt", "w") as out:
            out.write(f"Error: {e}")

if __name__ == "__main__":
    run_test()
