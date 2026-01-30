
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os
import matplotlib.pyplot as plt
import glob
from src.config import ARTIFACT_DIR, SENSOR_FEATURES
from src.causal_diagnosis import CausalDiagnosisEngine

def inject_catastrophic_spikes(df):
    """
    Injects the specific 'Catastrophic' faults defined in prompt.
    Returns: df_faulted, fault_mask (1=fault, 0=normal)
    """
    df_f = df.copy()
    mask = np.zeros(len(df))
    
    # 1. Thermal Runaway (+400 EGT) - Mid Flight
    idx_mid = int(len(df) * 0.5)
    df_f.iloc[idx_mid:idx_mid+50, df.columns.get_loc('EGT')] += 400
    mask[idx_mid:idx_mid+50] = 1
    
    # 2. Structural Shock (+3.0 VRTG) - Late Flight
    idx_end = int(len(df) * 0.9)
    df_f.iloc[idx_end:idx_end+10, df.columns.get_loc('VRTG')] += 3.0
    mask[idx_end:idx_end+10] = 1
    
    return df_f, mask

def run_integration_test():
    print("--- AeroGuard Final System Integration Test ---")
    
    # 1. Setup & Loading
    model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
    scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    explainer = shap.TreeExplainer(model)
    diagnosis_engine = CausalDiagnosisEngine()
    
    # Load One Raw File
    files = glob.glob("csv_output/*.csv")
    if not files: return
    
    COL_MAP = {
        'LATP': 'LATP', 'LONP': 'LONP', 'ALT': 'ALT', 'TAS': 'VEL', 'TH': 'TH',
        'N1_1': 'N1', 'N2_1': 'N2', 'EGT_1': 'EGT', 'FF_1': 'FF', 
        'VIB_1': 'VIB', 'VRTG': 'VRTG', 'OIP_1': 'OIL_P', 'OIT_1': 'OIL_T', 
        'FLAP': 'FLAP', 'HYDY': 'HYDY'
    }
    
    df_raw = pd.read_csv(files[0]) # Start with first available
    # Map and Clean
    df_raw = df_raw.rename(columns=COL_MAP)
    df_clean = df_raw[[c for c in SENSOR_FEATURES if c in df_raw.columns]]
    df_clean = df_clean.interpolate().fillna(method='bfill').fillna(method='ffill')
    df_clean = df_clean.head(2000) # Limit to 2000 frames for speed/viz clarity
    
    print(f"Loaded Base Flight: {len(df_clean)} frames.")
    
    # 2. Prepare Profiles
    # Profile A: Casual (Clean)
    # Profile B: Catastrophic (Injected)
    df_casual = df_clean.copy()
    df_catastrophic, fault_mask = inject_catastrophic_spikes(df_clean)
    
    # 3. Model Inference & Metrics
    print("Processing Profile A (Casual Flight)...")
    X_casual = scaler.transform(df_casual[SENSOR_FEATURES].values)
    scores_casual = model.decision_function(X_casual)
    preds_casual = model.predict(X_casual) # -1 Anomaly
    
    print("Processing Profile B (Catastrophic Flight)...")
    X_cat = scaler.transform(df_catastrophic[SENSOR_FEATURES].values)
    scores_cat = model.decision_function(X_cat)
    preds_cat = model.predict(X_cat)
    
    # 4. JSON Report Generation
    # We create one report per flight, summarizing worst event
    
    def generate_flight_report(df_source, scores, X_scaled, filename):
        min_score_idx = np.argmin(scores)
        min_score = scores[min_score_idx]
        
        status = "Healthy" if min_score > 0 else "SAFETY BREACH DETECTED"
        
        # Diagnosis if Anomaly
        diagnosis_data = {}
        if status != "Healthy":
            # Run SHAP
            shap_values = explainer.shap_values(X_scaled[min_score_idx].reshape(1,-1))
            shap_dict = dict(zip(SENSOR_FEATURES, shap_values[0]))
            
            # Run Causal Engine
            frame = df_source.iloc[min_score_idx]
            diag = diagnosis_engine.diagnose_anomaly(shap_dict, frame)
            diagnosis_data = diag
        else:
            diagnosis_data = {"note": "No significant anomalies found. Operation Nominal."}
            
        report = {
            "flight_profile": filename,
            "overall_status": status,
            "min_anomaly_score": round(min_score, 4),
            "diagnosis_details": diagnosis_data
        }
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Generated {filename}")
        return min_score

    min_score_casual = generate_flight_report(df_casual, scores_casual, X_casual, "casual_flight_output.json")
    min_score_cat = generate_flight_report(df_catastrophic, scores_cat, X_cat, "catastrophic_flight_output.json")

    # 5. Trust Metrics
    # Specificity: Casual Flight correct classification rate (Normal=1)
    # iForest predicts 1 for Normal.
    specificity = np.mean(preds_casual == 1)
    
    # Sensitivity (Recall): Catastrophic Slices correctly identified
    # Only check the injected frames defined by fault_mask
    fault_indices = np.where(fault_mask == 1)[0]
    fault_preds = preds_cat[fault_indices] # Should be -1
    sensitivity = np.mean(fault_preds == -1) if len(fault_indices) > 0 else 0
    
    # Separation Gap
    # Difference between Mean Normal Score and Mean Anomaly Score (in catastrophic flight)
    normal_indices_cat = np.where(fault_mask == 0)[0]
    mean_normal = np.mean(scores_casual) # Use clean flight as baseline
    mean_fault = np.mean(scores_cat[fault_indices]) if len(fault_indices) > 0 else 0
    separation_gap = mean_normal - mean_fault
    
    print("\n--- Trust & Reliability Metrics ---")
    print(f"Specificity (Casual Flight Cleanliness): {specificity:.2%}")
    print(f"Sensitivity (Catastrophic Breach Catch Rate): {sensitivity:.2%}")
    print(f"Separation Gap (Statistical Distance): {separation_gap:.4f}")
    
    if sensitivity > 0.99 and separation_gap > 0.2:
        print("RESULT: PASS (Regulator-Safe Trust Level Achieved)")
    else:
        print("RESULT: WARNING (Metrics Validation Needed)")

    # 6. Final Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(scores_casual, 'g-', label='Casual Flight (Normal)', alpha=0.7)
    plt.plot(scores_cat, 'r--', label='Catastrophic Flight (Injected)', alpha=0.8)
    
    # Highlight Safety Threshold
    plt.axhline(y=0.0, color='k', linestyle=':', label='Safety Threshold (0.0)')
    
    plt.title("AeroGuard Final Integration: Normal vs Catastrophic Profile")
    plt.xlabel("Flight Frames (Cycles)")
    plt.ylabel("Anomaly Score (Positive=Safe, Negative=Risk)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("final_demo_results.png")
    print("Visualization saved to final_demo_results.png")

if __name__ == "__main__":
    run_integration_test()
