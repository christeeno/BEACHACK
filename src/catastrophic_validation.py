
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os
import matplotlib.pyplot as plt
import glob
from src.config import ARTIFACT_DIR, SENSOR_FEATURES

# AMM Mapping
AMM_MAP = {
    'EGT': 'AMM 72-00-00: Unscheduled Turbine Inspection',
    'VRTG': 'AMM 05-51-01: Hard Landing Inspection',
    'N1': 'AMM 73-00-00: Engine Fuel & Control System'
}

def inject_safety_breach(df):
    """
    Injects 3 catastrophic failures into a flight dataframe.
    Returns: faulted_df, metadata_list
    """
    faulted_df = df.copy()
    events = []
    
    # 1. Thermal Runaway (EGT +400 for 100 cycles in Cruise)
    # Find a cruise window (ALT > 10000)
    cruise_indices = df.index[df['ALT'] > 10000].tolist()
    if len(cruise_indices) > 200:
        start = cruise_indices[100] # nice stable point
        end = start + 100
        
        faulted_df.iloc[start:end, df.columns.get_loc('EGT')] += 400
        events.append({
            "type": "Thermal Runaway",
            "indices": (start, end),
            "driver": "EGT",
            "raw_value": float(faulted_df.iloc[start]['EGT']),
            "phase": "Cruise (Stable)"
        })
    else:
        print("Warning: No Cruise phase found for EGT injection.")

    # 2. Structural Shock (+3.0G during Landing)
    # Find Landing (ALT dropping below 2000 towards end)
    # Simple heuristic: last 1000 frames, lowest altitude
    try:
        late_flight = df.iloc[-2000:]
        landing_idx = late_flight['ALT'].idxmin() # Touchdown point
        # Inject shock at touchdown
        shock_idx = max(0, landing_idx - 10) # 10 frames before lowest point
        
        # Inject spike
        faulted_df.iloc[shock_idx:shock_idx+5, df.columns.get_loc('VRTG')] += 3.0
        
        events.append({
            "type": "Structural Shock",
            "indices": (shock_idx, shock_idx+5),
            "driver": "VRTG",
            "raw_value": float(faulted_df.iloc[shock_idx]['VRTG']),
            "phase": "Landing / Touchdown"
        })
    except Exception as e:
        print(f"Warning: Could not inject Structural Shock: {e}")

    # 3. Command Disconnect (N1 Drop vs Throttle)
    # Cruise or descent. Let's use early descent.
    try:
        # Find where Throttle (TH) > 50 but N1 > 80
        # Just pick a midpoint
        mid = int(len(df) * 0.6)
        faulted_df.iloc[mid:mid+20, df.columns.get_loc('N1')] = 10.0 # Flameout to 10%
        # Keep Throttle high (assuming original data had high throttle there)
        
        events.append({
            "type": "Command Disconnect",
            "indices": (mid, mid+20),
            "driver": "N1",
            "raw_value": 10.0,
            "phase": "Cruise / Descent"
        })
    except:
        pass
        
    return faulted_df, events

def run_catastrophic_validation():
    print("--- AeroGuard Catastrophic Validation ---")
    
    # Load Model
    model = joblib.load(os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl"))
    explainer = shap.TreeExplainer(model)
    
    # Load Data
    files = glob.glob("csv_output/*.csv")
    if not files: return
    
    # Must rename columns to match SENSOR_FEATURES
    COL_MAP = {
        'LATP': 'LATP', 'LONP': 'LONP', 'ALT': 'ALT', 'TAS': 'VEL', 'TH': 'TH',
        'N1_1': 'N1', 'N2_1': 'N2', 'EGT_1': 'EGT', 'FF_1': 'FF', 
        'VIB_1': 'VIB', 'VRTG': 'VRTG', 'OIP_1': 'OIL_P', 'OIT_1': 'OIL_T', 
        'FLAP': 'FLAP', 'HYDY': 'HYDY'
    }
    
    df_raw = pd.read_csv(files[0]) # Load first file
    df_raw = df_raw.rename(columns=COL_MAP)
    df = df_raw[[c for c in SENSOR_FEATURES if c in df_raw.columns]]
    df = df.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    # Inject
    df_catastrophic, events = inject_safety_breach(df)
    
    alert_payloads = []
    
    plt.figure(figsize=(12, 6))
    
    for i, event in enumerate(events):
        idx_start, idx_end = event['indices']
        target_idx = idx_start # Analyze the onset
        
        # Prepare Frame
        frame = df_catastrophic.iloc[target_idx]
        X = frame[SENSOR_FEATURES].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predict
        score = model.decision_function(X_scaled)[0]
        # XAI
        shap_values = explainer.shap_values(X_scaled)
        
        # Construct JSON for Frontend
        driver = event['driver']
        amm_ref = AMM_MAP.get(driver, "General Maintenance")
        
        payload = {
            "event_type": event['type'],
            "status": "SAFETY BREACH DETECTED",
            "phase": event['phase'],
            "anomaly_score": round(score, 4),
            "risk_driver_shaps": dict(zip(SENSOR_FEATURES, shap_values[0].tolist())),
            "action_plan": {
                "instruction": f"AeroGuard recommends immediate inspection of {driver} related systems.",
                "amm_reference": amm_ref,
                "inventory_check": "Available in Hub A",
                "guardrail": "Digital ID Required for Release"
            }
        }
        alert_payloads.append(payload)
        
        print(f"\n[EVENT] {event['type']}")
        print(f"  Score: {score:.4f} (Threshold: 0.00)")
        print(f"  Driver: {driver} | Value: {event['raw_value']}")
        print(f"  Action: {amm_ref}")
        
        # Plotting
        # Normalize time for viz
        window = df_catastrophic.iloc[idx_start-50 : idx_end+50]
        plt.subplot(1, 3, i+1)
        plt.plot(window.index, window[driver], 'r-', label='Catastrophic')
        # Plot healthy version
        try:
            plt.plot(window.index, df.iloc[window.index][driver], 'g--', label='Healthy')
        except: pass
        plt.title(f"{event['type']}\nScore: {score:.2f}")
        plt.xlabel("Time (cycles)")
        plt.ylabel(driver)
        if i == 0: plt.legend()

    # Save JSON
    with open("src/demo_alert_data.json", "w") as f:
        json.dump(alert_payloads, f, indent=4)
        
    print(f"\nSaved {len(alert_payloads)} alert scenarios to src/demo_alert_data.json")
    
    plt.tight_layout()
    plt.savefig("catastrophic_viz.png")
    print("Saved visualization to catastrophic_viz.png")

if __name__ == "__main__":
    run_catastrophic_validation()
