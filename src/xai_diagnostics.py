
import shap
import joblib
import pandas as pd
import numpy as np
import os
import warnings
from src.config import ARTIFACT_DIR, SENSOR_FEATURES
from src.real_data import load_real_sample

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AeroGuardDiagnostician:
    def __init__(self):
        self.model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
        self.scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run training first.")
            
        print(f"Loading Intelligence Core from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Initialize SHAP Explainer
        # TreeExplainer is optimized for Tree-based models like Isolation Forest
        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
    def map_sensor_to_part(self, sensor_name):
        """
        Maps a sensor feature name to a physical aircraft component 
        and AMM (Aircraft Maintenance Manual) reference.
        """
        mapping = {
            'EGT': ('Engine Core / Combustion', 'AMM Chapter 72-00: Turbine Inspection'),
            'FF':  ('Fuel System / Flow', 'AMM Chapter 73-00: Engine Fuel & Control'),
            'N1':  ('Fan / Low Pressure Compressor', 'AMM Chapter 72-30: Compressor Section'),
            'N2':  ('High Pressure Compressor', 'AMM Chapter 72-30: Compressor Section'),
            'VRTG': ('Airframe / Structural', 'AMM Chapter 32-00: Landing Gear Stress Test'),
            'HYDY': ('Hydraulic System', 'AMM Chapter 29-10: Main Hydraulic Power'),
            'OIL_P': ('Oil System / Pressure', 'AMM Chapter 79-00: Engine Oil'),
            'OIL_T': ('Oil System / Temperature', 'AMM Chapter 79-00: Engine Oil'),
            'VIB':   ('Engine Mounts / Vibration', 'AMM Chapter 72-00: Engine Vibration Mon'),
            'FLAP':  ('Flight Controls / Flaps', 'AMM Chapter 27-50: Trailing Edge Flaps'),
            'ALT':   ('Avionics / Air Data', 'AMM Chapter 34-10: Air Data System'),
            'VEL':   ('Avionics / Air Data', 'AMM Chapter 34-10: Air Data System'),
            'LATP':  ('Navigation / GPS', 'AMM Chapter 34-30: Landing & Taxi Aids'),
            'LONP':  ('Navigation / GPS', 'AMM Chapter 34-30: Landing & Taxi Aids'),
            'TH':    ('Throttle / Auto-Throttle', 'AMM Chapter 76-10: Power Control'),
        }
        
        return mapping.get(sensor_name, ('General Systems', 'AMM Chapter 05: General Maintenance'))

    def diagnose(self, frame_data):
        """
        Diagnoses a single frame of sensor data (dictionary or Series).
        Returns detailed diagnostics if anomalous.
        """
        # Ensure input matches feature order
        if isinstance(frame_data, dict):
            frame_df = pd.DataFrame([frame_data])
        elif isinstance(frame_data, pd.Series):
            frame_df = pd.DataFrame([frame_data])
        else:
            frame_df = frame_data
            
        # Select and align features
        try:
            X_raw = frame_df[SENSOR_FEATURES]
        except KeyError as e:
            missing = [f for f in SENSOR_FEATURES if f not in frame_df.columns]
            return {"error": f"Missing features: {missing}"}

        # Scale
        X_scaled = self.scaler.transform(X_raw)
        
        # Predict Anomaly
        # iForest: -1 is anomaly, 1 is normal
        prediction = self.model.predict(X_scaled)[0]
        score = self.model.decision_function(X_scaled)[0]
        
        if prediction == 1:
            return {
                "status": "NORMAL",
                "anomaly_score": round(score, 4),
                "confidence_gap": "N/A" 
            }
            
        # --- ANOMALY DETECTED: RUN XAI ---
        
        # 1. Calculate SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # 2. Extract contributions
        # shap_values is a list for some versions? no, usually array for regression/iforest
        # For iForest in sklearn, shap returns matrix matching input X
        contributions = dict(zip(SENSOR_FEATURES, shap_values[0]))
        
        # 3. Identify Leading Offender
        # In iForest, lower (more negative) values usually push towards anomaly? 
        # Actually, for Isolation Forest, the "explanation" interpretation can be tricky.
        # Generally, features that lead to 'short paths' (isolation) have high impact.
        # SHAP attribution assigns magnitude. We look for high absolute magnitude.
        
        sorted_sensors = sorted(contributions, key=lambda k: abs(contributions[k]), reverse=True)
        top_driver = sorted_sensors[0]
        
        total_impact = sum(abs(v) for v in contributions.values())
        impact_pct = (abs(contributions[top_driver]) / total_impact) * 100 if total_impact > 0 else 0
        
        # 4. Map to Component
        part_name, amm_ref = self.map_sensor_to_part(top_driver)
        
        # 5. Construct Diagnostic Payload
        diagnosis_string = (
            f"High Anomaly Score detected ({score:.4f}); "
            f"{impact_pct:.1f}% of deviation driven by {top_driver}. "
            f"Localized to {part_name}."
        )
        
        return {
            "status": "ANOMALY DETECTED",
            "anomaly_score": round(score, 4),
            "diagnostic_message": diagnosis_string,
            "localized_part": part_name,
            "primary_sensor": top_driver,
            "contribution_pct": round(impact_pct, 2),
            "manual_reference": amm_ref,
            "shap_features": contributions,  # For waterfall chart
            "secondary_drivers": sorted_sensors[1:3]
        }

def run_livedemo():
    print("--- AeroGuard XAI Diagnostic Demo ---")
    diagnostician = AeroGuardDiagnostician()
    
    # Load some real data to find a candidate anomaly
    print("Scanning flight logs for anomalies to diagnose...")
    df = load_real_sample("csv_output", num_files=2)
    
    # Find an anomaly
    features = [c for c in df.columns if c != '_source_file']
    X = df[features].values
    X_scaled = diagnostician.scaler.transform(X)
    preds = diagnostician.model.predict(X_scaled)
    
    anomaly_indices = np.where(preds == -1)[0]
    
    if len(anomaly_indices) == 0:
        print("No anomalies found in sample. Adjusting sensitivity or loading more files...")
        return

    # Pick the most severe anomaly (lowest score)
    scores = diagnostician.model.decision_function(X_scaled)
    worst_idx = np.argmin(scores)
    
    anomalous_frame = df.iloc[worst_idx]
    
    print(f"\nAnalyzing Frame #{worst_idx} from flight {anomalous_frame.get('_source_file', 'Unknown')}")
    print(f"Raw Score: {scores[worst_idx]:.4f}")
    
    # Run Diagnosis
    result = diagnostician.diagnose(anomalous_frame)
    
    # Pretty Print Result
    print("\n" + "="*60)
    print(f"AEROGUARD DIAGNOSTIC REPORT")
    print("="*60)
    print(f"STATUS:      {result['status']}")
    print(f"COMPONENT:   {result.get('localized_part', 'N/A')}")
    print(f"RISK DRIVER: {result.get('primary_sensor', 'N/A')} ({result.get('contribution_pct', 0)}% Impact)")
    print(f"ACTION REF:  {result.get('manual_reference', 'N/A')}")
    print("-" * 60)
    print(f"MSG: {result.get('diagnostic_message', '')}")
    print("="*60)

if __name__ == "__main__":
    run_livedemo()
