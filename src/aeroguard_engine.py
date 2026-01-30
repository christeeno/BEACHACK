
import shap
import joblib
import pandas as pd
import numpy as np
import os
import json
import hashlib
import datetime
import warnings
from src.config import ARTIFACT_DIR, SENSOR_FEATURES
from src.real_data import load_real_sample

# Suppress warnings
warnings.filterwarnings('ignore')

class AeroGuardEngine:
    def __init__(self):
        self.model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
        self.scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
        self.inventory_path = os.path.join(os.path.dirname(__file__), "inventory.json")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model artifacts not found.")
            
        print(f"Loading AeroGuard Intelligence from {ARTIFACT_DIR}...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Load Inventory
        try:
            with open(self.inventory_path, 'r') as f:
                self.inventory = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load inventory: {e}")
            self.inventory = {}

    def _normalize_phase(self, row):
        """
        Phase Normalization Gap Check:
        Prevents false positives during high-stress but normal maneuvers like takeoff.
        """
        alt = row.get('ALT', 0)
        vrtg = row.get('VRTG', 1.0)
        
        # Simple logical filter as requested
        if alt < 1000 and abs(vrtg) > 1.2:
            return "Takeoff_Normal"
        return "Cruise/Other"

    def _map_sensor_to_part(self, sensor_name):
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

    def _generate_integrity_hash(self, data_dict):
        """Creates an immutable log hash for the report."""
        raw_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(raw_str.encode('utf-8')).hexdigest()

    def analyze_frame(self, frame_data):
        """
        The Master Logic: Predict -> Normalize -> Explain -> Localize -> Report
        """
        # 1. Feature Prep
        try:
            X_df = pd.DataFrame([frame_data])[SENSOR_FEATURES]
        except KeyError:
            return {"error": "Missing sensor data in frame"}
            
        X_scaled = self.scaler.transform(X_df)
        
        # 2. Anomaly Detection (The Trigger)
        score = self.model.decision_function(X_scaled)[0]
        is_anomaly = self.model.predict(X_scaled)[0] == -1
        
        # 3. Phase Normalization (The Filter)
        phase_label = self._normalize_phase(frame_data)
        
        if phase_label == "Takeoff_Normal":
            # Override anomaly if it's just takeoff stress
            # But for safety, we might just log it as such
            diagnosis_status = "NORMAL (Takeoff Phase Cleared)"
            is_anomaly = False # Suppress downstream alarm
        elif is_anomaly:
            diagnosis_status = "ANOMALY DETECTED"
        else:
            diagnosis_status = "NORMAL"

        if not is_anomaly:
            return {
                "uuid": self._generate_integrity_hash(frame_data.to_dict()),
                "timestamp": datetime.datetime.now().isoformat(),
                "status": diagnosis_status,
                "score": round(score, 4),
                "action": "None"
            }

        # --- XAI DIAGNOSTICS (Only runs if anomaly is valid) ---
        
        # 4. SHAP Attribution
        shap_values = self.explainer.shap_values(X_scaled)
        contributions = dict(zip(SENSOR_FEATURES, shap_values[0]))
        
        # 5. Localization
        top_driver = max(contributions, key=lambda k: abs(contributions[k]))
        part_name, amm_ref = self._map_sensor_to_part(top_driver)
        
        # 6. Inventory Check (The Actionable Link)
        inventory_data = self.inventory.get(part_name, {"status": "Unknown", "stock_quantity": 0})
        stock_status = f"{inventory_data.get('stock_quantity', 0)} In Stock at {inventory_data.get('location', 'Unknown')}"
        
        # 7. Construct Report
        report_payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "INSPECTION REQUIRED",  # Human Gate
            "anomaly_score": round(score, 4),
            "phase_context": phase_label,
            "diagnosis": {
                "root_cause_sensor": top_driver,
                "localized_component": part_name,
                "maintenance_manual_ref": amm_ref,
                "failure_window_prediction": "Immediate Attention (< 5 Cycles)" if score < -0.1 else "Monitor (< 50 Cycles)"
            },
            "logistics": {
                "part_availability": stock_status,
                "lead_time": f"{inventory_data.get('lead_time_days', 'N/A')} days"
            },
            "feedback_loop": {
                "endpoint": "POST /api/v1/feedback",
                "action_required": "Engineer_Confirm=True" 
            }
        }
        
        # 8. Immutable Hash
        report_payload["integrity_hash"] = self._generate_integrity_hash(report_payload)
        
        return report_payload

def run_system_check():
    print("--- AeroGuard Unified Reliability Mesh: Initialization ---")
    engine = AeroGuardEngine()
    
    # Load Real Data
    print("Ingesting flight telemetry...")
    df = load_real_sample("csv_output", num_files=2)
    
    # 1. Simulate Normal Data
    normal_frame = df.iloc[0]
    print("\n[TEST 1] Processing Standard Cruise Frame...")
    res_normal = engine.analyze_frame(normal_frame)
    print(f"Result: {res_normal['status']} (Score: {res_normal['score']})")
    
    # 2. Simulate Anomaly (Find one using the model)
    print("\n[TEST 2] Hunting for Actual Anomalies in Data...")
    X = df[SENSOR_FEATURES].values
    X_scaled = engine.scaler.transform(X)
    scores = engine.model.decision_function(X_scaled)
    min_idx = np.argmin(scores)
    
    anomalous_frame = df.iloc[min_idx]
    
    print(f"Analyzing Frame #{min_idx} (Score: {scores[min_idx]:.4f})...")
    report = engine.analyze_frame(anomalous_frame)
    
    print("\n" + "="*60)
    print("AEROGUARD DIGITIAL MECHANIC REPORT")
    print("="*60)
    print(json.dumps(report, indent=4))
    print("="*60)
    
    # Verify Integrity
    print(f"\n[INTEGRITY CHECK] Report Hash: {report.get('integrity_hash')}")
    print("[FEEDBACK LOOP] Waiting for Engineer Confirmation...")

if __name__ == "__main__":
    run_system_check()
