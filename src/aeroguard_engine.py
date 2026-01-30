
import shap
import joblib
import pandas as pd
import numpy as np
import os
import json
import hashlib
import datetime
import warnings # Add missing import
from src.config import ARTIFACT_DIR, SENSOR_FEATURES, DQI_THRESHOLD, LEDGER_PATH, HARD_LIMIT_EGT
from src.real_data import load_stratified_data

# Suppress sklearn warnings if needed
import warnings
warnings.filterwarnings('ignore')

class AeroGuardEngine:
    def __init__(self):
        self.model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
        self.scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
        self.inventory_path = os.path.join(os.path.dirname(__file__), "inventory.json")
        
        print(f"Loading AeroGuard Phase III Intelligence...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Load Inventory & Ledger
        self.inventory = self._load_json(self.inventory_path)
        self._init_ledger()

    def _load_json(self, path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}

    def _init_ledger(self):
        """Ensures the ledger exists and starts the chain."""
        if not os.path.exists(LEDGER_PATH):
            with open(LEDGER_PATH, 'w') as f:
                json.dump([{"block_id": 0, "prev_hash": "GENESIS", "timestamp": str(datetime.datetime.now())}], f)

    def _generate_dqi(self, frame):
        """
        Data Quality Index (DQI):
        Calculates the 'health' of the sensor stream based on missingness and variance.
        """
        # Check for NaNs
        missing_pct = frame[SENSOR_FEATURES].isna().mean()
        
        # Check for "Frozen" sensors (Static values are often a sign of sensor failure)
        # Note: In a real-time stream, this would check a rolling window.
        # For a single frame, we check if key sensors are exactly zero or improbable.
        is_frozen = 1 if frame.get('VIB', 1) == 0 else 0 
        
        dqi_score = 1.0 - (missing_pct * 0.5) - (is_frozen * 0.5)
        return max(0.0, float(dqi_score))

    def _update_ledger(self, report):
        """Immutable Ledger: Appends report to a cryptographically linked chain."""
        try:
            with open(LEDGER_PATH, 'r+') as f:
                ledger = json.load(f)
                prev_block = ledger[-1]
                prev_hash = hashlib.sha256(json.dumps(prev_block, sort_keys=True).encode()).hexdigest()
                
                new_block = {
                    "block_id": len(ledger),
                    "prev_hash": prev_hash,
                    "report_id": report.get("integrity_hash"),
                    "status": report.get("status"),
                    "timestamp": str(datetime.datetime.now())
                }
                ledger.append(new_block)
                f.seek(0)
                json.dump(ledger, f, indent=4)
        except Exception as e:
            print(f"Ledger Error: {e}")

    def calculate_lead_time_advantage(self, frame):
        """Digital Twin Ghosting: Compares AI detection vs. OEM Hard Limits."""
        current_egt = frame.get('EGT', 0)
        # Calculate how close we are to the 'Hard Limit' alert
        distance_to_fault = HARD_LIMIT_EGT - current_egt
        return f"Detected {distance_to_fault:.1f}C before OEM Hard Limit threshold."

    def analyze_frame(self, frame_data):
        # 1. Feature Prep & DQI
        X_df = pd.DataFrame([frame_data])[SENSOR_FEATURES]
        dqi_score = self._generate_dqi(frame_data)
        X_scaled = self.scaler.transform(X_df)
        
        # 2. Anomaly Detection
        score = self.model.decision_function(X_scaled)[0]
        # In iForest, lower score = more anomalous.
        is_anomaly = self.model.predict(X_scaled)[0] == -1
        
        # 3. Probabilistic Risk Envelope
        # For iForest, we use the decision score distribution to simulate a safety window.
        # A lower score implies higher severity/certainty of the anomaly.
        # Simple heuristic mapping for demo:
        # Score 0.2 (Normal) -> High cycles
        # Score -0.1 (Anomaly) -> Low cycles
        cycles_baseline = 100
        if score < 0:
            cycles_baseline = max(1, int((0.05 + score) * 1000)) # e.g. -0.01 -> 40 cycles? No logic is weird.
            # Use abs(score) approach from prompt
            lower_bound = max(1, int(abs(score) * 100)) # Simulated cycles remaining
        else:
            lower_bound = 500 # Safe
            
        upper_bound = int(lower_bound * 1.25)
        
        # 4. Phase Normalization (Optional context check)
        phase_label = "Cruise/Other" if frame_data.get('ALT', 0) > 10000 else "Maneuver"
        
        # Logic Gate: Only alert if Anomaly AND DQI is good
        if is_anomaly and dqi_score >= DQI_THRESHOLD:
            # 5. XAI & Logistics
            shap_values = self.explainer.shap_values(X_scaled)
            contributions = dict(zip(SENSOR_FEATURES, shap_values[0]))
            top_driver = max(contributions, key=lambda k: abs(contributions[k]))
            
            part_info = self.inventory.get(top_driver, {}) 
            
            report = {
                "status": "INSPECTION REQUIRED",
                "dqi_confidence": "HIGH" if dqi_score > 0.9 else "MEDIUM",
                "risk_envelope": f"{lower_bound} - {upper_bound} Cycles",
                "diagnosis": {
                    "primary_sensor": top_driver,
                    "lead_time_adv": self.calculate_lead_time_advantage(frame_data),
                    "task_card": part_info.get("task_card_id", "GEN-01")
                },
                "logistics": {
                    "stock": part_info.get("location", "Unknown"),
                    "manual_text": part_info.get("manual_text", "Refer to AMM.")
                }
            }
            report["integrity_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
            self._update_ledger(report) # Commit to Ledger
            return report
        
        return {"status": "NORMAL", "dqi": dqi_score}

# Simple Test Harness
if __name__ == "__main__":
    engine = AeroGuardEngine()
    print("Engine Initialized. Ledger created at:", LEDGER_PATH)
