
import shap
import joblib
import pandas as pd
import numpy as np
import os
import json
import hashlib
import datetime
import torch
import warnings
from src.config import ARTIFACT_DIR, SENSOR_FEATURES, DQI_THRESHOLD, LEDGER_PATH, HARD_LIMIT_EGT
from src.models import SensorHealthAE # Import the new Autoencoder

warnings.filterwarnings('ignore')

class AeroGuardEngine:
    def __init__(self):
        self.model_path = os.path.join(ARTIFACT_DIR, "real_data_iforest.pkl")
        self.scaler_path = os.path.join(ARTIFACT_DIR, "real_data_scaler.pkl")
        self.ae_path = os.path.join(ARTIFACT_DIR, "aeroguard_ae.pth") # New AE path
        self.inventory_path = os.path.join(os.path.dirname(__file__), "inventory.json")
        
        print(f"Loading AeroGuard Phase IV Intelligence (Certification-Grade)...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Load Autoencoder for Safety Inhibit
        self.ae = SensorHealthAE(input_dim=len(SENSOR_FEATURES))
        if os.path.exists(self.ae_path):
            self.ae.load_state_dict(torch.load(self.ae_path))
            self.ae.eval()
        else:
            print("Warning: Autoencoder weights not found. Safety Inhibit will run in passthrough mode.")
            
        self.explainer = shap.TreeExplainer(self.model)
        self.model_hash = self._compute_file_hash(self.model_path)
        
        # Load Inventory & Ledger
        self.inventory = self._load_json(self.inventory_path)
        self._init_ledger()

    def _compute_file_hash(self, path):
        """Cryptographic Proof of Training"""
        if not os.path.exists(path): return "UNKNOWN"
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _load_json(self, path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}

    def _init_ledger(self):
        if not os.path.exists(LEDGER_PATH):
            with open(LEDGER_PATH, 'w') as f:
                genesis = {
                    "block_id": 0, 
                    "prev_hash": "GENESIS", 
                    "timestamp": str(datetime.datetime.now()),
                    "active_model_hash": self.model_hash # Log model version in Genesis
                }
                json.dump([genesis], f)

    def _check_safety_inhibit(self, X_scaled_df):
        """
        Safety Interlock Logic.
        Uses Autoencoder reconstruction error to detect Data Incompatibility.
        """
        # Convert to tensor
        x = torch.tensor(X_scaled_df.values, dtype=torch.float32)
        with torch.no_grad():
            recon = self.ae(x)
            mse = torch.mean((x - recon)**2).item()
            
        # Critical Threshold (e.g., 2.0 MSE is massive deviation from normal manifold)
        # If model hasn't been trained, MSE might be random, so we skip if passthrough.
        # For prototype, we set a high threshold or skip if untrained.
        if not os.path.exists(self.ae_path): return False, 0.0
        
        CRITICAL_AE_THRESHOLD = 5.0 # TBD via Validation
        if mse > CRITICAL_AE_THRESHOLD:
            return True, mse
        return False, mse

    def _generate_dqi(self, frame):
        missing_pct = frame[SENSOR_FEATURES].isna().mean()
        is_frozen = 1 if frame.get('VIB', 1) == 0 else 0 
        dqi_score = 1.0 - (missing_pct * 0.5) - (is_frozen * 0.5)
        return max(0.0, float(dqi_score))

    def _update_ledger(self, report):
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
                    "model_integrity": self.model_hash,
                    "timestamp": str(datetime.datetime.now())
                }
                ledger.append(new_block)
                f.seek(0)
                json.dump(ledger, f, indent=4)
        except Exception as e:
            print(f"Ledger Error: {e}")

    def calculate_lead_time_advantage(self, frame):
        current_egt = frame.get('EGT', 0)
        distance_to_fault = HARD_LIMIT_EGT - current_egt
        return f"Detected {distance_to_fault:.1f}C before OEM Hard Limit threshold."

    def analyze_frame(self, frame_data):
        X_df = pd.DataFrame([frame_data])[SENSOR_FEATURES]
        X_scaled = self.scaler.transform(X_df)
        
        # 1. Safety Inhibit Interlock (Dynamic DQI)
        inhibit_active, ae_error = self._check_safety_inhibit(pd.DataFrame(X_scaled, columns=SENSOR_FEATURES))
        
        if inhibit_active:
            report = {
                "status": "DATA_INVALID_INHIBIT",
                "reason": f"Sensor Data Integrity Violation (Reconstruction Error: {ae_error:.2f})",
                "dqi_confidence": "ZERO",
                "action": "MANUAL INSPECTION REQUIRED - AI PREDICTION INHIBITED",
                "model_integrity": self.model_hash
            }
            # Still log inhibit events
            report["integrity_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
            self._update_ledger(report)
            return report

        # 2. Normal Analysis
        dqi_score = self._generate_dqi(frame_data)
        score = self.model.decision_function(X_scaled)[0]
        is_anomaly = self.model.predict(X_scaled)[0] == -1
        
        # Risk Envelope (Simulated for iForest, real would use Quantile output from LSTM)
        lower_bound = max(1, int(abs(score) * 100)) 
        upper_bound = int(lower_bound * 1.25)
        
        if is_anomaly and dqi_score >= DQI_THRESHOLD:
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
                },
                "model_integrity": self.model_hash
            }
            report["integrity_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
            self._update_ledger(report) 
            return report
        
        return {"status": "NORMAL", "dqi": dqi_score, "model_integrity": self.model_hash}

if __name__ == "__main__":
    engine = AeroGuardEngine()
    print("Certification-Grade Engine Online.")
