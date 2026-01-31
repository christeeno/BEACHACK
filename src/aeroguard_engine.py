
import shap
import joblib
import pandas as pd
import numpy as np
import os
import json
import datetime
import torch
import warnings
from src.config import ARTIFACT_DIR, SENSOR_FEATURES, DQI_THRESHOLD, LEDGER_PATH, HARD_LIMIT_EGT, MSE_THRESHOLD, DATA_PATH
from src.models import HybridModel # New Phase IV Model Class
from src.causal_discovery import CausalDiscoverer # Phase V
from src.signature_harvester import SignatureHarvester # Phase VI

warnings.filterwarnings('ignore')

class AeroGuardEngine:
    def __init__(self):
        # Paths
        self.lstm_path = os.path.join(ARTIFACT_DIR, "aeroguard_v1_lstm.pth")
        self.ae_path = os.path.join(ARTIFACT_DIR, "aeroguard_v1_ae.pth") 
        self.iforest_path = os.path.join(ARTIFACT_DIR, "aeroguard_v1_iforest.joblib")
        self.scaler_path = os.path.join(ARTIFACT_DIR, "aeroguard_v1_scaler.joblib")
        self.inventory_path = os.path.join(os.path.dirname(__file__), "inventory.json")
        
        print(f"Loading AeroGuard Phase IV Intelligence (Certification-Grade)...")
        
        # Load Artifacts
        if not os.path.exists(self.iforest_path):
            raise FileNotFoundError("Model artifacts not found. Please run src/train.py first.")
            
        self.iforest = joblib.load(self.iforest_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Initialize Hybrid Model Container
        self.hybrid = HybridModel(len(SENSOR_FEATURES), iforest=self.iforest, scaler=self.scaler)
        
        if os.path.exists(self.lstm_path):
            self.hybrid.lstm.load_state_dict(torch.load(self.lstm_path))
            self.hybrid.lstm.eval()
        else:
            print("Warning: LSTM weights not found.")
            
        if os.path.exists(self.ae_path):
            self.hybrid.ae.load_state_dict(torch.load(self.ae_path))
            self.hybrid.ae.eval()
        else:
            print("Warning: Autoencoder weights not found.")

        self.explainer = shap.TreeExplainer(self.iforest)
        self.model_hash = self._compute_file_hash(self.lstm_path) 
        
        # Phase V: Causal Discovery
        print("Initializing Causal Discovery Layer...")
        self.causal_engine = CausalDiscoverer()
        # In a real system, we'd load a persistent graph. Here we learn from synthetic baseline.
        if os.path.exists(DATA_PATH):
             # Ideally this is done offline, but for Demo we do it on init
             df_learn = pd.read_csv(DATA_PATH).head(2000)
             self.causal_engine.learn_causal_structure(df_learn)

        # Phase VI: Signature Harvesting
        self.harvester = SignatureHarvester()
        
        # Load Inventory & Ledger
        self.inventory = self._load_json(self.inventory_path)
        self._init_ledger()

    def _compute_file_hash(self, path):
        """Cryptographic Proof of Training - DISABLED"""
        return "HASH_DISABLED_BY_USER"

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
                    "active_model_hash": self.model_hash 
                }
                json.dump([genesis], f)

    def _generate_dqi(self, frame):
        if isinstance(frame, dict):
            frame = pd.DataFrame([frame])
            
        # Calculate missing percentage across features
        missing_pct = frame[SENSOR_FEATURES].isna().values.mean()
        
        # Frozen check (simplified for VIB)
        vib_val = frame['VIB'].iloc[0] if 'VIB' in frame.columns else 1
        is_frozen = 1 if vib_val == 0 else 0 
        
        dqi_score = 1.0 - (missing_pct * 0.5) - (is_frozen * 0.5)
        return max(0.0, float(dqi_score))

    def _update_ledger(self, report):
        try:
            with open(LEDGER_PATH, 'r+') as f:
                content = f.read()
                if not content:
                    ledger = []
                else: 
                    ledger = json.loads(content)
                    
                prev_block = ledger[-1] if ledger else {}
                prev_hash = "N/A"
                
                new_block = {
                    "block_id": len(ledger),
                    "prev_hash": prev_hash,
                    "report_id": report.get("integrity_hash", "N/A"),
                    "status": report.get("status"),
                    "model_integrity": self.model_hash,
                    "causal_matrix_hash": self.causal_engine.get_adjacency_hash(), # Phase V: Causal Provenance
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
        if distance_to_fault < 0:
             return "Fault already present (exceeds OEM limit)."
        return f"Detected {distance_to_fault:.1f}C before OEM Hard Limit threshold."

    def analyze_frame(self, frame_data):
        # Prepare Data
        X_df = pd.DataFrame([frame_data])[SENSOR_FEATURES]
        X_scaled = self.scaler.transform(X_df)
        X_tensor = torch.FloatTensor(X_scaled) # [1, features]
        
        # --- GATE 1: Statistical Separation (Isolation Forest) ---
        score = self.iforest.decision_function(X_scaled)[0]
        
        # --- GATE 2: DQI Safety Interlock ---
        dqi_score = self._generate_dqi(frame_data)
        if dqi_score < DQI_THRESHOLD:
            # Inhibited due to sensor quality
            report = {
                "status": "NORMAL", 
                "dqi": round(dqi_score, 2),
                "model_integrity": self.model_hash,
                "note": "Anomaly Signal Suppressed due to Low Sensor Health (DQI < 0.7). Maintenance not dispatched."
            }
            return report

        # --- GATE 3: Physics Validation (Autoencoder) ---
        mse = self.hybrid.get_reconstruction_mse(X_tensor)
        if mse > MSE_THRESHOLD:
             report = {
                "status": "DATA_INVALID_INHIBIT",
                "reason": f"Physics Violation (Reconstruction Error: {mse:.2f})",
                "dqi_confidence": "ZERO",
                "model_integrity": self.model_hash
             }
             report["integrity_hash"] = "N/A"
             self._update_ledger(report)
             return report
             
        # --- ANOMALY PROCESSING ---
        if score < 0:
            status = "INSPECTION REQUIRED" if score >= -0.1 else "URGENT ACTION REQUIRED"
            
            # Predict RUL Quantiles for risk_envelope
            q = self.hybrid.predict_rul_quantiles(X_tensor) # [5%, 50%, 95%]
            risk_env = f"{int(max(1, q[0]))} - {int(q[2])} Cycles"
            
            # Localization
            shap_values = self.explainer.shap_values(X_scaled)
            contributions = dict(zip(SENSOR_FEATURES, shap_values[0]))
            top_driver = max(contributions, key=lambda k: abs(contributions[k]))
            
            # Phase V: Causal Root Cause Analysis
            causal_chain = self.causal_engine.find_root_cause(top_driver)
            root_cause_sensor = causal_chain[0]
            
            part_info = self.inventory.get(root_cause_sensor)
            
            note = ""
            diagnosis_mode = "STANDARD"
            
            # Phase VI: Signature Harvesting Fallback
            if not part_info:
                # 1. Harvest Signature
                sig_vector = self.harvester.extract_fault_signature(shap_values[0], frame_data)
                
                # 2. Check Archives
                match_found, meta = self.harvester.find_match(sig_vector)
                
                if match_found:
                    # Adaptive Success
                    diagnosis_mode = "ADAPTIVE_MATCH"
                    part_info = {
                        "task_card_id": meta.get('label', 'ADAPTIVE-TASK'),
                        "manual_text": meta.get('manual_reference', 'Derived from historical learning.'),
                        "location": "See Logistics"
                    }
                    note = f"Identified via Signature Match ({meta.get('similarity_score',0):.2f})"
                else:
                    # 3. Discovery Mode
                    diagnosis_mode = "DISCOVERY_MODE"
                    status = "DISCOVERY_MODE" # Override status
                    part_info = {
                        "task_card_id": "UNKNOWN-SIG",
                        "manual_text": "Unknown system signature detected. Performing forensic data capture.",
                        "location": "N/A"
                    }
                    note = "New Failure Mode Detected. Awaiting Human Labeling."
                    # Capture signature in report for Ledger/Feedback
                    
            if part_info is None: # Safety fallback if harvester fails logic
                 part_info = {}

            report = {
                "status": status,
                "diagnosis_mode": diagnosis_mode,
                "dqi_confidence": "HIGH" if dqi_score > 0.9 else "MEDIUM",
                "risk_envelope": risk_env,
                "diagnosis": {
                    "primary_sensor": top_driver, # Symptom
                    "root_cause_chain": causal_chain, # [Root, ..., Symptom]
                    "lead_time_adv": self.calculate_lead_time_advantage(frame_data),
                    "task_card": part_info.get("task_card_id", "GEN-01"),
                    "extracted_signature": self.harvester.extract_fault_signature(shap_values[0], frame_data) if diagnosis_mode == "DISCOVERY_MODE" else None
                },
                "logistics": {
                    "stock": part_info.get("location", "Unknown"),
                    "manual_text": part_info.get("manual_text", "Refer to AMM.")
                },
                "note": note,
                "model_integrity": self.model_hash
            }
            report["integrity_hash"] = "N/A"
            self._update_ledger(report) 
            return report
        
        # --- NORMAL ---
        return {"status": "NORMAL", "dqi": dqi_score, "model_integrity": self.model_hash}

if __name__ == "__main__":
    engine = AeroGuardEngine()
    print("Certification-Grade Engine Online.")
