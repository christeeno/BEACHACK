
import numpy as np
import pandas as pd

class CausalDiagnosisEngine:
    def __init__(self):
        # Expanded AMM & Tooling Map
        self.knowledge_base = {
            "Internal Turbine Degradation": {
                "amm": "AMM 72-50: Turbine Section Inspection",
                "tooling": ["Borescope", "Flashlight"],
                "severity": "High"
            },
            "Fuel Nozzle Clogging": {
                "amm": "AMM 73-10: Fuel Injector Cleaning",
                "tooling": ["Flow Test Bench", "Wrench Set"],
                "severity": "Medium"
            },
            "Rotating Component Imbalance": {
                "amm": "AMM 72-00: Vibration Analysis",
                "tooling": ["Vibration Analyzer", "Trim Weights"],
                "severity": "Critical"
            },
            "Bearing Fatigue": {
                "amm": "AMM 72-20: Bearing Sump Inspection",
                "tooling": ["Magnetic Chip Detector"],
                "severity": "Critical"
            },
            "Hard Landing Event": {
                "amm": "AMM 05-51: Structural Inspection",
                "tooling": ["Leveling Equipment", "NDT Kit"],
                "severity": "Mandatory"
            },
            "Sensor Drift / Noise": {
                "amm": "AMM 31-10: Instrument Calibration",
                "tooling": ["Multimeter", "Signal Generator"],
                "severity": "Low"
            }
        }

    def infer_phase(self, telemetry):
        """Simple phase inference if not provided"""
        alt = telemetry.get('ALT', 0)
        vel = telemetry.get('VEL', 0)
        n1 = telemetry.get('N1', 0)
        
        if alt < 1000 and n1 > 85: return "Takeoff"
        if alt > 20000: return "Cruise"
        if alt < 2000 and vel < 150: return "Landing"
        return "Unknown"

    def diagnose_anomaly(self, shap_values, telemetry_snapshot, context_phase=None):
        """
        Ingests SHAP attribution and Telemetry Context to determine Root Cause.
        
        Args:
            shap_values (dict): Feature name -> SHAP contribution value
            telemetry_snapshot (dict/Series): Raw sensor values
            context_phase (str): Flight phase (Takeoff, Cruise, Landing)
        """
        if context_phase is None:
            context_phase = self.infer_phase(telemetry_snapshot)
            
        # 1. Identify Primary Driver (feature with max absolute SHAP impact)
        # We focus on the one pushing towards anomaly (usually negative direction for iForest)
        # But here we take max magnitude.
        
        sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        primary_driver, primary_shap = sorted_drivers[0]
        
        root_cause = "Unknown Anomaly"
        confidence = 0.5
        reasoning = []
        
        # Helper accessors
        def val(k): return telemetry_snapshot.get(k, 0)
        
        # --- Part 2: Boolean Physics Logic ---
        
        # Rule A: EGT Analysis
        if primary_driver == 'EGT':
            # "If EGT > Threshold AND Fuel_Flow < Mean (Stable)"
            # We use heuristics for "Stable/Normal" here since we don't have rolling mean handy
            # Assume FF is not spiking with EGT
            ff_val = val('FF')
            egt_val = val('EGT')
            
            # Logic: If EGT is hot but Fuel Flow is normal, it's not simply "more fuel burning"
            # It implies poor efficiency or cooling failure.
            # If FF was also high, it might just be high throttle.
            
            # Check secondary driver
            secondary_driver = sorted_drivers[1][0] if len(sorted_drivers) > 1 else None
            
            if secondary_driver == 'FF':
                root_cause = "Fuel Control Unit (High Flow)"
                confidence = 0.8
                reasoning.append("EGT High driven by Fuel Flow Spike.")
            else:
                root_cause = "Internal Turbine Degradation"
                confidence = 0.9
                reasoning.append("EGT Spike with Stable Fuel Flow implies efficiency loss.")

        # Rule B: Vibration Analysis (Phase Dependent)
        elif primary_driver in ['VIB', 'N1', 'N2']:
            vib_mag = val('VIB')
            
            if context_phase == "Takeoff":
                if vib_mag > 3.0: # Arbitrary high threshold
                    root_cause = "Rotating Component Imbalance"
                    confidence = 0.95
                    reasoning.append(f"Critical Vibration ({vib_mag}) during Takeoff.")
                else:
                    root_cause = "Normal Operational Variance"
                    confidence = 0.6
                    reasoning.append(f"Elevated Vibration ({vib_mag}) is typical during Takeoff.")
            elif context_phase == "Cruise":
                root_cause = "Bearing Fatigue"
                confidence = 0.85
                reasoning.append("Vibration spike during stable Cruise indicates Hardware Wear.")
            else:
                root_cause = "Engine Mount Looseness"
                confidence = 0.5
                reasoning.append("Vibration detected in transient phase.")

        # Rule C: Structural / Hard Landing
        elif primary_driver == 'VRTG':
            vrtg_val = val('VRTG')
            if context_phase == "Landing" or context_phase == "Unknown": # Assume landing if high G
                root_cause = "Hard Landing Event"
                confidence = 0.99
                reasoning.append(f"Vertical G-Load ({vrtg_val}) exceeding limits at Touchdown.")
            else:
                root_cause = "Turbulence / Airframe Stress"
                confidence = 0.7
                reasoning.append("G-Load spike during flight.")

        # Fallback
        else:
            root_cause = f"Unmapped Anomaly ({primary_driver})"
            confidence = 0.3
            reasoning.append("Sensor correlation pattern not recognized.")

        # --- Part 3 & 4: Actionable Directive & Guardrail ---
        
        kb_entry = self.knowledge_base.get(root_cause, self.knowledge_base["Sensor Drift / Noise"])
        
        # Transparency Guardrail
        if confidence < 0.4:
            kb_entry = self.knowledge_base["Sensor Drift / Noise"]
            msg = "Low Diagnostic Confidence: Manual inspection required to verify sensor integrity."
        else:
            msg = f"Determined {root_cause} with {int(confidence*100)}% Logic Confidence."

        return {
            "root_cause_diagnosis": root_cause,
            "confidence_score": round(confidence, 2),
            "evidence_summary": "; ".join(reasoning),
            "statistical_evidence": f"{primary_driver} SHAP: {primary_shap:.4f}",
            "mechanical_diagnosis": msg,
            "maintenance_action": {
                "amm_reference": kb_entry["amm"],
                "required_tooling": kb_entry["tooling"],
                "severity_class": kb_entry["severity"]
            }
        }

# Example Usage Block (for demonstration)
if __name__ == "__main__":
    engine = CausalDiagnosisEngine()
    
    # Mock Data: EGT High, FF Normal (Cruise)
    mock_shap = {'EGT': 0.8, 'FF': 0.1, 'N1': 0.05}
    mock_telemetry = {'EGT': 950, 'FF': 3000, 'ALT': 35000}
    
    print("--- Test Case 1: Thermal Issue ---")
    result = engine.diagnose_anomaly(mock_shap, mock_telemetry, "Cruise")
    import json
    print(json.dumps(result, indent=2))
    
    # Mock Data: Vib High during Takeoff
    print("\n--- Test Case 2: Takeoff Vib (Normal) ---")
    mock_shap_vib = {'VIB': 0.9, 'N1': 0.2}
    mock_telemetry_vib = {'VIB': 1.2, 'ALT': 500, 'N1': 95} # 1.2 is mild
    result2 = engine.diagnose_anomaly(mock_shap_vib, mock_telemetry_vib, "Takeoff")
    print(json.dumps(result2, indent=2))
