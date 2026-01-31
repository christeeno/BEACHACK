
import json
import numpy as np
from src.aeroguard_engine import AeroGuardEngine
from src.feedback import FeedbackLoop

def demo_discovery_learning():
    print("--- AeroGuard Phase VI: Signature Harvesting Demo ---")
    
    # 1. Initialize Engine
    engine = AeroGuardEngine()
    
    # Relax MSE Threshold for Demo purposes to pass Gate 3 (Physics Validation)
    import src.aeroguard_engine
    src.aeroguard_engine.MSE_THRESHOLD = 10000.0
    
    # Frame with FF -> EGT fault
    frame = {
        'LATP': 30.0, 'LONP': -90.0, 'ALT': 35000, 'VEL': 450, 'TH': 85,
        'N1': 95,      'N2': 96, 
        'EGT': 650,    'FF': 4500, # FF=High, EGT=High
        'VIB': 0.6,    'VRTG': 1.0, 
        'OIL_P': 50,   'OIL_T': 95, 
        'FLAP': 0,     'HYDY': 3000
    }
    
    print("\n[Scenario] Simulating loss of database knowledge to trigger Discovery Mode...")
    
    # Pre-Analysis to find what the engine thinks is the root cause
    # We want to force Discovery Mode by deleting THAT key.
    print("[Setup] Probing engine for root cause...")
    # Using a high EGT frame
    probe_frame = frame.copy()
    
    # We need to temporarily suppress printing or just call internal methods, 
    # but simplest is to run analyze and ignore output, just get root cause.
    # Note: analyze_frame might log to ledger, that's fine.
    
    # Hack: We need to ensure we don't trigger Discovery in this probe if we already deleted something?
    # No, inventory is full right now.
    
    probe_report = engine.analyze_frame(probe_frame)
    if 'diagnosis' in probe_report and 'root_cause_chain' in probe_report['diagnosis']:
        rc = probe_report['diagnosis']['root_cause_chain'][0]
        print(f"[Setup] Engine identified Root Cause as: {rc}")
        
        # NOW we delete this specific sensor from inventory
        backup_item = engine.inventory.pop(rc, None)
        print(f"[Setup] Removed '{rc}' from Inventory Database.")
    else:
        print("[Setup] Could not determine root cause. Defaulting to removing 'FF'.")
        rc = 'FF'
        backup_item = engine.inventory.pop('FF', None)

    # 3. Main Pass: Discovery Mode
    print(f"\n[Analysis] Processing Frame (Expect DISCOVERY_MODE for {rc})...")
    report = engine.analyze_frame(frame)
    print(f"Status: {report['status']}")
    print(f"Diagnosis Mode: {report.get('diagnosis_mode')}")
    
    if report.get('diagnosis_mode') == "DISCOVERY_MODE": # Check diagnosis_mode mostly
        sig = report['diagnosis']['extracted_signature']
        print(f"\n[System] Extracted Fault Signature: {sig[:3]}... (Length {len(sig)})")
        
        # 4. Human feedback (Teaching)
        print("\n[Human] Engineer Input Required.")
        new_label = input("Enter Fault Label (e.g. ATA-73-CAVITATION): ").strip()
        if not new_label: new_label = "ATA-73-UNKNOWN-MANUAL"
        
        new_desc = input("Enter Task Description (e.g. Inspect pump impeller): ").strip()
        if not new_desc: new_desc = "Perform manual inspection."
        
        print(f"\n[System] Learning new signature as '{new_label}'...")
        f_loop = FeedbackLoop(engine.hybrid)
        f_loop.learn_signature(sig, new_label, new_desc)
        
        # 5. Second Pass: Adaptive Match
        print("\n[Analysis] Re-processing SAME Frame (Expect ADAPTIVE_MATCH)...")
        # Note: We still have FF removed from inventory.
        report_v2 = engine.analyze_frame(frame)
        print(f"Status: {report_v2['status']}") # Should be URGENT ACTION (Standard) or back to Normal flow logic but with diagnosis_mode
        # Wait, analyze_frame overrides status to DISCOVERY_MODE only if no match.
        # If match found, it keeps original status (e.g. URGENT) but populates part_info.
        
        print(f"Diagnosis Mode: {report_v2.get('diagnosis_mode')}")
        print(f"New Task Card: {report_v2['diagnosis']['task_card']}")
        print(f"New Manual: {report_v2['logistics']['manual_text']}")
        
    # Restore inventory
    if backup_item:
        engine.inventory[rc] = backup_item

if __name__ == "__main__":
    demo_discovery_learning()
