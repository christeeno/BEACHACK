
import json
import os
from src.aeroguard_engine import AeroGuardEngine
from src.feedback import FeedbackLoop
from src.config import SENSOR_FEATURES # needed?

def get_user_input():
    print("\n[Operator Terminal] Engine Reported: Risk of Failure.")
    print("Diagnosis: Fuel Flow (FF) caused EGT Spike.")
    print("Do you agree with this diagnosis? (y/n)")
    return input("> ").strip().lower()

def demo_feedback_loop():
    print("--- AeroGuard Phase V: Feedback Loop Demo ---")
    
    # 1. Setup Engine & Feedback Loop
    engine = AeroGuardEngine()
    # Mocking the Optimizer/Model part since we are focusing on Causal Feedback here
    # For a real weight update, we'd need the actual model instance passed to FeedbackLoop
    f_loop = FeedbackLoop(engine.hybrid) 
    
    # 2. Simulate the "Bad" Diagnosis
    # Let's say the engine thinks FF -> EGT
    parent = "FF"
    child = "EGT"
    
    # 3. Ask Operator
    choice = get_user_input()
    
    if choice == 'n':
        print("\n[Action] REJECTING Diagnosis.")
        print(f"[Feedback] Marking causal link {parent} -> {child} as INCORRECT.")
        
        # 4. Correct the Model (Blacklist the edge)
        f_loop.refine_causal_link(parent, child, user_rejected=True)
        
        print("\n[System] Optimization Complete. Future alerts will NOT assume this causality.")
        
        # Verify Blacklist
        if os.path.exists("src/causal_blacklist.json"):
            with open("src/causal_blacklist.json", 'r') as f:
                print("Current Blacklist:", f.read())
                
    else:
        print("\n[Action] ACCEPTING Diagnosis.")
        print("[Feedback] Diagnosis Confirmed. Reinforcing Model confidence (No Change).")

if __name__ == "__main__":
    demo_feedback_loop()
