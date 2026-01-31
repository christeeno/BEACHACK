
import pandas as pd
import numpy as np
from src.aeroguard_engine import AeroGuardEngine
from src.config import SENSOR_FEATURES

def test_phase_iv_logic():
    print("--- AeroGuard Phase IV Validation Protocol ---")
    try:
        engine = AeroGuardEngine()
    except Exception as e:
        print(f"FAILED to initialize Engine: {e}")
        return

    # 1. Normal Frame
    normal_frame = {feat: 0.0 for feat in SENSOR_FEATURES} # Assuming scaled 0 is mean/normal
    # In reality, raw values should be provided because engine scales them.
    # We should provide RAW "nominal" values if possible, or just values that we know map to normal.
    # Since I don't have the scaler means handy easily without loading it, I'll rely on the synthetic data generation logic
    # or just use values that likely won't trigger iforest.
    # Let's mock a "nominal" frame based on the fact that the scaler centers data.
    # But wait, `analyze_frame` takes raw data and scales it. 
    # So I should provide plausible raw values.
    print("Loading valid sample from dataset...")
    df = pd.read_csv("data/synthetic_dashlink.csv")
    # Take a random sample from the middle (likely cruise/normal)
    normal_frame = df.iloc[len(df)//2][SENSOR_FEATURES].to_dict()
    
    print("\n[TEST 1] Normal Flight Condition")
    res = engine.analyze_frame(normal_frame)
    print(res)

    # 2. Critical Anomaly (Thermal Runaway)
    critical_frame = normal_frame.copy()
    critical_frame['EGT'] = 1200 # Way above hard limit (900)
    critical_frame['N1'] = 98 # High load
    
    print("\n[TEST 2] Critical Anomaly (Thermal Runaway)")
    res = engine.analyze_frame(critical_frame)
    print(res)

    # 3. DQI Inhibition (Frozen Sensor)
    dqi_frame = normal_frame.copy()
    dqi_frame['VIB'] = 0.0 # Frozen
    # And maybe some missing data if I could inject NaNs, but dict keys are presence.
    # let's set VIB to 0 (Frozen logic in engine: if VIB == 0 -> is_frozen=1)
    
    print("\n[TEST 3] DQI Safety Interlock (Frozen VIB)")
    res = engine.analyze_frame(dqi_frame)
    print(res)

    # 4. Physics Violation (High MSE)
    # Random noise that doesn't make physical sense
    physics_frame = {k: np.random.uniform(-1000, 1000) for k in SENSOR_FEATURES}
    
    print("\n[TEST 4] Physics Validation (Random Noise / High MSE)")
    res = engine.analyze_frame(physics_frame)
    print(res)

if __name__ == "__main__":
    test_phase_iv_logic()
