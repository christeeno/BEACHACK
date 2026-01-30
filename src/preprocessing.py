import pandas as pd
import numpy as np

def classify_flight_phase(df):
    """
    Classifies flight phases into 'TAKEOFF', 'CRUISE', 'LANDING' based on Altitude and Velocity.
    NOTE: In the synthetic data, 'FLIGHT_PHASE' is already generated. 
    This function is for when we have raw sensor data without labels.
    """
    if 'FLIGHT_PHASE' in df.columns:
        return df
    
    # Logic: 
    # Takeoff/Climb: Increasing Altitude, High Thrust
    # Cruise: Stable High Altitude, Stable Velocity
    # Landing/Descent: Decreasing Altitude
    
    # Simple rule-based approximation
    conditions = [
        (df['ALT'] < 10000) & (df['VEL'].diff() > 0), # Takoff/Climb
        (df['ALT'] > 10000) & (df['ALT'].diff().abs() < 50), # Cruise (Stable Alt)
        (df['ALT'] < 20000) & (df['ALT'].diff() < 0) # Descent
    ]
    choices = ['TAKEOFF', 'CRUISE', 'LANDING']
    
    # Default to Cruise if unsure or smooth out
    df['FLIGHT_PHASE'] = np.select(conditions, choices, default='CRUISE')
    return df

def calculate_efficiency_drift(df):
    """
    Calculates 'Efficiency Drift': Ratio of Fuel Flow to Thrust.
    Rising ratio with constant Thrust = Degradation.
    """
    # Avoid division by zero
    df['EFFICIENCY_DRIFT'] = df['FF'] / (df['TH'] + 1e-5)
    
    # Normalize/Smooth if needed (e.g., Rolling average)
    df['EFFICIENCY_DRIFT_SMOOTH'] = df['EFFICIENCY_DRIFT'].rolling(window=10, min_periods=1).mean()
    
    return df

def process_dataframe(df):
    """
    Master preprocessing function.
    """
    df = classify_flight_phase(df)
    df = calculate_efficiency_drift(df)
    return df
