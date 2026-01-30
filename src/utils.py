import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from src.config import RANDOM_SEED, SENSOR_FEATURES, SEQUENCE_LENGTH

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_synthetic_data(num_flights=50, max_cycles=200):
    """
    Generates a synthetic DASHlink-like dataset.
    Mimics degradation in EGT, N1, VIB over cycles.
    """
    data = []
    
    for flight_id in range(1, num_flights + 1):
        # Random initial health state
        health_deg = 0
        
        # Flight duration variations
        cycles = np.random.randint(50, max_cycles)
        
        for cycle in range(cycles):
            # Phase simulation (simplified: 0-10% climb, 10-90% cruise, 90-100% descent)
            phase_progress = cycle / cycles
            if phase_progress < 0.1:
                phase = 'CLIMB'
                alt = 1000 + (25000 * (phase_progress / 0.1))
                th = 90 + np.random.normal(0, 1)
            elif phase_progress > 0.9:
                phase = 'DESCENT'
                alt = 25000 - (25000 * ((phase_progress - 0.9) / 0.1))
                th = 30 + np.random.normal(0, 1)
            else:
                phase = 'CRUISE'
                alt = 25000 + np.random.normal(0, 100)
                th = 75 + np.random.normal(0, 1) # Constant thrust in cruise
            
            # Linear degradation accumulation
            health_deg += np.random.normal(0.01, 0.002)
            
            # Generate Features based on relationships described by user
            
            # 1. State
            latp = 10.0 + np.random.normal(0, 0.1)
            lonp = 20.0 + np.random.normal(0, 0.1)
            vel = th * 5 + np.random.normal(0, 5) # Speed correlated with thrust
            
            # 2. Propulsion (The "Heart") with Degradation
            # N1/N2 track Throttle (TH) but diverge slightly with health degradation
            n1 = th * 1.05 - (health_deg * 0.5) + np.random.normal(0, 0.5)
            n2 = th * 1.10 - (health_deg * 0.4) + np.random.normal(0, 0.5)
            
            # EGT rises with degradation (Efficiency drift)
            # Baseline EGT depends on Throttle
            egt_baseline = 400 + (th * 3) 
            egt = egt_baseline + (health_deg * 10) + np.random.normal(0, 2)
            
            # Fuel Flow: rises for same thrust as engine degrades
            ff_baseline = th * 100 
            ff = ff_baseline + (health_deg * 50) + np.random.normal(0, 10)
            
            oil_p = 80 - (health_deg * 0.1) + np.random.normal(0, 1)
            oil_t = 90 + (health_deg * 0.2) + np.random.normal(0, 1)
            
            # 3. Vibration & Structural
            # Anomalies: Random shocks injected
            vib_base = th * 0.01
            if np.random.random() > 0.98:
                 # Structural shock / Hard landing / Turbulence
                vib = vib_base * 5 + np.random.normal(0, 0.5)
                vrtg = 1.0 + np.random.normal(0.5, 0.2)
            else:
                vib = vib_base + (health_deg * 0.05)
                vrtg = 1.0 + np.random.normal(0, 0.05)

            # 4. Integrity
            hydy = 3000 + np.random.normal(0, 10) # Hydraulic pressure
            flap = 0 if phase == 'CRUISE' else (15 if phase == 'CLIMB' else 30)
            
            # Remaining Useful Life (Simplified target)
            rul = max_cycles - cycle
            
            row = {
                'FLIGHT_ID': flight_id,
                'CYCLE': cycle,
                'FLIGHT_PHASE': phase,
                'LATP': latp, 'LONP': lonp, 'ALT': alt, 'VEL': vel,
                'TH': th, 'N1': n1, 'N2': n2, 'EGT': egt, 'FF': ff,
                'VIB': vib, 'VRTG': vrtg, 'OIL_P': oil_p, 'OIL_T': oil_t,
                'FLAP': flap, 'HYDY': hydy,
                'RUL': rul
            }
            data.append(row)
            
    return pd.DataFrame(data)

def load_data(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
    else:
        print("Data file not found. Generating synthetic DASHlink data using logic...")
        df = generate_synthetic_data()
        # Save for inspection
        os.makedirs(os.path.dirname("data/synthetic_dashlink.csv"), exist_ok=True)
        df.to_csv("data/synthetic_dashlink.csv", index=False)
        return df

def prepare_sequences(df, sequence_length=SEQUENCE_LENGTH, features=SENSOR_FEATURES):
    """
    Prepares sequences for LSTM.
    Returns: X (batch, seq_len, features), y (batch, 1)
    """
    # Normalize features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X = []
    y = []
    
    # Group by Flight ID to avoid mixing flights in sequences
    for _, group in df.groupby('FLIGHT_ID'):
        data = group[features].values
        labels = group['RUL'].values
        
        if len(data) < sequence_length:
            continue
            
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(labels[i + sequence_length])
            
    return np.array(X), np.array(y), scaler
