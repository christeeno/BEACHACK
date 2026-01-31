
import pandas as pd
import numpy as np
import os
from src.causal_discovery import CausalDiscoverer

def test_causal_directionality():
    print("--- Phase V: Causal Directionality Verification ---")
    
    # Generate Synthetic Causal Data: A -> B with lag
    # A = Fuel Flow (FF), B = EGT
    # Logic: FF spikes at t=10, EGT spikes at t=12
    
    n_samples = 1000
    t = np.arange(n_samples)
    
    # Random Baseline
    ff = np.random.normal(3000, 50, n_samples)
    egt = np.random.normal(600, 10, n_samples)
    
    # Inject Signal
    # FF Spike
    ff[500:510] += 500
    # EGT Reaction (Lag = 2 samples)
    egt[502:512] += 100
    
    data = pd.DataFrame({'FF': ff, 'EGT': egt, 'N1': np.random.normal(0,1,n_samples)})
    
    # Run Discovery
    discoverer = CausalDiscoverer()
    discoverer.learn_causal_structure(data, max_lag=5)
    
    print("\nAdjacency Matrix (FF=Row, EGT=Col means FF->EGT):")
    print(discoverer.adjacency_matrix.loc[['FF'], ['EGT']])
    
    # Assert FF -> EGT exists
    has_ff_egt = discoverer.adjacency_matrix.loc['FF', 'EGT'] == 1
    
    # Assert EGT -> FF does NOT exist (Physics Prior or simply Granger logic)
    has_egt_ff = discoverer.adjacency_matrix.loc['EGT', 'FF'] == 1
    
    if has_ff_egt and not has_egt_ff:
        print("\n[SUCCESS] Correctly identified FF -> EGT (Fuel Flow causes Temp Rise).")
    elif has_egt_ff:
        print("\n[FAILURE] False Reverse Causality (EGT -> FF) detected.")
    else:
        print("\n[FAILURE] Failed to detect causal link.")

if __name__ == "__main__":
    test_causal_directionality()
