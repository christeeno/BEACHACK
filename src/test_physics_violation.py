
import pandas as pd
import numpy as np
from src.causal_discovery import CausalDiscoverer

def test_physics_priors():
    print("--- Phase V: Physics Priors Verification ---")
    
    discoverer = CausalDiscoverer()
    
    # Create data that VIOLATES physics
    # EGT -> TH (Temperature causes throttle to move)
    # Theoretically forbidden by ['EGT', 'TH'] in discoverer
    
    n_samples = 1000
    egt = np.random.normal(600, 10, n_samples)
    th = np.random.normal(80, 5, n_samples)
    
    # Inject False Causality: EGT spikes, THEN Throttle moves
    egt[500:510] += 200
    th[502:512] += 10 
    
    data = pd.DataFrame({'EGT': egt, 'TH': th, 'N1': np.zeros(n_samples)})
    
    # Run Discovery
    discoverer.learn_causal_structure(data, max_lag=5)
    
    # Check if edge exists
    edge_exists = discoverer.adjacency_matrix.loc['EGT', 'TH'] == 1
    
    if not edge_exists:
        print("\n[SUCCESS] Physics Prior correctly BLOCKED forbidden edge (EGT -> TH).")
    else:
        print("\n[FAILURE] System allowed forbidden physics edge (EGT -> TH).")

if __name__ == "__main__":
    test_physics_priors()
