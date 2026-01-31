
import pandas as pd
import numpy as np
import json
import os
import hashlib
from src.config import SENSOR_FEATURES

class CausalDiscoverer:
    def __init__(self):
        # Physics Priors: Forbidden Directed Edges [Cause, Effect]
        # e.g. ['EGT', 'TH']: EGT cannot cause Throttle to move (auto-throttle reacts to N1 usually, but physics-wise TH drives engine state)
        self.forbidden_edges = [
            ['EGT', 'TH'],   # Temp doesn't drive Throttle
            ['N1', 'TH'],    # RPM doesn't physically push the lever (though FADEC loop might, we treat TH as Input)
            ['VIB', 'TH'],   # Vibration doesn't cause Throttle
            ['EGT', 'FF'],   # EGT doesn't cause Fuel Flow (FF causes EGT)
            ['N1', 'FF']     # N1 doesn't put fuel in (FF drives N1)
        ]
        
        self.blacklisted_edges_path = "src/causal_blacklist.json"
        self._load_blacklist()
        
        # Adjacency Matrix (Sensor -> Sensor)
        # 1 = Causal, 0 = No Link
        self.adjacency_matrix = pd.DataFrame(
            0, index=SENSOR_FEATURES, columns=SENSOR_FEATURES
        )

    def _load_blacklist(self):
        if os.path.exists(self.blacklisted_edges_path):
            try:
                with open(self.blacklisted_edges_path, 'r') as f:
                    self.blacklisted_edges = json.load(f)
            except:
                self.blacklisted_edges = []
        else:
            self.blacklisted_edges = []

    def save_blacklist(self):
        with open(self.blacklisted_edges_path, 'w') as f:
            json.dump(self.blacklisted_edges, f)

    def learn_causal_structure(self, data_df, max_lag=5):
        """
        Learns causal edges using Lagged Correlation.
        If A(t-k) correlates with B(t) significantly higher than B(t-k) correlates with A(t),
        we infer A -> B.
        """
        correlations = {}
        
        # Reset matrix
        self.adjacency_matrix[:] = 0
        
        features = [f for f in SENSOR_FEATURES if f in data_df.columns]
        
        for a in features:
            for b in features:
                if a == b: continue
                
                # Check Priors & Blacklist
                if [a, b] in self.forbidden_edges or [a, b] in self.blacklisted_edges:
                    continue
                
                # Lagged Analysis
                # Does A lead B? (A causes B)
                score_a_leads_b = 0
                for lag in range(1, max_lag + 1):
                    # Corr(A(t-lag), B(t))
                    c = data_df[a].shift(lag).corr(data_df[b])
                    if abs(c) > score_a_leads_b:
                        score_a_leads_b = abs(c)
                        
                # Does B lead A? (B causes A)
                score_b_leads_a = 0
                for lag in range(1, max_lag + 1):
                    # Corr(B(t-lag), A(t))
                    c = data_df[b].shift(lag).corr(data_df[a])
                    if abs(c) > score_b_leads_a:
                        score_b_leads_a = abs(c)
                
                # Directionality Test
                # If A->B signal is significantly stronger than B->A (reverse causality or confounding)
                # Threshold for significance
                params = {
                    "correlation_threshold": 0.5,
                    "dominance_ratio": 1.1 # Lead must be 10% stronger than lag
                }
                
                if score_a_leads_b > params["correlation_threshold"]:
                    if score_a_leads_b > (score_b_leads_a * params["dominance_ratio"]):
                        self.adjacency_matrix.loc[a, b] = 1
                        # print(f"Discovered Causal Link: {a} -> {b} (Score: {score_a_leads_b:.2f})")

    def find_root_cause(self, symptom_sensor, depth=3):
        """
        Traverses the graph backwards to find the root parent.
        Returns a chain: [Root, ..., Middle, Symptom]
        """
        chain = [symptom_sensor]
        curr = symptom_sensor
        
        for _ in range(depth):
            # Find parents: nodes X where matrix[X, curr] == 1
            parents = self.adjacency_matrix.index[self.adjacency_matrix[curr] == 1].tolist()
            if not parents:
                break
            
            # Simple greedy path: take the first parent (or semantic prioritization)
            # In V1, we just take the first physical parent found
            parent = parents[0] 
            
            # Avoid cycles
            if parent in chain:
                break
                
            chain.insert(0, parent)
            curr = parent
            
        return chain

# import hashlib
# ...
    def get_adjacency_hash(self):
        """Returns SHA-256 of the current graph state for the Ledger - DISABLED"""
        return "HASH_DISABLED_BY_USER"

    def get_forbidden_edges(self):
        return self.forbidden_edges
