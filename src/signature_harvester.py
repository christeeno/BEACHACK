
import numpy as np
import pandas as pd
import json
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from src.config import SENSOR_FEATURES

class SignatureHarvester:
    def __init__(self, knowledge_path="src/harvested_knowledge.json"):
        self.knowledge_path = knowledge_path
        self.known_signatures = []
        self.known_labels = []
        self._load_knowledge()

    def _load_knowledge(self):
        """Loads captured signatures and their labels/task cards."""
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, 'r') as f:
                    data = json.load(f)
                    self.known_signatures = [item['signature'] for item in data]
                    self.known_labels = data # Keep full metadata
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
                self.known_signatures = []
                self.known_labels = []
        else:
            self.known_signatures = []
            self.known_labels = []

    def extract_fault_signature(self, shap_values, telemetry_frame):
        """
        Converts a fault event into a normalized 'DNA' vector.
        Vector = Normalized(SHAP_Contributions + Key_Sensor_Ratios)
        For simplicity V1: Just uses normalized SHAP vector.
        """
        # Ensure shap_values is a flat vector matching SENSOR_FEATURES count
        if isinstance(shap_values, dict):
            vec = np.array([shap_values.get(f, 0.0) for f in SENSOR_FEATURES])
        elif isinstance(shap_values, (list, np.ndarray)):
            vec = np.array(shap_values).flatten()[:len(SENSOR_FEATURES)]
        else:
            return np.zeros(len(SENSOR_FEATURES)).tolist()
            
        # Normalize to unit vector (direction matters more than magnitude for 'Shape' of fault)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        return vec.tolist()

    def find_match(self, signature_vector, threshold=0.9):
        """
        Checks if this signature matches a known one.
        Returns: (MatchFound: bool, MetaData: dict)
        """
        if not self.known_signatures:
            return False, {}
            
        # Compute Cosine Similarity against all knowns
        # signature_vector is [1, N], knowns is [M, N]
        known_matrix = np.array(self.known_signatures)
        sig_matrix = np.array(signature_vector).reshape(1, -1)
        
        sims = cosine_similarity(sig_matrix, known_matrix)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        
        if best_score > threshold:
            match_meta = self.known_labels[best_idx]
            match_meta['similarity_score'] = float(best_score)
            return True, match_meta
            
        return False, {"closest_match_score": float(best_score)}

    def archive_signature(self, signature_vector, label, manual_ref):
        """
        Saves a new signature with human-provided context.
        """
        new_entry = {
            "signature": signature_vector,
            "label": label,
            "manual_reference": manual_ref,
            "timestamp": "Now" # In real app use datetime
        }
        
        # Load fresh (in case of concurrency), append, save
        self._load_knowledge()
        self.known_labels.append(new_entry) 
        # structure update for consistency
        with open(self.knowledge_path, 'w') as f:
            json.dump(self.known_labels, f, indent=4)
        
        # Update in-memory lists
        self.known_signatures.append(signature_vector)
        print(f"Signature Archived: {label}")

