import shap
import numpy as np
import torch
import pandas as pd

class ExplainabilityEngine:
    def __init__(self, model, background_data, feature_names):
        """
        Args:
            model: The trained PyTorch model (wrapper used).
            background_data: A representative sample of training data (numpy array).
            feature_names: List of feature string names.
        """
        self.model = model
        self.feature_names = feature_names
        # Summarize background data to Speed up SHAP (kmeans)
        self.background_summary = shap.kmeans(background_data.reshape(background_data.shape[0], -1), 10)
        self.explainer = None 

    def _predict_wrapper(self, X_flattened):
        """
        Wraps the model prediction for SHAP KernelExplainer.
        SHAP passes flattened arrays (samples, features*seq_len).
        We must reshape back to (samples, seq_len, features) for the LSTM.
        """
        # Infer dimensions
        num_features = len(self.feature_names)
        seq_len = X_flattened.shape[1] // num_features
        
        X_reshaped = X_flattened.reshape(-1, seq_len, num_features)
        
        # PyTorch Prediction
        self.model.lstm.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X_reshaped, dtype=torch.float32)
            preds = self.model.lstm(tensor_X).numpy()
            
        return preds.flatten()

    def initialize_explainer(self):
        # Using KernelExplainer as a generic robust method for black-box models
        # For production deep learning, GradientExplainer or DeepExplainer is preferred but finicky with LSTM states
        self.explainer = shap.KernelExplainer(self._predict_wrapper, self.background_summary)

    def explain_instance(self, X_instance_seq):
        """
        Generates explanation for a single instance.
        Args:
            X_instance_seq: (seq_len, features)
        """
        if self.explainer is None:
            self.initialize_explainer()
            
        # Flatten input
        X_flat = X_instance_seq.reshape(1, -1)
        shap_values = self.explainer.shap_values(X_flat, nsamples=50) # Reduced nsamples for speed demo
        
        # We have shap values for every timepoint x feature.
        # We average across time to get "Global Feature Importance" for this specific alert
        # shap_values[0] is (1, seq_len * features)
        
        sv_reshaped = shap_values[0].reshape(X_instance_seq.shape) # (seq_len, features)
        
        # Average contribution of each feature over the time window
        avg_feature_impact = np.mean(np.abs(sv_reshaped), axis=0) 
        
        risk_drivers = {}
        for i, feat in enumerate(self.feature_names):
            risk_drivers[feat] = avg_feature_impact[i]
            
        # Sort by impact
        sorted_drivers = dict(sorted(risk_drivers.items(), key=lambda item: item[1], reverse=True))
        
        # Get Top 3
        top_3 = dict(list(sorted_drivers.items())[:3])
        return top_3
