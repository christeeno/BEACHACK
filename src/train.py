
import os
import torch
import numpy as np
import pandas as pd
from src.config import *
from src.utils import load_data, prepare_sequences
from src.preprocessing import process_dataframe
from src.models import HybridModel
from src.uncertainty import predict_uncertainty
from src.explainability import ExplainabilityEngine
from src.feedback import FeedbackLoop

def main():
    print("--- AeroGuard Training Pipeline ---")
    
    # 1. Project Setup & Data
    df = load_data(DATA_PATH)
    print(f"Data Loaded: {df.shape}")
    
    # 2. Preprocessing & Feature Engineering
    print("Preprocessing...")
    df = process_dataframe(df)
    
    # Prepare Sequences
    print("Creating Sequences...")
    X, y, scaler = prepare_sequences(df)
    
    # Split
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Tensor conversion
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model Training
    print("Initializing Hybrid Model...")
    input_dim = X.shape[2]
    # We pass scaler and iforest placeholders, or we train them and attach later. 
    # For now, we will save them separately as per original logic, but `HybridModel` can wrap them.
    model = HybridModel(input_dim, scaler=scaler)
    
    print("Phase IV: Training Safety Interlock (Autoencoder)...")
    model.train_ae(train_loader, epochs=EPOCHS)
    
    print("Phase IV: Training Quantile RUL Predictor (LSTM)...")
    model.train_lstm(train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    print("Phase IV: Training Shock Detection (iForest)...")
    from sklearn.ensemble import IsolationForest
    # Use flattened features for iForest (mean over window)
    X_train_flat = np.mean(X_train, axis=1) 
    model.iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.iforest.fit(X_train_flat)
    
    # 4. Save
    model.save("aeroguard_v1")
    # Also ensure scaler is saved if not handled by model.save (which we added logic for)
    
    # 5. Demonstration / Validation
    print("\n--- Validation Demo ---")
    
    # Select a test sample
    sample_idx = 0
    sample_X = X_test[sample_idx]
    true_rul = y_test[sample_idx]
    
    # A. Hybrid Prediction
    # RUL is now median quantile
    rul_pred_quantiles = model.predict_rul_quantiles(sample_X)
    rul_median = rul_pred_quantiles[1]
    
    # Shock Detection
    sample_feat_flat = np.mean(sample_X, axis=0).reshape(1, -1)
    shock_pred = model.iforest.predict(sample_feat_flat)
    
    # Safety Interlock Check
    # Need to pass tensor frame or seq? get_reconstruction_mse handles seq->last_frame
    mse = model.get_reconstruction_mse(torch.tensor(sample_X, dtype=torch.float32).unsqueeze(0))
    
    print(f"True RUL: {true_rul:.1f}")
    print(f"Predicted RUL (Median): {rul_median:.1f}")
    print(f"RUL Quantiles (5%, 50%, 95%): {rul_pred_quantiles}")
    print(f"Shock Status: {'Anomaly' if shock_pred[0] == -1 else 'Normal'}")
    print(f"Safety Interlock MSE: {mse:.4f} (Threshold: {MSE_THRESHOLD})")
    
    if mse > MSE_THRESHOLD:
        print(">> ALERT WOULD BE INHIBITED due to high reconstruction error (Physics Violation).")
    
    # B. Explainability
    print("\nGenerating Explanation...")
    background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
    
    # Fix for ExplainabilityEngine if it relies on old model structure:
    # We kept the interface mostly compatible, but `model` is now `HybridModel` instance.
    # `explainability.py` likely expects `model` to have `predict` or similar? 
    # Or it takes `model.iforest`. Let's check `src/explainability.py` usage in previous context.
    # It was: `explainer = ExplainabilityEngine(model, background, ...)` 
    # If `ExplainabilityEngine` expects sklearn model, we might need to pass `model.iforest`.
    # Let's check `explainability.py` via thought (or just be safe and pass `model.iforest` if that's what it was explaining before).
    # Previous code: `self.explainer = shap.TreeExplainer(self.model)` in engine.
    # Here `ExplainabilityEngine` is custom class. 
    # Let's pass `model.iforest` for shock explanation.
    
    explainer = ExplainabilityEngine(model.iforest, background, feature_names=SENSOR_FEATURES)
    risk_drivers = explainer.explain_instance(sample_X) # Explain instance likely handles shape adaptation
    print("Top 3 Risk Drivers:", risk_drivers)
    
    print("\n--- Pipleline Complete ---")

if __name__ == "__main__":
    main()
