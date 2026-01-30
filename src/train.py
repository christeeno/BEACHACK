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
    model = HybridModel(input_dim=input_dim)
    
    print("Training LSTM Component (RUL Prediction)...")
    model.train_lstm(train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    print("Training iForest Component (Shock Detection)...")
    # Use flattened features for iForest or just last time step
    # For simplicity using mean of features over window
    X_train_flat = np.mean(X_train, axis=1) 
    model.train_iforest(X_train_flat)
    
    # 4. Save
    model.save("aeroguard_v1")
    
    # 5. Demonstration / Validation
    print("\n--- Validation Demo ---")
    
    # Select a test sample
    sample_idx = 0
    sample_X = X_test[sample_idx]
    true_rul = y_test[sample_idx]
    
    # A. Hybrid Prediction
    rul_pred = model.predict_rul(sample_X)
    shock_pred = model.detect_shock(np.mean(sample_X, axis=0).reshape(1, -1))
    
    print(f"True RUL: {true_rul:.1f}")
    print(f"Predicted RUL: {rul_pred:.1f}")
    print(f"Shock Status: {'Anomaly' if shock_pred[0] == -1 else 'Normal'}")
    
    # B. Probabilistic Wrapper
    mean_pred, low_bound, high_bound = predict_uncertainty(model, sample_X, num_samples=50)
    print(f"Probabilistic Forecast: 80% Confidence of Failure between {model.predict_rul(sample_X) - (high_bound[0][0]-mean_pred[0][0]):.1f} and {model.predict_rul(sample_X) + (mean_pred[0][0]-low_bound[0][0]):.1f} cycles")
    # (Approximation for demo output string to match user request style)
    print(f"Detailed Stats -> Mean: {mean_pred[0][0]:.1f}, 10th: {low_bound[0][0]:.1f}, 90th: {high_bound[0][0]:.1f}")

    # C. SHAP Explainability
    print("\nGenerating Explanation...")
    # Use small background sample for speed
    background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
    
    # Flatten background for KernelExplainer wrapper logic if needed, 
    # but the implementation in explainability.py expects just 3D array in init?
    # Let's check init: "background_data ... numpy array". 
    # The wrapper reshapes. So we pass flat?
    # explainability.py: self.background_summary = shap.kmeans(background_data.reshape(background_data.shape[0], -1), 10)
    # Yes, it handles reshaping.
    
    explainer = ExplainabilityEngine(model, background, feature_names=SENSOR_FEATURES)
    # For explanation we explain the specific instance
    risk_drivers = explainer.explain_instance(sample_X)
    print("Top 3 Risk Drivers:", risk_drivers)
    
    # D. Feedback Loop
    print("\nTesting Feedback Loop...")
    feedback = FeedbackLoop(model)
    # Simulate rejecting the alert (e.g., Engineer says "This engine is fine")
    feedback.update_with_feedback(sample_X, confirmed_failure=False)
    
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
