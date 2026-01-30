import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

class AsymmetricLoss(nn.Module):
    """
    Custom Asymmetric Loss Function.
    Penalizes 'Late Predictions' (Overestimating RUL) more heavily than Early Predictions.
    
    Late Prediction (Predicted > True): Danger (Safety Risk). High Penalty.
    Early Prediction (Predicted < True): Cost (Maintenance Risk). Lower Penalty.
    """
    def __init__(self, alpha=2.0, beta=0.5):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha # Penalty for Late (Safety critical)
        self.beta = beta   # Penalty for Early (Economic cost)

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        # If diff > 0 (Predicted > True, Late), use alpha
        # If diff < 0 (Predicted < True, Early), use beta
        loss = torch.where(diff > 0, 
                           self.alpha * (diff**2), 
                           self.beta * (diff**2))
        return torch.mean(loss)

class HybridModel:
    """
    Wraps the LSTM RUL Predictor and Isolation Forest Anomaly Detector.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        self.lstm = BiDirLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        
    def train_lstm(self, train_loader, epochs=10, lr=0.001, device='cpu'):
        self.lstm.to(device)
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        criterion = AsymmetricLoss()
        
        self.lstm.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                pred = self.lstm(X_batch)
                loss = criterion(pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def train_iforest(self, X_train_features):
        """
        Train Isolation Forest on flattened features or specific vibration features.
        X_train_features: 2D array [samples, features]
        """
        print("Training Isolation Forest for Shock Detection...")
        self.iforest.fit(X_train_features)
        
    def predict_rul(self, X_seq, device='cpu'):
        self.lstm.to(device)
        self.lstm.eval()
        with torch.no_grad():
            X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
            if X_seq.dim() == 2:
                X_seq = X_seq.unsqueeze(0)
            pred = self.lstm(X_seq)
        return pred.item()

    def detect_shock(self, X_features):
        """
        Returns -1 for Anomaly, 1 for Normal
        """
        return self.iforest.predict(X_features)

    def save(self, text_path_base='aeroguard'):
        torch.save(self.lstm.state_dict(), f"{text_path_base}_lstm.pth")
        joblib.dump(self.iforest, f"{text_path_base}_iforest.joblib")
        print("Models saved.")

class BiDirLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(BiDirLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Bidirectional outputs 2 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # RUL Regression
        )
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        # Take the output of the last time step
        # For bidirectional, we usually concat the last hidden state of fwd and bwd
        # PyTorch's output is [batch, seq, directions*hidden]
        out = out[:, -1, :] 
        out = self.fc(out)
        return out
