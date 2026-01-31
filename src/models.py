
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

class PinballLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for Deterministic Uncertainty.
    Calculates loss for multiple quantiles simultaneously.
    """
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds: [batch, num_quantiles]
        # target: [batch, 1] OR [batch]
        if target.dim() == 1:
            target = target.unsqueeze(1)
            
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i:i+1]
            loss = torch.max((q - 1) * error, q * error)
            total_loss += torch.mean(loss)
            
        return total_loss

class BiDirQuantileLSTM(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2, dropout=0.2):
        super(BiDirQuantileLSTM, self).__init__()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(n_hidden * 2, 3) # 5th, 50th, 95th quantiles

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SimpleAutoencoder(nn.Module):
    def __init__(self, n_features):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, n_features)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class HybridModel:
    def __init__(self, n_features, iforest=None, scaler=None):
        self.iforest = iforest
        self.scaler = scaler
        self.lstm = BiDirQuantileLSTM(n_features)
        self.ae = SimpleAutoencoder(n_features)
        
    def train_ae(self, loader, epochs=10, device='cpu'):
        """Trains the Safety Interlock Autoencoder."""
        self.ae.to(device)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.ae.train()
        print("Training Gate 3 Safety Interlock (Autoencoder)...")
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                
                # For AE, we can use the last frame of the sequence or flatten?
                # The user spec shows `batch_x` being passed directly.
                # If batch_x is [batch, seq, feat], we typically use the last frame for a "snapshot" AE 
                # OR we treat every frame as an independent sample.
                # Given user code: `output = self.ae(batch_x)`, and `ae` is Linear, 
                # we probably want to operate on the feature vector.
                # Let's assume we train on the last time step for RUL context, or flattened if that was the intent.
                # However, AE is usually per-frame.
                # Let's take the last frame (most recent sensor reading) to validate "current state".
                
                if batch_x.dim() == 3:
                   x_input = batch_x[:, -1, :]
                else:
                   x_input = batch_x
                   
                optimizer.zero_grad()
                output = self.ae(x_input)
                loss = criterion(output, x_input)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, AE Loss: {total_loss/len(loader):.4f}")

    def train_lstm(self, train_loader, epochs=10, lr=0.001, device='cpu'):
        self.lstm.to(device)
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        criterion = PinballLoss(quantiles=[0.05, 0.5, 0.95])
        
        self.lstm.train()
        print("Training RUL Predictor (LSTM)...")
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                preds = self.lstm(X_batch) # [batch, 3]
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Quantile Loss: {total_loss/len(train_loader):.4f}")

    def get_reconstruction_mse(self, X_tensor):
        """Calculates MSE for Gate 3 validation."""
        self.ae.eval()
        with torch.no_grad():
            if X_tensor.dim() == 2: # Frame [1, feat]
                 # OK
                 pass
            elif X_tensor.dim() == 3: # Seq [1, seq, feat] -> Take last
                 X_tensor = X_tensor[:, -1, :]
                 
            reconstruction = self.ae(X_tensor)
            return torch.mean((X_tensor - reconstruction)**2).item()

    def predict_rul(self, X_tensor):
        """Returns the median (50th percentile) RUL."""
        quantiles = self.predict_rul_quantiles(X_tensor)
        return quantiles[1]

        """Returns [5%, 50%, 95%] RUL quantiles."""
        self.lstm.eval()
        with torch.no_grad():
            if not isinstance(X_tensor, torch.Tensor):
                X_tensor = torch.tensor(X_tensor, dtype=torch.float32)

            if X_tensor.ndim == 2: 
                X_tensor = X_tensor.unsqueeze(0)
                
            return self.lstm(X_tensor).numpy()[0]
            
    def save(self, text_path_base='aeroguard'):
        torch.save(self.lstm.state_dict(), f"{text_path_base}_lstm.pth")
        torch.save(self.ae.state_dict(), f"{text_path_base}_ae.pth")
        if self.iforest:
             joblib.dump(self.iforest, f"{text_path_base}_iforest.joblib")
             if self.scaler:
                 joblib.dump(self.scaler, f"{text_path_base}_scaler.joblib") # Ensure scaler is paired
        print("Models saved.")
