
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

class SensorHealthAE(nn.Module):
    """
    Denoising Autoencoder for Data Quality Index (DQI).
    Monitors sensor stream health. High reconstruction error = Sensor Drift/Failure.
    """
    def __init__(self, input_dim=15, hidden_dim=8):
        super(SensorHealthAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class HybridModel:
    """
    Wraps the Quantile LSTM RUL Predictor and Isolation Forest Anomaly Detector.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        self.lstm = BiDirQuantileLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.ae = SensorHealthAE(input_dim=input_dim) # DQI Guardrail
        
    def train_lstm(self, train_loader, epochs=10, lr=0.001, device='cpu'):
        self.lstm.to(device)
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        criterion = PinballLoss(quantiles=[0.05, 0.5, 0.95])
        
        self.lstm.train()
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

    def train_ae(self, train_loader, epochs=5, lr=0.001, device='cpu'):
        """Trains the Data Quality Autoencoder"""
        self.ae.to(device)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.ae.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, _ in train_loader:
                # Use flattened features for AE? Or sequence? 
                # Assuming X_batch is [batch, seq, feat], take last frame or flatten?
                # Let's take the last frame for DQI snapshoting
                last_frame = X_batch[:, -1, :].to(device)
                
                optimizer.zero_grad()
                recon = self.ae(last_frame)
                loss = criterion(recon, last_frame)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, AE DQI Loss: {total_loss/len(train_loader):.4f}")

    def train_iforest(self, X_train_features):
        print("Training Isolation Forest for Shock Detection...")
        self.iforest.fit(X_train_features)
        
    def predict_rul_quantiles(self, X_seq, device='cpu'):
        """
        Returns: lower (5%), median (50%), upper (95%)
        """
        self.lstm.to(device)
        self.lstm.eval()
        with torch.no_grad():
            X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
            if X_seq.dim() == 2:
                X_seq = X_seq.unsqueeze(0)
            preds = self.lstm(X_seq) # [1, 3]
            preds = preds.cpu().numpy()[0]
        return preds[0], preds[1], preds[2] # lower, median, upper

    def check_dqi_reconstruction(self, X_frame, device='cpu'):
        """
        Returns MSE error for the frame. High error = Low DQI.
        """
        self.ae.to(device)
        self.ae.eval()
        with torch.no_grad():
            x = torch.tensor(X_frame, dtype=torch.float32).to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            recon = self.ae(x)
            loss = torch.mean((x - recon)**2).item()
        return loss

    def detect_shock(self, X_features):
        return self.iforest.predict(X_features)
        
    def save(self, text_path_base='aeroguard'):
        torch.save(self.lstm.state_dict(), f"{text_path_base}_lstm.pth")
        torch.save(self.ae.state_dict(), f"{text_path_base}_ae.pth")
        joblib.dump(self.iforest, f"{text_path_base}_iforest.joblib")
        print("Models saved.")

class BiDirQuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(BiDirQuantileLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Bidirectional outputs 2 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3) # Output 3 Quantiles: 5%, 50%, 95%
        )
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out
