import torch
import numpy as np

def predict_uncertainty(model, X_seq, num_samples=50, device='cpu'):
    """
    Performs Monte Carlo Dropout to estimate predictive uncertainty.
    
    Args:
        model: Trained PyTorch LSTM model.
        X_seq: Input sequence(s) [batch, seq_len, features].
        num_samples: Number of MC forward passes.
        
    Returns:
        mean_pred: Average prediction (RUL).
        lower_bound: 10th percentile (Conservative safety estimate).
        upper_bound: 90th percentile.
    """
    model.lstm.to(device)
    # Enable Dropout during inference
    model.lstm.train() 
    
    predictions = []
    
    X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
    if X_seq.dim() == 2:
        X_seq = X_seq.unsqueeze(0)
        
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model.lstm(X_seq)
            predictions.append(pred.cpu().numpy())
            
    predictions = np.array(predictions) # [num_samples, batch, 1]
    
    # Calculate statistics across the 'num_samples' dim
    mean_pred = np.mean(predictions, axis=0) # [batch, 1]
    std_pred = np.std(predictions, axis=0)
    
    # 80% Confidence Interval (10th to 90th percentile)
    lower_bound = np.percentile(predictions, 10, axis=0)
    upper_bound = np.percentile(predictions, 90, axis=0)
    
    return mean_pred, lower_bound, upper_bound
