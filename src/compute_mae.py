import numpy as np, torch
from src.config import DATA_PATH, TEST_SIZE
from src.utils import load_data, prepare_sequences
from src.preprocessing import process_dataframe
from src.models import HybridModel

# Load and preprocess data
df = load_data(DATA_PATH)
df = process_dataframe(df)
X, y, _ = prepare_sequences(df)
split = int(len(X)*(1-TEST_SIZE))
# Use a subset for quick evaluation
X_test = X[split:][:200]
y_test = y[split:][:200]

# Load model
model = HybridModel(input_dim=X.shape[2])
model.lstm.load_state_dict(torch.load('aeroguard_v1_lstm.pth'))
model.lstm.eval()

# Predict RUL for test subset
preds = []
with torch.no_grad():
    for seq in X_test:
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        pred = model.lstm(seq_tensor).item()
        preds.append(pred)

preds = np.array(preds)
mae = np.mean(np.abs(preds - y_test))
print('Test MAE (sample 200):', mae)
