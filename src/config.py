
# Configuration for AeroGuard
import os

# Dataset
DATA_PATH = "data/synthetic_dashlink.csv"
SEQUENCE_LENGTH = 50  # Sliding window size
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Artifacts
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Features
# Sensor features to be used for model input
SENSOR_FEATURES = [
    'LATP', 'LONP', 'ALT', 'VEL', 'TH', 'N1', 'N2', 'EGT', 'FF', 
    'VIB', 'VRTG', 'OIL_P', 'OIL_T', 'FLAP', 'HYDY'
]

# Phase III Additions
DQI_THRESHOLD = 0.7
CONFIDENCE_LEVEL = 0.95
MSE_THRESHOLD = 5.0
import os
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "ledger.json")
HARD_LIMIT_EGT = 900.0 # Standard OEM threshold for EGT alerts
# Features to exclude from training but keep for context/phases
CONTEXT_FEATURES = ['FLIGHT_PHASE', 'RUL', 'CYCLE', 'FLIGHT_ID']

# Training
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Model
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
