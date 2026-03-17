import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load training data (historical)
train_csv_path = Path("data/processed/senate_trades_history.csv")
train_df = pd.read_csv(train_csv_path)
print(f"Training data: {len(train_df)} records from CSV")

# Load test data (recent 90 days)
test_json_path = Path("capitol_trades_90d.json")
with open(test_json_path, 'r') as f:
    test_raw = json.load(f)
test_df = pd.DataFrame(test_raw)
print(f"Test data: {len(test_df)} records from JSON")

# Optional: Further split training data if needed
# train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
# print(f"Validation split: {len(val_df)} records")

# Now you can use:
# - train_df for model training
# - test_df for evaluation

print("\n✅ Data loaded. Ready for ML pipeline.")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
