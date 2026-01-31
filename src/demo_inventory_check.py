
import json
import os
import pandas as pd
from src.config import SENSOR_FEATURES

def verify_inventory():
    path = os.path.join("src", "inventory.json")
    with open(path, 'r') as f:
        inv = json.load(f)
    
    print(f"--- AeroGuard Inventory Database ({len(inv)} Components) ---")
    data = []
    for sensor, info in inv.items():
        data.append({
            "Sensor": sensor,
            "Part Name": info.get("part_number"),
            "ATA Task Card": info.get("task_card_id"),
            "Stock": info.get("stock_quantity"),
            "Lead Time": f"{info.get('lead_time_days')} Days"
        })
        
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Check coverage
    missing = [s for s in SENSOR_FEATURES if s not in inv]
    if missing:
        print(f"\n[WARNING] Missing mapping for sensors: {missing}")
    else:
        print("\n[SUCCESS] 100% Sensor Coverage in Logistics Database.")

if __name__ == "__main__":
    verify_inventory()
