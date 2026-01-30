
import pandas as pd
import glob
import os
import numpy as np

def profile_flights(data_dir="csv_output"):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    metrics = []
    
    print(f"Profiling {len(files)} flights (sampling first 100 for speed)...")
    
    # Sample 100 files to get a representative distribution
    sample_files = files[:100] 
    
    for f in sample_files:
        try:
            # Read minimal columns to check profile
            df = pd.read_csv(f, usecols=['ALT', 'LATP', 'LONP', 'N1_1'])
            
            duration = len(df) # 1 sec per frame usually
            max_alt = df['ALT'].max()
            
            metrics.append({
                "file": f,
                "duration_frames": duration,
                "max_altitude": max_alt,
                "type": "High Altitude" if max_alt > 28000 else "Low Altitude"
            })
        except Exception as e:
            pass
            
    df_metrics = pd.DataFrame(metrics)
    
    # Define Long Haul vs Short Haul
    # Assuming Short < 1 hour (3600 frames), Long > 1 hour
    # Actually DASHlink clips might be shorter. Let's look at distribution
    
    median_dur = df_metrics['duration_frames'].median()
    df_metrics['category'] = np.where(df_metrics['duration_frames'] > median_dur, "Long Haul", "Short Haul")
    
    print("\n--- Flight Profile Summary ---")
    print(df_metrics['category'].value_counts())
    print(f"Median Duration: {median_dur} frames")
    
    # Select Stratified Sample (e.g., 15 Long, 15 Short)
    long_haul = df_metrics[df_metrics['category'] == "Long Haul"].head(15)
    short_haul = df_metrics[df_metrics['category'] == "Short Haul"].head(15)
    
    stratified = pd.concat([long_haul, short_haul])
    
    print(f"\nSelected {len(stratified)} stratified files for robust training.")
    print(stratified[['file', 'duration_frames', 'category']].head())
    
    return stratified['file'].tolist()

if __name__ == "__main__":
    profile_flights()
