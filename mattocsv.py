import scipy.io
import pandas as pd
import os
import glob
import numpy as np

# Directory containing .mat files
mat_folder = "Tail_666_9"
output_folder = "csv_output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all .mat files in the folder
mat_files = glob.glob(os.path.join(mat_folder, "*.mat"))

print(f"Found {len(mat_files)} .mat files to convert")

# Metadata keys to skip
skip_keys = ['__header__', '__version__', '__globals__']

success_count = 0
error_count = 0

def extract_data(value):
    """Extract data from various .mat file formats"""
    if isinstance(value, np.ndarray):
        # Check if it's a structured array with 'data' field
        if value.dtype.names and 'data' in value.dtype.names:
            # Extract the 'data' field from the structured array
            try:
                data_field = value[0, 0]['data']
                if isinstance(data_field, np.ndarray):
                    return data_field.flatten()
            except (IndexError, TypeError):
                pass
        # Regular array - just flatten it
        return value.flatten()
    return None

for mat_file in mat_files:
    try:
        # Load .mat file
        mat = scipy.io.loadmat(mat_file)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(mat_file))[0]
        
        # Create a dictionary for all variables
        data_dict = {}
        max_len = 0
        
        # Extract all data variables
        for key in mat.keys():
            if key not in skip_keys:
                value = mat[key]
                flat = extract_data(value)
                
                if flat is not None and len(flat) > 0:
                    # Convert to numeric if possible
                    try:
                        flat = flat.astype(float)
                    except (ValueError, TypeError):
                        try:
                            flat = flat.astype(str)
                        except:
                            continue
                    
                    data_dict[key] = flat
                    max_len = max(max_len, len(flat))
        
        if not data_dict:
            print(f"Skipped (no data): {mat_file}")
            continue
        
        # Pad shorter arrays to make them equal length
        for key in data_dict:
            current_len = len(data_dict[key])
            if current_len < max_len:
                # Pad with empty string for string arrays, NaN for numeric
                if data_dict[key].dtype.kind in ['U', 'S', 'O']:
                    data_dict[key] = np.pad(data_dict[key], (0, max_len - current_len), 
                                            mode='constant', constant_values='')
                else:
                    data_dict[key] = np.pad(data_dict[key].astype(float), 
                                            (0, max_len - current_len), 
                                            constant_values=np.nan)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data_dict)
        output_path = os.path.join(output_folder, f"{base_name}.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Converted: {mat_file} -> {output_path}")
        success_count += 1
        
    except Exception as e:
        print(f"Error processing {mat_file}: {e}")
        error_count += 1

print(f"\nConversion complete!")
print(f"Successfully converted: {success_count} files")
print(f"Errors: {error_count} files")
print(f"CSV files saved to '{output_folder}' folder")
