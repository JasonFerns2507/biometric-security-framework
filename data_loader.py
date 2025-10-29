import pandas as pd #DataFrames
import numpy as np #for numbers
import os #for file paths
from scipy.io import arff # For loading .arff files

print("--- Starting WISDM Data Loading ---")

# WISDM Dataset (Accelerometer & Gyroscope - .arff files)
# 'wisdm-dataset' is directly inside 'BiometricAuthProject'
wisdm_base_path = 'wisdm-dataset' 
wisdm_arff_subfolder = os.path.join(wisdm_base_path, 'arff_files')

# Define lists for device types, sensor types, and subject IDs
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']
# Subject IDs range from 1600 to 1649 (50 subjects)
subject_ids = list(range(1600, 1650)) 


# --- Function to Load WISDM Data (ARFF files) ---
def load_wisdm_arff_file(filepath, sensor_type_display):
    """
    Loads a single WISDM .arff file into a pandas DataFrame.
    Handles decoding of byte strings to regular strings for categorical columns.
    """
    data = None
    try:
        # loadarff returns a tuple: (data_array, metadata)
        data_array, meta = arff.loadarff(filepath)
        
        # Convert the structured numpy array to a pandas DataFrame
        data = pd.DataFrame(data_array)

        # Decode byte strings to regular strings for object-type columns (like 'ACTIVITY')
        for col in data.select_dtypes(['object']).columns:
            try:
                data[col] = data[col].str.decode('utf-8')
            except AttributeError: # Handle cases where it's not a byte string but an object
                pass 
        
        print(f"Successfully loaded {sensor_type_display} data from: {filepath}")
        # print(f"{sensor_type_display} Data Head:") 
        # print(data.head())
        print(f"{sensor_type_display} Data Shape: {data.shape}\n")
        # print(f"{sensor_type_display} Columns: {data.columns.tolist()}\n") 
    except FileNotFoundError:
        print(f"Error: {sensor_type_display} file not found at {filepath}. Please check the path and filename.")
    except Exception as e:
        print(f"An error occurred while loading {sensor_type_display} data from {filepath}: {e}\n")
    return data

# --- Load WISDM Data for all subjects and all sensor types ---
# This dictionary will store all loaded DataFrames
# Structure: all_wisdm_data[subject_id][device][sensor_type] = DataFrame
all_wisdm_data = {}

for subject_id in subject_ids:
    all_wisdm_data[subject_id] = {}
    for device in devices:
        all_wisdm_data[subject_id][device] = {}
        for sensor in sensors:
            # Construct the full file path for the current subject, device, and sensor
            filepath = os.path.join(wisdm_arff_subfolder, device, sensor, f'data_{subject_id}_{sensor}_{device}.arff')
            
            sensor_type_display = f'WISDM Subject {subject_id} {device.capitalize()} {sensor.capitalize()}'
            
            # Load the data using the defined function
            df = load_wisdm_arff_file(filepath, sensor_type_display)
            
            # Store the loaded DataFrame
            all_wisdm_data[subject_id][device][sensor] = df

print("\n--- All WISDM Data Loading Attempts Complete ---")

# The 'all_wisdm_data' dictionary now contains all loaded DataFrames.
# Example access:
# To get phone accelerometer data for subject 1600:
# subject_1600_phone_accel_df = all_wisdm_data[1600]['phone']['accel']
# print(f"Example: Subject 1600 Phone Accel Data Shape: {subject_1600_phone_accel_df.shape}")
