import pandas as pd
import numpy as np
import os 
from scipy.io import arff # Import arff for loading .arff files

# Define lists for device types, sensor types, and subject IDs
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']
subject_ids = list(range(1600, 1650)) # Subject IDs range from 1600 to 1649 (50 subjects)

# WISDM Dataset paths (copied from data_loader.py)
wisdm_base_path = 'wisdm-dataset' 
wisdm_arff_subfolder = os.path.join(wisdm_base_path, 'arff_files')

# Function to Load WISDM Data (ARFF files) - copied from data_loader.py
def load_wisdm_arff_file(filepath, sensor_type_display):
    """
    Loads a single WISDM .arff file into a pandas DataFrame.
    Handles decoding of byte strings to regular strings for categorical columns.
    Also strips leading/trailing double quotes from column names.
    """
    data = None
    try:
        data_array, meta = arff.loadarff(filepath)
        data = pd.DataFrame(data_array)
        
        
        data.columns = [col.strip('"') for col in data.columns]
        

        for col in data.select_dtypes(['object']).columns:
            try:
                data[col] = data[col].str.decode('utf-8')
            except AttributeError:
                pass 
        
        # print(f"Successfully loaded {sensor_type_display} data from: {filepath}") 
        # print(f"{sensor_type_display} Data Shape: {data.shape}\n") 
    except FileNotFoundError:
        print(f"Error: {sensor_type_display} file not found at {filepath}. Please check the path and filename.")
    except Exception as e:
        print(f"An error occurred while loading {sensor_type_display} data from {filepath}: {e}\n")
    return data

# This dictionary will store all loaded DataFrames
all_wisdm_data = {}

print("--- Re-loading WISDM Data for Preprocessing ---")
for subject_id in subject_ids:
    all_wisdm_data[subject_id] = {}
    for device in devices:
        all_wisdm_data[subject_id][device] = {}
        for sensor in sensors:
            filepath = os.path.join(wisdm_arff_subfolder, device, sensor, f'data_{subject_id}_{sensor}_{device}.arff')
            sensor_type_display = f'WISDM Subject {subject_id} {device.capitalize()} {sensor.capitalize()}'
            df = load_wisdm_arff_file(filepath, sensor_type_display)
            all_wisdm_data[subject_id][device][sensor] = df
print("--- WISDM Data Re-loaded for Preprocessing. ---")

print("--- Starting Data Preprocessing and Structuring ---")

# Definition of features to be used.
# The WISDM dataset has 93 features. 
# A good starting point might be average values, standard deviations, and resultant.
selected_features = [
    'XAVG', 'YAVG', 'ZAVG',          # Average sensor value per axis
    'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', # Standard deviation per axis
    'RESULTANT'                      # Average resultant value
    #More Features can be added later on
]

# Definition of activities to perform authentication on.
# 'A' is Walking, 'B' is Jogging
target_activities = ['A', 'B'] 

# 1. Data Cleaning/Preparation (Brief Check) 
# We'll just check one DataFrame as an example.
# In project scope, all the data frames are loaded
print("\n--- Checking for Missing Values (Example: Subject 1600 Phone Accel) ---")
# Ensure the example_df is accessed only if subject 1600 data exists
if 1600 in all_wisdm_data and 'phone' in all_wisdm_data[1600] and 'accel' in all_wisdm_data[1600]['phone']:
    example_df = all_wisdm_data[1600]['phone']['accel']
    # Ensure selected_features are present in the example_df before checking
    # This check is now robust because column names are stripped of quotes
    if example_df is not None and all(feature in example_df.columns for feature in selected_features): # Added check for example_df is not None
        print(example_df[selected_features].isnull().sum())
    else:
        print("Selected features not found in example_df or example_df is None. Check feature names or data loading.")
        if example_df is not None:
            print("Available columns:", example_df.columns.tolist())
else:
    print("Subject 1600 Phone Accel data not available for missing value check.")

# 2. Data Structuring for Biometrics
# Goal: Create a unified DataFrame where each row represents a subject's Biometric profile for a specific activity, combining features from all sensors.
# Columns will be: 'subject_id', 'activity_code', and all selected_features (prefixed by sensor/device)

processed_biometric_data = []

# Iterate through all subjects, devices, and sensors
for subject_id in all_wisdm_data.keys():
    for device in all_wisdm_data[subject_id].keys():
        for sensor in all_wisdm_data[subject_id][device].keys():
            df = all_wisdm_data[subject_id][device][sensor]
            
           
            if df is not None:
                # Filter for the target activities
                df_filtered = df[df['ACTIVITY'].isin(target_activities)].copy()
                
                if not df_filtered.empty:
                    # Group by activity and subject to get one feature vector per subject-activity pair
                    # Calculate the mean of the selected features for each group
                    grouped_data = df_filtered.groupby(['class', 'ACTIVITY'])[selected_features].mean().reset_index()
                    
                    # Rename columns to indicate source (e.g., 'phone_accel_XAVG')
                    # This helps in distinguishing features from different sensors after concatenation
                    new_columns = {col: f"{device}_{sensor}_{col}" for col in selected_features}
                    grouped_data = grouped_data.rename(columns=new_columns)
                    
                    # Append to our list
                    processed_biometric_data.append(grouped_data)
            # --- End FIX ---

# Concatenate all processed data into a single DataFrame
if processed_biometric_data:
    final_biometric_df = pd.concat(processed_biometric_data, ignore_index=True)
    print("\n--- Final Processed Biometric Data (Head) ---")
    print(final_biometric_df.head())
    print(f"\nFinal Processed Biometric Data Shape: {final_biometric_df.shape}")
    print(f"Final Processed Biometric Data Columns: {final_biometric_df.columns.tolist()}")

    # 3. Separate Features (X) and Target (y)
    # X will be our feature matrix, y will be the subject_id
    X = final_biometric_df.drop(columns=['class', 'ACTIVITY']) # Features
    y = final_biometric_df['class'] # Target (subject_id)

    print(f"\nShape of Feature Matrix (X): {X.shape}")
    print(f"Shape of Target Vector (y): {y.shape}")
    print(f"Unique Subjects in Data (y): {y.nunique()}")

else:
    print("\nNo data was processed. Check your data loading and filtering criteria.")

print("\n--- Data Preprocessing and Structuring Complete ---")

# The 'X' and 'y' DataFrames are now ready for training and testing your biometric model.
