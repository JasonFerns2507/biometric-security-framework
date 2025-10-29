import pandas as pd
import numpy as np
import os 
import arff 
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns # Used for statistical data visualization, built on Matplotlib

# Scikit-learn (sklearn) imports for machine learning tasks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc

#  Confirmation of liac-arff import 
# To check if the 'arff' module (from liac-arff library) is successfully loaded.
# If it's not, it prints an error and exits, preventing further NameErrors.
try:
    print("liac-arff module loaded successfully.")
except NameError:
    print("Error: liac-arff module (imported as 'arff') is not defined. Please ensure it's installed and accessible.")
    exit()

# Settings
# All key project parameters are defined here for easy modification
subject_ids = list(range(1600, 1608)) # A small subset of subjects for a quick demonstration
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']
selected_features = [
    'XAVG', 'YAVG', 'ZAVG',
    'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV',
    'RESULTANT'
]
# Use of single-letter codes 'A' and 'B' for Walking and Jogging
target_activities = ['A', 'B'] 

wisdm_base_path = 'wisdm-dataset'
wisdm_arff_subfolder = os.path.join(wisdm_base_path, 'arff_files')

# Simulated Cloud Storage URL Prefix 
# This conceptual URL demonstrates how data ingestion would work in a cloud environment
CLOUD_STORAGE_PREFIX = "s3://my-biometric-bucket/wisdm-data/"

# Function to Load WISDM Data (ARFF files)
def load_wisdm_arff_file(filepath_local, sensor_type_display):
    """
    Loads a single WISDM .arff file into a pandas DataFrame using liac-arff.
    Simulates fetching from a cloud storage URL.
    """
    data = None # Initialize data to None in case loading fails
    # Construct a conceptual cloud URL to demonstrate cloud-native architecture
    cloud_url = CLOUD_STORAGE_PREFIX + filepath_local.replace(wisdm_base_path + os.sep, "")
    
    print(f"Simulating download from cloud storage: {cloud_url}")
    time.sleep(0.01) # Simulate network latency

    try:
        # Local File system path
        with open(filepath_local, 'r') as f:
            arff_data = arff.load(f)
        
        data = pd.DataFrame(arff_data['data'])
        data.columns = [attr[0].strip('"') for attr in arff_data['attributes']] 

        
        if 'ACTIVITY' in data.columns:
            
            data['ACTIVITY'] = data['ACTIVITY'].astype(str).fillna('').str.strip()
        
        # General decoding for other object-type columns
        for col in data.select_dtypes(['object']).columns:
            if col != 'ACTIVITY' and not data[col].empty and isinstance(data[col].iloc[0], bytes):
                data[col] = data[col].str.decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Local file not found at {filepath_local} (simulated from {cloud_url}). Skipping.")
        return None
    except Exception as e:
        print(f"Error loading {filepath_local}: {e}")
        traceback.print_exc()
        return None
    return data

# --- Load all ARFFs ---
all_wisdm_data = {}
print("--- Loading ARFF Data (Simulating Cloud Storage Access) ---")
# Loop through all settings to load data for each subject, device, and sensor
for subject_id in subject_ids:
    all_wisdm_data[subject_id] = {}
    for device in devices:
        all_wisdm_data[subject_id][device] = {}
        for sensor in sensors:
            filepath_local = os.path.join(wisdm_arff_subfolder, device, sensor, f'data_{subject_id}_{sensor}_{device}.arff')
            sensor_type_display = f"Subject {subject_id} {device.capitalize()} {sensor.capitalize()}"
            df = load_wisdm_arff_file(filepath_local, sensor_type_display)
            all_wisdm_data[subject_id][device][sensor] = df

# --- Process data ---
# This section structures the loaded data into a format suitable for the model
processed_biometric_data = []
print("\n--- Data Structuring (Local Processing) ---")
for subject_id in all_wisdm_data.keys():
    for device in all_wisdm_data[subject_id].keys():
        for sensor in all_wisdm_data[subject_id][device].keys():
            df = all_wisdm_data[subject_id][device][sensor]
            if df is not None:
                # Identify features that are present in the current DataFrame
                usable_feats = [feat for feat in selected_features if feat in df.columns]
                if not usable_feats:
                    continue # Skip if no usable features are found
                if 'ACTIVITY' in df.columns:
                    # Filter the data to include only the target activities
                    df_filtered = df[df['ACTIVITY'].isin(target_activities)].copy()
                    if not df_filtered.empty:
                        # Aggregate features for each subject-activity pair
                        grouped_data = df_filtered.groupby(['class', 'ACTIVITY'])[usable_feats].mean().reset_index()
                        # Rename columns to show the sensor source
                        new_columns = {col: f"{device}_{sensor}_{col}" for col in usable_feats}
                        grouped_data = grouped_data.rename(columns=new_columns)
                        processed_biometric_data.append(grouped_data)
            else:
                print(f"Warning: No DataFrame for {subject_id} {device} {sensor}")

if not processed_biometric_data:
    print("No data processed! Exiting.")
    exit()

# Assemble per (subject, activity)
# This section combines all the processed data into a single, comprehensive DataFrame
subject_activity_keys = set()
for df in processed_biometric_data:
    for _, row in df.iterrows():
        subject_activity_keys.add((row['class'], row['ACTIVITY']))

rows = []
for subj, act in subject_activity_keys:
    row_dict = {'class': subj, 'ACTIVITY': act}
    for df in processed_biometric_data:
        match_row = df[(df['class'] == subj) & (df['ACTIVITY'] == act)]
        if not match_row.empty:
            for col in match_row.columns:
                if col not in ['class', 'ACTIVITY']:
                    row_dict[col] = match_row.iloc[0][col]
    rows.append(row_dict)

final_biometric_df = pd.DataFrame(rows)
print(f"Combined shape before NaN handling: {final_biometric_df.shape}")

# --- Handle Missing Data ---
# This section cleans up any remaining NaNs in the combined DataFrame
test_cols = [c for c in final_biometric_df.columns if c not in ('class', 'ACTIVITY')]
final_biometric_df['missing_count'] = final_biometric_df[test_cols].isnull().sum(axis=1)
print(final_biometric_df['missing_count'].value_counts().sort_index())
feature_cols = [c for c in final_biometric_df.columns if c not in ('class', 'ACTIVITY', 'missing_count')]
min_features = max(1, int(len(feature_cols) * 0.5))
final_biometric_df = final_biometric_df.drop('missing_count', axis=1)
final_biometric_df = final_biometric_df[final_biometric_df[feature_cols].count(axis=1) >= min_features]

print(f"Shape after flexible NaN drop: {final_biometric_df.shape}")

if final_biometric_df.empty:
    print("All rows removed after NaN check. Try adding more subjects or checking file completeness.")
    exit()

# Prepare datasets
X = final_biometric_df[feature_cols].fillna(0)
y = final_biometric_df['class']

print(f"Final Feature Matrix (X) Shape: {X.shape}")
print(f"Final Target Vector (y) Shape: {y.shape}")
print(f"Unique Subjects: {y.nunique()}\n")

# Split train/test (stratified, test_size=0.5 allows one per class)
# This splits the data at the subject level for a realistic evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test: X={X_test.shape}, y={y_test.shape}")

# Model training
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
print("\nModel trained.")

# Score calculation 
# This section generates genuine and impostor scores for the model
genuine_scores = []
impostor_scores = []
y_pred_proba = knn_model.predict_proba(X_test)

class_to_idx = {cls: idx for idx, cls in enumerate(knn_model.classes_)}


for i, true_subject_id in enumerate(y_test):
    idx_true = class_to_idx.get(true_subject_id, None)
    if idx_true is not None:
        genuine_scores.append(y_pred_proba[i, idx_true])
    for impostor_id in knn_model.classes_:
        if impostor_id != true_subject_id:
            idx_imp = class_to_idx[impostor_id]
            impostor_scores.append(y_pred_proba[i, idx_imp])

print(f"\nGenerated {len(genuine_scores)} genuine and {len(impostor_scores)} impostor scores.")

if len(genuine_scores) == 0:
    print("No genuine scores found! Check your train/test split strategy.")
    exit()

# Normalize and evaluate
all_scores = np.array(genuine_scores + impostor_scores).reshape(-1, 1)
all_labels = np.array([1]*len(genuine_scores) + [0]*len(impostor_scores))
scaler = MinMaxScaler().fit(all_scores)
norm_scores = scaler.transform(all_scores).flatten()

fpr, tpr, thresholds = roc_curve(all_labels, norm_scores)
roc_auc = auc(fpr, tpr)
frr = 1 - tpr
eer_value = None
min_diff = float('inf')
for i, threshold in enumerate(thresholds):
    diff = abs(fpr[i] - frr[i])
    if diff < min_diff:
        min_diff = diff
        eer_threshold = threshold
        eer_value = (fpr[i] + frr[i])/2

print(f"AUC: {roc_auc:.4f}, EER: {eer_value:.4f}")

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'--',c='gray')
plt.scatter(eer_value, 1-eer_value, color='red',label=f"EER={eer_value:.2f}")
plt.xlabel("FAR")
plt.ylabel("TAR (1-FRR)")
plt.legend()
plt.grid()
plt.title("ROC Curve")
plt.show()

# AI Agent Simulation 
AUTHENTICATE_THRESHOLD = 0.7
CHALLENGE_THRESHOLD = 0.4

# Simulated Cloud API Endpoint for AI Agent 
# In a real cloud deployment, this would be an actual HTTP endpoint

CLOUD_AI_AGENT_API_ENDPOINT = "https://api.mybiometrics.cloud/agent/decision"

def call_cloud_ai_agent_api(probe_features, claimed_subject_id, model, scaler, class_to_idx, auth_thresh, challenge_thresh):
    """
    Simulates an API call to a cloud-hosted AI Agent for a decision.
    """
    print(f"Simulating API call to: {CLOUD_AI_AGENT_API_ENDPOINT}")
    print(f"Request payload: {{'probe_features': ..., 'claimed_id': {claimed_subject_id}}}")
    time.sleep(0.05) # Simulate network latency and processing time in the cloud

    # The actual decision logic runs here, as if it were on the cloud function
    decision, action, normalized_score, predicted_id = ai_agent_decision(
        probe_features, claimed_subject_id, model, scaler, class_to_idx, auth_thresh, challenge_thresh
    )
    print(f"API response received: {{'decision': '{decision}', 'action': '{action}'}}")
    return decision, action, normalized_score, predicted_id


def ai_agent_decision(probe_features, claimed_subject_id, model, scaler, class_to_idx, auth_thresh, challenge_thresh):
    """
    The core AI Agent decision logic (would be executed on the cloud function).
    """
    arr = probe_features.values.reshape(1, -1)
    proba = model.predict_proba(arr)[0]
    predicted_subject_id = model.classes_[np.argmax(proba)]
    score_for_claimed = proba[class_to_idx[claimed_subject_id]] if claimed_subject_id in class_to_idx else 0.0
    normalized_score = scaler.transform(np.array([[score_for_claimed]]))[0][0]
    if normalized_score >= auth_thresh:
        if predicted_subject_id == claimed_subject_id:
            return "Authenticated", "Allow Access", normalized_score, predicted_subject_id
        else:
            return "Challenged (Mismatch in Top Prediction)", "Require Secondary Auth", normalized_score, predicted_subject_id
    elif normalized_score >= challenge_thresh:
        return "Challenged", "Require Secondary Auth", normalized_score, predicted_subject_id
    else:
        return "Denied", "Deny Access & Log", normalized_score, predicted_subject_id

simulated_results = []
np.random.seed(42)
num_simulations = min(5, len(X_test))
sim_indices = np.random.choice(X_test.index, num_simulations, replace=False)
for idx in sim_indices:
    probe_features = X_test.loc[idx]
    true_subject_id = y_test.loc[idx]
    claimed_id_genuine = true_subject_id
    
    #Call the simulated cloud API for genuine attempt
    out = call_cloud_ai_agent_api(probe_features, claimed_id_genuine, knn_model, scaler, class_to_idx,
                                AUTHENTICATE_THRESHOLD, CHALLENGE_THRESHOLD)
    simulated_results.append(dict(Scenario='Genuine', True_Subject=true_subject_id,
            Claimed_ID=claimed_id_genuine, Predicted_ID=out[3],
            Normalized_Score=f"{out[2]:.4f}", Decision=out[0], Action=out[1]))
    
    impostors = [x for x in y_train.unique() if x != true_subject_id]
    if impostors:
        claimed_id_impostor = np.random.choice(impostors)
        # Call the simulated cloud API for impostor attempt 
        out = call_cloud_ai_agent_api(probe_features, claimed_id_impostor, knn_model, scaler, class_to_idx,
                                AUTHENTICATE_THRESHOLD, CHALLENGE_THRESHOLD)
        simulated_results.append(dict(Scenario='Impostor', True_Subject=true_subject_id,
            Claimed_ID=claimed_id_impostor, Predicted_ID=out[3],
            Normalized_Score=f"{out[2]:.4f}", Decision=out[0], Action=out[1]))
    else:
        simulated_results.append({'Scenario':'Impostor', 'True_Subject':true_subject_id, 'Claimed_ID':'N/A',
                'Predicted_ID':'N/A', 'Normalized_Score':'N/A', 'Decision':'N/A', 'Action':'Not enough impostors'})

print(pd.DataFrame(simulated_results).to_string(index=False))
