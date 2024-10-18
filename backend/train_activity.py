import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Paths to directories for healthy individuals and epileptic patients
healthy_individuals_dir = 'Healthy Individuals'
epileptic_patients_dir = 'Epileptic Patients'

# Function to safely load CSV data and handle empty or invalid files
def safe_load_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.iloc[:, :1]  # Drop any extra columns, keep the first column (EEG data)
        if df.empty:
            print(f"Skipping empty file: {file_path}")
            return None
        return df
    except pd.errors.EmptyDataError:
        print(f"Skipping empty or invalid file: {file_path}")
        return None

# Load data from healthy individuals and epileptic patients
healthy_individuals_files = [f for f in os.listdir(healthy_individuals_dir) if f.endswith('.csv')]
epileptic_patients_files = [f for f in os.listdir(epileptic_patients_dir) if f.endswith('.csv')]

healthy_individuals_data = [safe_load_csv(os.path.join(healthy_individuals_dir, f)) for f in healthy_individuals_files if safe_load_csv(os.path.join(healthy_individuals_dir, f)) is not None]
epileptic_patients_data = [safe_load_csv(os.path.join(epileptic_patients_dir, f)) for f in epileptic_patients_files if safe_load_csv(os.path.join(epileptic_patients_dir, f)) is not None]

# Check if data was loaded successfully
if not healthy_individuals_data or not epileptic_patients_data:
    print("Error: No valid data loaded from CSV files. Exiting...")
    exit(1)

# Ensure all files have consistent dimensions by only keeping the first column
healthy_individuals_data = [data.values.flatten() for data in healthy_individuals_data]
epileptic_patients_data = [data.values.flatten() for data in epileptic_patients_data]

# Function to extract features from EEG data
def extract_features(data):
    features = []
    # Statistical features
    features.append(np.mean(data))
    features.append(np.std(data))
    features.append(np.min(data))
    features.append(np.max(data))
    features.append(np.median(data))
    features.append(skew(data))
    features.append(kurtosis(data))
    
    # Frequency domain features using FFT
    fft_values = fft(data)
    fft_magnitude = np.abs(fft_values)
    features.append(np.mean(fft_magnitude))
    features.append(np.std(fft_magnitude))
    
    # Band power features (Delta, Theta, Alpha, Beta, Gamma)
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}
    for band, (low, high) in bands.items():
        band_power = np.mean(fft_magnitude[(np.fft.fftfreq(len(data), 1/512) >= low) & 
                                            (np.fft.fftfreq(len(data), 1/512) <= high)])
        features.append(band_power)

    return features

# Extract features from data
healthy_individuals_features = np.array([extract_features(data) for data in healthy_individuals_data])
epileptic_patients_features = np.array([extract_features(data) for data in epileptic_patients_data])

# Preprocess the data
scaler = StandardScaler()
healthy_individuals_features_scaled = scaler.fit_transform(healthy_individuals_features)
epileptic_patients_features_scaled = scaler.transform(epileptic_patients_features)

# Save the scaler for later use in preprocessing during analysis
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Concatenate the data from both groups
X = np.concatenate((healthy_individuals_features_scaled, epileptic_patients_features_scaled), axis=0)
y_epilepsy = np.concatenate((np.zeros(len(healthy_individuals_features_scaled)), np.ones(len(epileptic_patients_features_scaled))))

### 1. **Epilepsy Diagnosis** using Random Forest ###
# Split the data into train and test sets for epilepsy diagnosis
X_train, X_test, y_train, y_test = train_test_split(X, y_epilepsy, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest (Epilepsy Diagnosis)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
rf_classifier_epilepsy = RandomForestClassifier(random_state=42)
grid_search_epilepsy = GridSearchCV(rf_classifier_epilepsy, param_grid, cv=5, scoring='accuracy')
grid_search_epilepsy.fit(X_train, y_train)

# Save the best Random Forest model for epilepsy diagnosis
with open('models/epilepsy_rf.pkl', 'wb') as f:
    pickle.dump(grid_search_epilepsy.best_estimator_, f)

# Predictions and evaluation of the epilepsy model
y_pred_epilepsy = grid_search_epilepsy.best_estimator_.predict(X_test)
accuracy_epilepsy = accuracy_score(y_test, y_pred_epilepsy)
print('Epilepsy Diagnosis Accuracy:', accuracy_epilepsy)
print('Epilepsy Classification Report:')
print(classification_report(y_test, y_pred_epilepsy, zero_division=0))

### 2. **Activity Tracking** using Random Forest ###
# Define activity labels for each file
activity_labels = {
    '00_desk work': 0,
    '00_idle sitting': 1,
    '00_laying': 2,
    '00_sitting': 3,
    '00_standing': 4,
    '00_watching tv': 5,
    '01_cooking': 6,
    '01_stairs': 7,
    '01_walking': 8,
    '02_jogging': 9,
    '02_running': 10,
    '02_jumping': 11
}

# Function to strip the numeric suffix from file names and clean the activity label
def strip_numeric_suffix(file_name):
    activity_code = file_name.split('_')[0] + '_' + ''.join([i for i in file_name.split('_')[1] if not i.isdigit()]).strip()
    activity_code = activity_code.replace(".csv", "")
    return activity_code.strip()

# Create activity labels for the data
y_activity = []
for file in healthy_individuals_files + epileptic_patients_files:
    activity = strip_numeric_suffix(file)
    if activity in activity_labels:
        y_activity.append(activity_labels[activity])
    else:
        print(f"Activity code not found in mapping: {activity}")
        continue

# Split the data for supervised learning (Activity Tracking)
X_train_activity, X_test_activity, y_train_activity, y_test_activity = train_test_split(X, y_activity, test_size=0.2, random_state=42)

# Random Forest Classifier for Activity Tracking
rf_activity_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_activity_classifier.fit(X_train_activity, y_train_activity)

# Save the Random Forest model for Activity Tracking
with open('models/activity_rf.pkl', 'wb') as f:
    pickle.dump(rf_activity_classifier, f)

# Predictions and evaluation of Activity Tracking
y_pred_activity = rf_activity_classifier.predict(X_test_activity)
accuracy_activity = accuracy_score(y_test_activity, y_pred_activity)
print('Activity Tracking Accuracy:', accuracy_activity)
print('Activity Classification Report:')
print(classification_report(y_test_activity, y_pred_activity, zero_division=0))
