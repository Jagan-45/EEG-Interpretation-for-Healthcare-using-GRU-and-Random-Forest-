import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the K-Means model
with open('models/activity_kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Preprocess the input EEG data
def preprocess_activity_data(file):
    eeg_data = pd.read_csv(file, header=None).values.flatten()
    scaler = StandardScaler()
    eeg_data_scaled = scaler.fit_transform(eeg_data.reshape(-1, 1)).flatten()
    return eeg_data_scaled.reshape(1, -1)

# Analyze the uploaded EEG data
def analyze_activity(file):
    eeg_data_scaled = preprocess_activity_data(file)
    predicted_cluster = kmeans.predict(eeg_data_scaled)
    return predicted_cluster[0]  # Return the predicted activity cluster
