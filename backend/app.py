import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from emotion_analysis import analyze_emotion
from tensorflow.keras.models import load_model
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models for activity tracking, epilepsy diagnosis, emotion analysis, and scaler
with open('models/activity_rf.pkl', 'rb') as f:
    activity_model = pickle.load(f)

with open('models/epilepsy_rf.pkl', 'rb') as f:
    epilepsy_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load emotion model
emotion_model = load_model('models/emotion_model.h5')

# Function to extract features from EEG data
def extract_features(eeg_data):
    features = []
    features.append(np.mean(eeg_data))
    features.append(np.std(eeg_data))
    features.append(np.min(eeg_data))
    features.append(np.max(eeg_data))
    features.append(np.median(eeg_data))
    features.append(skew(eeg_data))
    features.append(kurtosis(eeg_data))
    
    fft_values = np.fft.fft(eeg_data)
    fft_magnitude = np.abs(fft_values)
    features.append(np.mean(fft_magnitude))
    features.append(np.std(fft_magnitude))
    
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}
    for band, (low, high) in bands.items():
        band_power = np.mean(fft_magnitude[(np.fft.fftfreq(len(eeg_data), 1/512) >= low) & 
                                            (np.fft.fftfreq(len(eeg_data), 1/512) <= high)])
        features.append(band_power)

    return features

# Preprocess the input EEG data for activity analysis
def preprocess_activity_data(file):
    eeg_data = pd.read_csv(file, header=None)
    
    if eeg_data.empty:
        raise ValueError("Uploaded CSV is empty.")
    
    eeg_data = eeg_data.values.flatten()
    
    # Convert to numeric and handle errors, setting invalid parsing as NaN
    eeg_data = pd.to_numeric(eeg_data, errors='coerce')
    eeg_data = eeg_data[~np.isnan(eeg_data)]

    if len(eeg_data) == 0:
        raise ValueError("All values in the uploaded file were invalid.")

    features = extract_features(eeg_data)
    features = np.array(features).reshape(1, -1)  # Reshape for the scaler

    features_scaled = scaler.transform(features)
    return features_scaled

# Analyze the uploaded EEG data for activity and epilepsy
def analyze_activity_and_epilepsy(file):
    eeg_data_scaled = preprocess_activity_data(file)

    # Activity tracking
    activity_prediction = activity_model.predict(eeg_data_scaled)
    predicted_activity = activity_prediction[0]  # Get the predicted class directly

    # Epilepsy diagnosis
    epilepsy_prediction = epilepsy_model.predict(eeg_data_scaled)
    is_epileptic = bool(np.round(epilepsy_prediction[0]))

    return predicted_activity, is_epileptic

# Emotion analysis route
@app.route('/api/emotion', methods=['POST'])
def emotion_analysis_route():
    file = request.files['file']
    if file:
        emotional_state, plot_path = analyze_emotion(file)  # Call the imported function
        full_path = os.path.join(os.getcwd(), plot_path)
        print("---------------------------------------------------------------------")
        print("Emotional State: ", emotional_state)
        print("-----------------------------------------------------------------------")
        return jsonify({
            'emotion': emotional_state,
            'plot': plot_path
        })
    return jsonify({'error': 'No file uploaded'}), 400

# Activity analysis and epilepsy diagnosis route
@app.route('/api/activity', methods=['POST'])
def activity_analysis_route():
    file = request.files['file']
    if file:
        activity_cluster, is_epileptic = analyze_activity_and_epilepsy(file)
        print("--------------------------------------------------------------------")
        print("Activity Cluster: ", activity_cluster)
        print("Epileptic: ", is_epileptic)
        print("---------------------------------------------------------------------")
        return jsonify({
            'activity': int(activity_cluster),
            'epileptic': is_epileptic
        })
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == "__main__":
    app.run(debug=True)
