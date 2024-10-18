import numpy as np
import pandas as pd
from eeg_utils import preprocess_eeg_data
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the pre-trained GRU model
model = tf.keras.models.load_model('models/emotion_model.h5')
print("Model loaded successfully.")
print(model.input_shape)

def analyze_emotion(file):
    # Load the model
    model = tf.keras.models.load_model('models/emotion_model.h5')
    print("Model loaded successfully.")
    print(model.input_shape)
    
    # Read the input CSV file
    df = pd.read_csv(file)
    
    # Preprocess the EEG data
    # test_df = pd.read_csv('/home/jagan/Music/mental-state.csv')  # Replace with your test data file path
    # y_test = test_df['label'].values  # Assuming true labels are in a 'label' column

    X_input = preprocess_eeg_data(df)

    # Check input shape before passing to the model
    print(f"Input shape for model: {X_input.shape}")

    # Predict emotional state
    y_pred = np.argmax(model.predict(X_input), axis=1)
    emotional_states = {0: 'Sleep/Unconscious', 1: 'Calm', 2: 'Active/Stress'}
    emotional_state = emotional_states[y_pred[0]]
    print("emotional states", emotional_states[2])
    print("y pred:", y_pred)
    print("Predicted emotional state:", emotional_state)
    print(emotional_state)

    # Generate FFT plot using all relevant columns
    # Get all FFT columns dynamically
    fft_columns = [col for col in df.columns if 'freq_' in col]
    # print(df)
    # print(fft_columns)
    
    # Check if there are any FFT columns
    if not fft_columns:
        print("No FFT columns found in DataFrame.")
        return None, None

    # Plot the FFT data
    plt.figure(figsize=(16, 10))
    for col in fft_columns:
        plt.plot(df[col], label=col)  # Plot each FFT column
    plt.title(f"EEG Signal - Emotion: {emotional_state}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plot_path = 'static/emotion_plot.png'
    plt.savefig(plot_path)
    plt.close()
    

    return emotional_state, plot_path