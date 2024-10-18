import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt

# Load dataset
data = pd.read_csv('/home/jagan/Development/eeg_final/backend/emotions.csv')

# Preprocess data
def preprocess_inputs(df):
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    df['label'] = df['label'].replace(label_mapping)

    y = df['label']
    X = df.drop('label', axis=1)

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

X, y = preprocess_inputs(data)

# Keras Tuner function to build the GRU model
def build_gru_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(hp.Int('units', 32, 256, step=32), return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5)))
    model.add(tf.keras.layers.GRU(hp.Int('units2', 32, 128, step=32)))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # Three classes for emotions
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Keras Tuner setup
tuner = kt.RandomSearch(build_gru_model, objective='val_accuracy', max_trials=1, executions_per_trial=2, directory='tuner', project_name='emotion')

# Reshape for model input
X = np.expand_dims(X, axis=2)

# Train the tuner to find the best model
tuner.search(X, y, epochs=10, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('models/emotion_model.h5')
