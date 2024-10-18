import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from eeg_utils import preprocess_eeg_data
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('/home/jagan/Development/eeg_final/backend/emotions.csv')

def simple_preprocess_eeg_data(df):
   
    if 'label' in df.columns:
        X = df.drop('label', axis=1).copy()
    else:
        X = df.copy()
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
  
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    return X_reshaped


x=simple_preprocess_eeg_data(df)
y = df['label'].astype('category').cat.codes.values

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


model = tf.keras.models.load_model('models/emotion_model.h5')

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=35)


def plot_learning_curve(history):

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('static/emotional_learning_curve.png')


plot_learning_curve(history)


