import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_eeg_data(df):
    """
    Preprocess EEG data from the input DataFrame.
    Args:
    - df (pd.DataFrame): Input DataFrame with EEG data.
    Returns:
    - np.ndarray: Preprocessed EEG data for model input.
    """
    # Drop the 'label' column if present
    X = df.drop('label', axis=1).copy() if 'label' in df.columns else df
    
    # Check the shape of the input DataFrame
    print(f"Input DataFrame shape: {X.shape}")

    # Ensure the input has 2548 features
    if X.shape[1] > 1:
        # If there are more than 1 feature, average them to create a single feature
        X = X.mean(axis=1).values.reshape(-1, 1)  # Convert to 1D and then reshape to 2D
    elif X.shape[1] < 1:
        raise ValueError("The input DataFrame must have at least one feature.")

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check the number of rows in the scaled data
    print(f"Number of rows in scaled data: {X_scaled.shape[0]}")

    # If not enough rows, pad with zeros
    if X_scaled.shape[0] < 2548:
        additional_rows = 2548 - X_scaled.shape[0]
        padding = np.zeros((additional_rows, X_scaled.shape[1]))
        X_scaled = np.vstack((X_scaled, padding))

    # Ensure we only use the first 2548 rows for reshaping
    X_scaled = X_scaled[:2548]

    # Reshape the data to 3D: (batch_size, time_steps, features)
    return X_scaled.reshape(1, 2548, 1)  # Use only the first 2548 time steps