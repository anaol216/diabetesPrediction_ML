"""
Data Preprocessing for Diabetes Dataset

This script handles data preprocessing including loading, splitting,
scaling, and validation for the diabetes prediction model.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os


def load_diabetes_data():
    """
    Load the diabetes dataset from sklearn.
    
    Returns:
        tuple: (X, y, feature_names)
    """
    print("Loading diabetes dataset...")
    diab = load_diabetes()
    X = diab['data']
    y = diab['target']
    feature_names = diab['feature_names']
    
    print(f"Dataset loaded successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    
    return X, y, feature_names


def create_dataframe(X, y, feature_names):
    """
    Create a pandas DataFrame from arrays.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        pd.DataFrame: Combined dataframe
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


def validate_data(X, y):
    """
    Validate the dataset for missing values and data quality.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        bool: True if data is valid
    """
    print("\nValidating data...")
    
    # Check for NaN values
    if np.isnan(X).any():
        print("WARNING: NaN values found in features!")
        return False
    
    if np.isnan(y).any():
        print("WARNING: NaN values found in target!")
        return False
    
    # Check for infinite values
    if np.isinf(X).any():
        print("WARNING: Infinite values found in features!")
        return False
    
    if np.isinf(y).any():
        print("WARNING: Infinite values found in target!")
        return False
    
    # Check shapes match
    if X.shape[0] != y.shape[0]:
        print("ERROR: Number of samples in X and y don't match!")
        return False
    
    print("Data validation passed!")
    return True


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, scaler_type='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Args:
        X_train: Training features
        X_test: Testing features
        scaler_type: Type of scaler ('standard' or 'minmax')
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print(f"\nScaling features using {scaler_type} scaler...")
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling completed!")
    
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, output_dir='ouputs'):
    """
    Save preprocessed data and scaler to disk.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        scaler: Fitted scaler object
        output_dir: Directory to save files
    """
    print(f"\nSaving preprocessed data to '{output_dir}' directory...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Preprocessed data saved successfully!")


def load_preprocessed_data(input_dir='ouputs'):
    """
    Load preprocessed data and scaler from disk.
    
    Args:
        input_dir: Directory containing saved files
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print(f"\nLoading preprocessed data from '{input_dir}' directory...")
    
    X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
    
    with open(os.path.join(input_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    print("Preprocessed data loaded successfully!")
    
    return X_train, X_test, y_train, y_test, scaler


def preprocess_pipeline(test_size=0.2, random_state=42, scaler_type='standard', save_data=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        test_size: Proportion of data for testing
        random_state: Random seed
        scaler_type: Type of scaler to use
        save_data: Whether to save preprocessed data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("="*80)
    print("DIABETES DATASET - PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load data
    X, y, feature_names = load_diabetes_data()
    
    # Validate data
    if not validate_data(X, y):
        raise ValueError("Data validation failed!")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_type)
    
    # Save if requested
    if save_data:
        save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def main():
    """
    Main function to run preprocessing pipeline.
    """
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(
        test_size=0.2,
        random_state=42,
        scaler_type='standard',
        save_data=True
    )
    
    print(f"\nPreprocessed data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    main()
