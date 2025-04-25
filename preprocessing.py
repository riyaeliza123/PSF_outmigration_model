# preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    data['date'] = pd.to_datetime(data['date'])
    return data

def feature_engineering(data):
    # Feature Engineering for Seasonality
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_year'] = data['date'].dt.dayofyear  # For capturing seasonality

    # Adding cyclic features to capture seasonality (sine and cosine transformations)
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
    data['sin_day_of_year'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['cos_day_of_year'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

    # Generate rolling features
    data['rolling_temp_mean_10'] = data['Temp'].rolling(window=10).mean()
    data['rolling_flow_mean_10'] = data['Flow'].rolling(window=10).mean()
    data['rolling_level_mean_10'] = data['Level'].rolling(window=10).mean()

    return data

def prepare_features(data, target='count'):
    features = ['Temp', 'Flow', 'Level', 'sin_month', 'cos_month', 'sin_day_of_year', 'cos_day_of_year', 
                'rolling_temp_mean_10', 'rolling_flow_mean_10', 'rolling_level_mean_10']

    # Remove rows with missing values
    data = data.dropna(subset=features + [target])

    # Split into features and target
    X = data[features]
    y = data[target]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
