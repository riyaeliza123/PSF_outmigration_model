# predict.py

import pickle
import pandas as pd
import numpy as np
from preprocessing import feature_engineering

def load_model_and_scaler(model_path='model.pkl', scaler_path='scaler.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

def prepare_prediction_data(prediction_year):
    start_date = f'{prediction_year}-01-01'
    end_date = f'{prediction_year}-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    prediction_data = pd.DataFrame({'date': date_range})
    prediction_data['year'] = prediction_data['date'].dt.year
    prediction_data['month'] = prediction_data['date'].dt.month
    prediction_data['day'] = prediction_data['date'].dt.day
    prediction_data['day_of_year'] = prediction_data['date'].dt.dayofyear

    # Add cyclic features
    prediction_data['sin_month'] = np.sin(2 * np.pi * prediction_data['month'] / 12)
    prediction_data['cos_month'] = np.cos(2 * np.pi * prediction_data['month'] / 12)
    prediction_data['sin_day_of_year'] = np.sin(2 * np.pi * prediction_data['day_of_year'] / 365)
    prediction_data['cos_day_of_year'] = np.cos(2 * np.pi * prediction_data['day_of_year'] / 365)

    # Simulate future temperature, flow, and level values
    prediction_data['Temp'] = np.random.uniform(15, 25, size=len(prediction_data))
    prediction_data['Flow'] = np.random.uniform(0.5, 2.5, size=len(prediction_data))
    prediction_data['Level'] = np.random.uniform(0.4, 1.0, size=len(prediction_data))

    # Calculate rolling features
    prediction_data['rolling_temp_mean_10'] = prediction_data['Temp'].rolling(window=10).mean().fillna(method='ffill')
    prediction_data['rolling_flow_mean_10'] = prediction_data['Flow'].rolling(window=10).mean().fillna(method='ffill')
    prediction_data['rolling_level_mean_10'] = prediction_data['Level'].rolling(window=10).mean().fillna(method='ffill')

    return prediction_data

def predict(prediction_year, model, scaler):
    prediction_data = prepare_prediction_data(prediction_year)

    # Prepare features and scale them
    features = ['Temp', 'Flow', 'Level', 'sin_month', 'cos_month', 'sin_day_of_year', 'cos_day_of_year', 
                'rolling_temp_mean_10', 'rolling_flow_mean_10', 'rolling_level_mean_10']
    
    X_pred = prediction_data[features]
    X_pred_scaled = scaler.transform(X_pred)

    # Make predictions
    predictions = model.predict(X_pred_scaled)
    prediction_data['pred'] = predictions

    # Adjust predictions and calculate cumulative percentages
    prediction_data["pred_adjust"] = prediction_data["pred"].apply(lambda x: 0 if x <= 0 else x)
    total_percentage = prediction_data["pred_adjust"].sum()
    prediction_data["pred_adjust_cumulative_percentage"] = (prediction_data["pred_adjust"] / total_percentage).cumsum() * 100

    return prediction_data
