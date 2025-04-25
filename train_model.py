import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import pickle
from sklearn.metrics import mean_absolute_error


# Get the absolute path to the CSV file (using the script directory)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
csv_file_path = os.path.join(script_dir, 'data', 'preprocessed', 'preprocessed_ck.csv')

# Load the historical data from the CSV file
data = pd.read_csv(csv_file_path)

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

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

# Define target and features
target = 'count'
features = ['Temp', 'Flow', 'Level', 'sin_month', 'cos_month', 'sin_day_of_year', 'cos_day_of_year', 
            'rolling_temp_mean_10', 'rolling_flow_mean_10', 'rolling_level_mean_10']

# Remove rows with missing values
data = data.dropna(subset=features + [target])

# Split into training and test sets
X = data[features]
y = data[target]

# Split the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection using Lasso Regression with Cross Validation
alpha_vals = np.arange(0.1, 1.0, 0.05)
best_alpha = None
best_score = float('inf')

# Time Series Cross Validation
tscv = TimeSeriesSplit(n_splits=5)

# Lasso for Feature Selection
for alpha in alpha_vals:
    lasso = Lasso(alpha=alpha)
    scores = []
    for train_index, val_index in tscv.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        lasso.fit(X_train_fold, y_train_fold)
        y_pred = lasso.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        scores.append(rmse)
    avg_score = np.mean(scores)
    if avg_score < best_score:
        best_score = avg_score
        best_alpha = alpha

# Fit the final Lasso model with the best alpha value
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train_scaled, y_train)

# Select important features
selected_features = SelectFromModel(lasso, prefit=True).get_support()
selected_feature_names = X_train.columns[selected_features]

# Create new datasets with only the selected features
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

# Scale the selected features
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train the XGBoost model on the selected features
model.fit(X_train_selected_scaled, y_train)

# Make predictions on the test set
preds = model.predict(X_test_selected_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Save the trained model and scaler
model_filename = 'model/outmigrationModel.pkl'
scaler_filename = 'model/scaler.pkl'

# Save the model
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Model and scaler saved to {model_filename} and {scaler_filename}.")
