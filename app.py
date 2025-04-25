# app.py

import streamlit as st
import matplotlib.pyplot as plt
from predict import predict

import pickle

# Load the model and scaler
model_filename = 'model.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Streamlit user interface components
st.title("Salmon Outmigration Timing Predictions")
prediction_year = st.number_input("Enter the year for prediction:", min_value=2020, max_value=2030, step=1)
lower_percentile = st.slider("Select the lower percentile of outmigration:", min_value=0, max_value=100, value=5)
upper_percentile = st.slider("Select the upper percentile of outmigration:", min_value=0, max_value=100, value=10)

# Make predictions
prediction_data = predict(prediction_year, model, scaler)

# Filter by percentile range
result_df = prediction_data[(prediction_data["pred_adjust_cumulative_percentage"] >= lower_percentile) & 
                             (prediction_data["pred_adjust_cumulative_percentage"] <= upper_percentile)]

start_date_ts = result_df.iloc[0]['date']
end_date_ts = result_df.iloc[-1]['date']

# Display the predicted date range
st.write(f"The predicted date range for salmon counts between {lower_percentile}% and {upper_percentile}% of total adjusted counts in {prediction_year} is between {start_date_ts.strftime('%Y-%m-%d')} and {end_date_ts.strftime('%Y-%m-%d')}.")

# Plot the predictions
st.subheader("Prediction Plot")
df_plot = prediction_data.copy()
df_plot = df_plot.set_index('date')

plt.figure(figsize=(10, 6))
plt.plot(df_plot.index, df_plot['pred'], label='Prediction', color='red')
plt.plot(df_plot.index, df_plot['pred_adjust'], label=f'Adjusted Prediction ({lower_percentile}% to {upper_percentile}%)', color='blue')

# Highlight the prediction range
plt.axvline(x=start_date_ts, color='blue', linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(x=end_date_ts, color='blue', linestyle='--', linewidth=2, alpha=0.7)

plt.title(f'Prediction and Adjusted Prediction Over Time in {prediction_year}')
plt.xlabel('Date')
plt.ylabel('Count of Salmon')
plt.legend()

# Show the plot in the Streamlit app
st.pyplot(plt)
