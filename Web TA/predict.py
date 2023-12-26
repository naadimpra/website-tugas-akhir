import pandas as pd
import pymysql
from datetime import timedelta

# Connect to the MySQL database
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="test_traffic"
)

# Define the query to fetch the last three rows
query = "SELECT * FROM ( SELECT * FROM trainsql ORDER BY date_time DESC LIMIT 6 ) AS subquery ORDER BY date_time ASC;"
#query = 'SELECT * FROM trainsql ORDER BY date_time ASC LIMIT 6'
# Execute the query and load the result into a DataFrame
df_raw = pd.read_sql(query, connection)

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load the encoder, scaler, and model
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler2.joblib')
model = joblib.load('best_xgboost_model_gridsearch.joblib')

# Load the dataset
#df_raw = pd.read_csv('Train.csv')

# Convert the 'date_time' column to datetime and sort the dataset
df_raw['date_time'] = pd.to_datetime(df_raw['date_time'])
df_raw.sort_values('date_time', inplace=True)

# Extracting non-numeric columns
non_numeric_cols = ['is_holiday', 'weather_type', 'weather_description']

# for col in non_numeric_cols:
#     print(f"Unique values for {col} in df_raw: {df_raw[col].unique()}")


# Group by 'date_time' and aggregate: mean for numeric columns, mode for non-numeric columns
agg_funcs = {col: 'mean' for col in df_raw.columns if col not in non_numeric_cols}
agg_funcs.update({col: lambda x: x.mode()[0] if not x.mode().empty else np.nan for col in non_numeric_cols})

df_aggregated = df_raw.groupby('date_time').agg(agg_funcs)


# Extract unique values for categorical columns from df_raw
unique_values = {col: df_raw[col].unique() for col in non_numeric_cols}

# One-hot encode categorical features using unique values
encoded_data = encoder.transform(df_aggregated[non_numeric_cols])

# Create a DataFrame with encoded data and columns
df_encode = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

# Ensure all unique values in df_raw are included in df_encode
for col in non_numeric_cols:
    for value in unique_values[col]:
        column_name = f"{col}_{value}"
        if column_name not in df_encode.columns:
            df_encode[column_name] = 0  # Add missing column with zeros

# Reset index of df_encode
df_encode.index = df_aggregated.index

# Concatenate with df_aggregated
df = pd.concat([df_aggregated, df_encode], axis=1)

# Add hour from the 'date_time' column
df['hour'] = df['date_time'].dt.hour
df = df.drop(columns=non_numeric_cols)

# Feature engineering: create lagged and rolling features
target = 'traffic_volume'
for i in range(1, 4):
    df[f'traffic_volume_lag_{i}'] = df[target].shift(i)
df['traffic_volume_rolling_mean'] = df[target].rolling(window=3).mean().shift(1)
df['traffic_volume_rolling_std'] = df[target].rolling(window=3).std().shift(1)

# Remove rows with NaN values resulting from lagged features
df.dropna(inplace=True)

# Split the dataset into features and the target
X = df.drop(target, axis=1)
y = df[target]

# Save 'date_time' for later use
date_time = df['date_time']

# Drop 'date_time' column before scaling
df = df.drop(columns=['date_time'])

scaler = joblib.load('scaler2.joblib')
# Check categories in encoder
# print(scaler.get_feature_names_out())

# Scale the numerical features
df_scaled = scaler.transform(df)  # Use the previously loaded scaler

# Convert scaled data back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=[col for col in df.columns if col != 'date_time'])
df_scaled['date_time'] = date_time.values

X = df_scaled.drop(columns=['date_time'])

X = X.drop(target, axis=1)
df = pd.concat([date_time, df], axis=1)

# Function to append new row with updated lagged features
def append_new_row(df, new_prediction, max_lags=3):
    new_row = df.iloc[-1].copy()  # Copy the last row to use as a base for the new row
    for i in range(max_lags-1, 0, -1):
        new_row[f'traffic_volume_lag_{i+1}'] = new_row[f'traffic_volume_lag_{i}']
    new_row['traffic_volume_lag_1'] = new_prediction
    new_row['forecasted_traffic_volume'] = np.nan  # Reset the forecasted value
    return df.append(new_row, ignore_index=True)

# Initialize DataFrame for dynamic forecasting
df_dynamic_forecast = X.copy()
df_dynamic_forecast['forecasted_traffic_volume'] = np.nan

# Number of steps to forecast
forecast_steps = 10

for i in range(forecast_steps):
    # Predict the traffic volume for the next time step
    current_prediction = model.predict(df_dynamic_forecast.iloc[i:i+1].drop(columns=['forecasted_traffic_volume']))[0]
    df_dynamic_forecast.at[df_dynamic_forecast.index[i], 'forecasted_traffic_volume'] = current_prediction

    # Append a new row with updated lagged features for the next prediction, if not at the last step
    if i + 1 < forecast_steps:
        df_dynamic_forecast = append_new_row(df_dynamic_forecast, current_prediction)

# Initialize the DataFrame
test_date_times = df['date_time'].reset_index(drop=True)

# Initialize the DataFrame without setting 'date_time' as the index
df_result = pd.DataFrame({
    'date_time': test_date_times,
    'actual_traffic_volume': y.reset_index(drop=True),
    'lag_1': np.nan,
    'lag_2': np.nan,
    'lag_3': np.nan,
    'forecasted_traffic_volume': df_dynamic_forecast['forecasted_traffic_volume'].reset_index(drop=True)
})

# Set the initial lagged values from the historical data
df_result.loc[0, 'lag_1'] = df_raw.iloc[-4]['traffic_volume']  # Most recent record
df_result.loc[0, 'lag_2'] = df_raw.iloc[-5]['traffic_volume']  # Second most recent record
df_result.loc[0, 'lag_3'] = df_raw.iloc[-6]['traffic_volume']  # Third most recent record

# Update the lagged values with the forecasted values in each step
for i in range(1, len(df_result)):
    df_result.loc[i, 'lag_1'] = df_result.loc[i - 1, 'forecasted_traffic_volume']
    df_result.loc[i, 'lag_2'] = df_result.loc[i - 1, 'lag_1']
    df_result.loc[i, 'lag_3'] = df_result.loc[i - 1, 'lag_2']

for i in range(1, len(df_result)):
    if pd.isna(df_result.loc[i, 'date_time']):
        df_result.loc[i, 'date_time'] = df_result.loc[i-1, 'date_time'] + pd.Timedelta(hours=1)

# Display the DataFrame
print(df_result.head(forecast_steps))


