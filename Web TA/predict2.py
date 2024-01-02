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

#PREDICTING AND NEW LAG
# Select the first row (index 0) from X
X_first_row = X.iloc[0:1]

# Use the model to predict the traffic volume for the first row
predicted_traffic_volume = model.predict(X_first_row)

# Create a new DataFrame with the same index as df
df_predicted = pd.DataFrame(index=df.index)

# Initialize the 'predicted_traffic_volume' column with NaNs
df_predicted['predicted_traffic_volume'] = np.nan

# Select the first row (index 0) from X
X_first_row = X.iloc[0:1]

# Use the model to predict the traffic volume for the first row
predicted_traffic_volume = model.predict(X_first_row)

# Assign the predicted value to the first row in df_predicted
df_predicted.loc[df_predicted.index[0], 'predicted_traffic_volume'] = predicted_traffic_volume[0]


# df.iloc[1, df.columns.get_loc('traffic_volume_lag_1')] = df_predicted.iloc[0, df_predicted.columns.get_loc('predicted_traffic_volume')]
# df.iloc[2, df.columns.get_loc('traffic_volume_lag_2')] = df_predicted.iloc[1, df_predicted.columns.get_loc('predicted_traffic_volume')]
# df.iloc[3, df.columns.get_loc('traffic_volume_lag_3')] = df_predicted.iloc[2, df_predicted.columns.get_loc('predicted_traffic_volume')]

# Initialize the 'predicted_traffic_volume' column with NaNs
df_result = np.nan

df_result = pd.concat([df, df_predicted], axis=1)

# print(df_result)
# # Display the updated DataFrame (or the first few rows to check the changes)
print(df_result[['date_time', 'traffic_volume_lag_1', 'traffic_volume_lag_2', 'traffic_volume_lag_3', 'predicted_traffic_volume']])

print(df)




