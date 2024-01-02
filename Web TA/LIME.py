import pandas as pd
import pymysql
from datetime import timedelta
import shap
import pandas as pd
import numpy as np
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
encoder = joblib.load('encoderbaru.joblib')
scaler = joblib.load('scalerbaru.joblib')
model = joblib.load('best_xgboost_model.joblib')

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

# Create a LIME explainer object
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X), 
    feature_names=X.columns, 
    class_names=['traffic_volume'], 
    mode='regression'
)

# Choose an instance to explain
instance_index = 0  # For example, explain the first instance in your dataset
instance = X.iloc[instance_index]

# Generate LIME explanation for this instance
exp = explainer.explain_instance(
    data_row=instance, 
    predict_fn=model.predict
)

# Visualize the explanation
exp.show_in_notebook(show_table=True)
