from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
import math
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import shap
import pymysql
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Configure MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="test_traffic"
)
cursor = db.cursor()

# Load the encoder, scaler, and model
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler2.joblib')
model = joblib.load('best_xgboost_model_gridsearch.joblib')

# Define global variables
X = None
explainer = None
model = None

# Number of rows to display per page
ROWS_PER_PAGE = 10

def _force_plot_html(explainer, shap_values, index):
    base_value = explainer.expected_value
    force_plot = shap.plots.force(base_value, shap_values[index], matplotlib=True)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html

# Column labels
column_labels = {
    'date_time': 'Date and Time',
    'is_holiday': 'Is Holiday',
    'air_pollution_index': 'Air Pollution Index',
    'humidity': 'Humidity',
    'wind_speed': 'Wind Speed',
    'wind_direction': 'Wind Direction',
    'visibility_in_miles': 'Visibility in Miles',
    'dew_point': 'Dew Point',
    'temperature': 'Temperature',
    'rain_p_h': 'Rainfall (mm per hour)',
    'snow_p_h': 'Snowfall (mm per hour)',
    'clouds_all': 'Clouds (percentage)',
    'weather_type': 'Weather Type',
    'weather_description': 'Weather Description',
    'traffic_volume': 'Traffic Volume'
}

def get_data(filter_column, filter_value, sort_column, sort_order, page, rows_per_page):
    # Retrieve column names
    table_name = "trainsql"
    cursor.execute(f"DESCRIBE {table_name}")
    columns = [column[0] for column in cursor.fetchall()]

    # Build the WHERE clause for filtering
    where_clause = ''
    if filter_column and filter_value:
        if "-" in filter_value:
            # Range filter
            start_range, end_range = map(int, filter_value.split('-'))
            where_clause = f"WHERE {filter_column} BETWEEN {start_range} AND {end_range}"
        else:
            # Exact match or partial search
            where_clause = f"WHERE {filter_column} LIKE '%{filter_value}%'"

    # Build the ORDER BY clause for sorting
    order_by_clause = ''
    if sort_column:
        if sort_column == request.args.get('sort_column') and sort_order == 'asc':
            # If the same column is clicked again and the current order is ascending, toggle to descending
            sort_order = 'desc'
        else:
            # For any other case, set the order to ascending
            sort_order = 'asc'
        order_by_clause = f"ORDER BY {sort_column} {sort_order}"

    # Calculate the LIMIT and OFFSET for pagination
    offset = (page - 1) * rows_per_page
    limit = rows_per_page

    # Retrieve data from MySQL table with pagination, filtering, and sorting
    query = f"SELECT * FROM {table_name} {where_clause} {order_by_clause} LIMIT {limit} OFFSET {offset}"
    cursor.execute(query)
    data = cursor.fetchall()

    # Calculate total number of rows for pagination
    total_rows_query = f"SELECT COUNT(*) FROM {table_name} {where_clause}"
    cursor.execute(total_rows_query)
    total_rows = cursor.fetchone()[0]

    # Calculate total number of pages
    total_pages = math.ceil(total_rows / rows_per_page)

    return data, columns, total_rows, total_pages

@app.route('/table')
def table():
    # Retrieve filter and sort parameters from query string
    filter_column = request.args.get('filter_column', '')
    filter_value = request.args.get('filter_value', '')
    sort_column = request.args.get('sort_column', '')
    sort_order = request.args.get('sort_order', 'asc')  # Default to ascending order
    page = int(request.args.get('page', 1))
    rows_per_page = int(request.args.get('rows', ROWS_PER_PAGE))

    # Retrieve data using the helper function
    data, columns, total_rows, total_pages = get_data(filter_column, filter_value, sort_column, sort_order, page, rows_per_page)

    # Pass data, columns, and pagination information to the template
    return render_template('table.html', data=data, columns=columns, page=page, rowsPerPage=rows_per_page, totalPages=total_pages,
                           totalRows=total_rows, filter_column=filter_column, filter_value=filter_value, sort_column=sort_column,
                           sort_order=sort_order, column_labels=column_labels)

@app.route('/insert', methods=['GET', 'POST'])
def insert():
    table_name = "trainsql"
    holiday_options = ['None', 'Christmas Day', 'Columbus Day', 'Independence Day', 'Labor Day', 'Martin Luther King Jr Day', 'Memorial Day', 'New Years Day', 'State Fair', 'Thanksgiving Day', 'Veterans Day', 'Washingtons Birthday']  # Example values
    weather_type_options = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Snow', 'Squall', 'Thunderstorm']            # Example values
    weather_description_options = [
    'Sky is Clear',
    'broken clouds',
    'drizzle',
    'few clouds',
    'fog',
    'freezing rain',
    'haze',
    'heavy intensity drizzle',
    'heavy intensity rain',
    'heavy snow',
    'light intensity drizzle',
    'light intensity shower rain',
    'light rain',
    'light shower snow',
    'light snow',
    'mist',
    'moderate rain',
    'overcast clouds',
    'proximity shower rain',
    'proximity thunderstorm',
    'proximity thunderstorm with rain',
    'scattered clouds',
    'snow',
    'thunderstorm',
    'thunderstorm with heavy rain',
    'thunderstorm with light drizzle',
    'thunderstorm with light rain',
    'very heavy rain'
]

    
    if request.method == 'POST':
        # Retrieve form data
        date_time_str = request.form['date_time']
        is_holiday = request.form['is_holiday']
        air_pollution_index = int(request.form['air_pollution_index'])
        humidity = int(request.form['humidity'])
        wind_speed = int(request.form['wind_speed'])
        wind_direction = int(request.form['wind_direction'])
        visibility_in_miles = int(request.form['visibility_in_miles'])
        dew_point = int(request.form['dew_point'])
        temperature = float(request.form['temperature'])
        rain_p_h = float(request.form['rain_p_h'])
        snow_p_h = float(request.form['snow_p_h'])
        clouds_all = int(request.form['clouds_all'])
        weather_type = request.form['weather_type']
        weather_description = request.form['weather_description']
        traffic_volume = int(request.form['traffic_volume'])

        try:
            date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            # Handle the error if the date_time is not in the expected format
            # You might want to return an error message to the user
            return "Invalid date and time format", 400
        
        formatted_date_time = date_time_obj.strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert data into the MySQL table
        query = f"""
            INSERT INTO {table_name} 
            (date_time, is_holiday, air_pollution_index, humidity, wind_speed, wind_direction,
            visibility_in_miles, dew_point, temperature, rain_p_h, snow_p_h, clouds_all,
            weather_type, weather_description, traffic_volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (formatted_date_time, is_holiday, air_pollution_index, humidity, wind_speed, wind_direction,
                  visibility_in_miles, dew_point, temperature, rain_p_h, snow_p_h, clouds_all,
                  weather_type, weather_description, traffic_volume)
        print (values)
        cursor.execute(query, values)
        db.commit()

        # Redirect to the index page after successful insertion
        return redirect(url_for('index'))

    # If the request method is GET, render the insert.html template
    return render_template('insert.html', 
                           holiday_options=holiday_options,
                           weather_type_options=weather_type_options,
                           weather_description_options=weather_description_options)

def update_lagged_features(df, new_prediction, max_lags=3):
    for i in range(max_lags-1, 0, -1):
        df[f'traffic_volume_lag_{i+1}'] = df[f'traffic_volume_lag_{i}']
    df['traffic_volume_lag_1'] = new_prediction
    
@app.route('/chart')
def chart():
    # Connect to your database and fetch the most recent 24 records
    table_name = "trainsql"

    # Retrieve the 24 most recent records from MySQL table
    query = f"""
        WITH RowNumberedData AS (
            SELECT date_time, traffic_volume,
                ROW_NUMBER() OVER (ORDER BY date_time) AS row_num
            FROM {table_name}
        )
        SELECT date_time, traffic_volume
        FROM RowNumberedData
        WHERE row_num BETWEEN (
            SELECT MIN(row_num) 
            FROM RowNumberedData
            WHERE date_time >= (SELECT MAX(date_time) FROM RowNumberedData) - INTERVAL 1 DAY - INTERVAL 10 HOUR
        ) AND (
            SELECT MAX(row_num) 
            FROM RowNumberedData
            WHERE date_time <= (SELECT MAX(date_time) FROM RowNumberedData) - INTERVAL 10 HOUR
        );
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Extract relevant data for the chart (e.g., date_time and traffic_volume)
    chart_data = [(entry[0], entry[1]) for entry in data]

    return render_template('chart.html', chart_data=chart_data)

from flask import render_template


@app.route('/')
def index():
    # Define the query to fetch the last three rows
    query = "SELECT * FROM ( SELECT * FROM trainsql ORDER BY date_time DESC LIMIT 13 ) AS subquery ORDER BY date_time ASC;"
    #query = "SELECT * FROM testsql LIMIT 7"
    #query = 'SELECT * FROM trainsql ORDER BY date_time ASC LIMIT 100'
    # Execute the query and load the result into a DataFrame
    df_raw = pd.read_sql(query, db)

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

    scaler = joblib.load('scalerbaru.joblib')
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

    forecasted_traffic_volume = model.predict(X)

    # Create a DataFrame for the results
    df_result = pd.DataFrame({
        'date_time': date_time,
        'actual_traffic_volume': y.values,  # Use y.values to get the underlying NumPy array
        'lag_1': df['traffic_volume_lag_1'].values,
        'lag_2': df['traffic_volume_lag_2'].values,
        'lag_3': df['traffic_volume_lag_3'].values,
        'forecasted_traffic_volume': forecasted_traffic_volume
    })
    # # Function to append new row with updated lagged features
    # def append_new_row(df, new_prediction, max_lags=3):
    #     new_row = df.iloc[-1].copy()  # Copy the last row to use as a base for the new row
    #     for i in range(max_lags-1, 0, -1):
    #         new_row[f'traffic_volume_lag_{i+1}'] = new_row[f'traffic_volume_lag_{i}']
    #     new_row['traffic_volume_lag_1'] = new_prediction
    #     new_row['forecasted_traffic_volume'] = np.nan  # Reset the forecasted value
    #     return df.append(new_row, ignore_index=True)

    # # Initialize DataFrame for dynamic forecasting
    # df_dynamic_forecast = X.copy()
    # df_dynamic_forecast['forecasted_traffic_volume'] = np.nan

    # # Number of steps to forecast
    # forecast_steps = 10

    # for i in range(forecast_steps):
    #     # Predict the traffic volume for the next time step
    #     current_prediction = model.predict(df_dynamic_forecast.iloc[i:i+1].drop(columns=['forecasted_traffic_volume']))[0]
    #     df_dynamic_forecast.at[df_dynamic_forecast.index[i], 'forecasted_traffic_volume'] = current_prediction

    #     # Append a new row with updated lagged features for the next prediction, if not at the last step
    #     if i + 1 < forecast_steps:
    #         df_dynamic_forecast = append_new_row(df_dynamic_forecast, current_prediction)

    # # Initialize the DataFrame
    # test_date_times = df['date_time'].reset_index(drop=True)

    # # Initialize the DataFrame without setting 'date_time' as the index
    # df_result = pd.DataFrame({
    #     'date_time': test_date_times,
    #     'actual_traffic_volume': y.reset_index(drop=True),
    #     'lag_1': np.nan,
    #     'lag_2': np.nan,
    #     'lag_3': np.nan,
    #     'forecasted_traffic_volume': df_dynamic_forecast['forecasted_traffic_volume'].reset_index(drop=True)
    # })

    # # Set the initial lagged values from the historical data
    # df_result.loc[0, 'lag_1'] = df_raw.iloc[-4]['traffic_volume']  # Most recent record
    # df_result.loc[0, 'lag_2'] = df_raw.iloc[-5]['traffic_volume']  # Second most recent record
    # df_result.loc[0, 'lag_3'] = df_raw.iloc[-6]['traffic_volume']  # Third most recent record

    # # Update the lagged values with the forecasted values in each step
    # for i in range(1, len(df_result)):
    #     df_result.loc[i, 'lag_1'] = df_result.loc[i - 1, 'actual_traffic_volume']
    #     df_result.loc[i, 'lag_2'] = df_result.loc[i - 1, 'lag_1']
    #     df_result.loc[i, 'lag_3'] = df_result.loc[i - 1, 'lag_2']

    # for i in range(1, len(df_result)):
    #     if pd.isna(df_result.loc[i, 'date_time']):
    #         df_result.loc[i, 'date_time'] = df_result.loc[i-1, 'date_time'] + pd.Timedelta(hours=1)
            
    # df_result.dropna(inplace=True)
    print(df_result)

    # # Display the DataFrame
    # print(df_result.head(forecast_steps))
    # Convert df_result to a format suitable for Chart.js
    chart_labels = df_result['date_time'].astype(str).tolist()
    forecasted_values = df_result['forecasted_traffic_volume'].tolist()
    actual_values = df_result['actual_traffic_volume'].tolist()
    
    table_name = "trainsql"
    # Fetch chart data (similar to what you have in chart())
    query = f"""
        WITH RowNumberedData AS (
            SELECT date_time, traffic_volume,
                ROW_NUMBER() OVER (ORDER BY date_time) AS row_num
            FROM {table_name}
        )
        SELECT date_time, traffic_volume
        FROM RowNumberedData
        WHERE row_num BETWEEN (
            SELECT MIN(row_num) 
            FROM RowNumberedData
            WHERE date_time >= (SELECT MAX(date_time) FROM RowNumberedData) - INTERVAL 1 DAY - INTERVAL 10 HOUR
        ) AND (
            SELECT MAX(row_num) 
            FROM RowNumberedData
            WHERE date_time <= (SELECT MAX(date_time) FROM RowNumberedData) - INTERVAL 10 HOUR
        );
    """
    cursor.execute(query)
    data = cursor.fetchall()
    chart_data = [(entry[0], entry[1]) for entry in data]

    # Pass chart data along with forecast data to the template
    return render_template('index.html', labels=chart_labels, values=forecasted_values, actual_values=actual_values, result=df_result.head(10).to_dict(orient='records'), chart_data=chart_data)

def give_shap_plot():
    
    # Define the query to fetch the last three rows
    query = "SELECT * FROM ( SELECT * FROM trainsql ORDER BY date_time DESC LIMIT 13 ) AS subquery ORDER BY date_time ASC;"
    #query = "SELECT * FROM testsql LIMIT 6"
    #query = 'SELECT * FROM trainsql ORDER BY date_time ASC LIMIT 6'
    # Execute the query and load the result into a DataFrame
    df_raw = pd.read_sql(query, db)
    
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

    forecasted_traffic_volume = model.predict(X)

    # Create a DataFrame for the results
    df_result = pd.DataFrame({
        'date_time': date_time,
        'actual_traffic_volume': y.values,  # Use y.values to get the underlying NumPy array
        'lag_1': df['traffic_volume_lag_1'].values,
        'lag_2': df['traffic_volume_lag_2'].values,
        'lag_3': df['traffic_volume_lag_3'].values,
        'forecasted_traffic_volume': forecasted_traffic_volume
    })
    # # Function to append new row with updated lagged features
    # def append_new_row(df, new_prediction, max_lags=3):
    #     new_row = df.iloc[-1].copy()  # Copy the last row to use as a base for the new row
    #     for i in range(max_lags-1, 0, -1):
    #         new_row[f'traffic_volume_lag_{i+1}'] = new_row[f'traffic_volume_lag_{i}']
    #     new_row['traffic_volume_lag_1'] = new_prediction
    #     new_row['forecasted_traffic_volume'] = np.nan  # Reset the forecasted value
    #     return df.append(new_row, ignore_index=True)

    # # Initialize DataFrame for dynamic forecasting
    # df_dynamic_forecast = X.copy()
    # df_dynamic_forecast['forecasted_traffic_volume'] = np.nan

    # # Number of steps to forecast
    # forecast_steps = 6

    # for i in range(forecast_steps):
    #     # Predict the traffic volume for the next time step
    #     current_prediction = model.predict(df_dynamic_forecast.iloc[i:i+1].drop(columns=['forecasted_traffic_volume']))[0]
    #     df_dynamic_forecast.at[df_dynamic_forecast.index[i], 'forecasted_traffic_volume'] = current_prediction

    #     # Append a new row with updated lagged features for the next prediction, if not at the last step
    #     if i + 1 < forecast_steps:
    #         df_dynamic_forecast = append_new_row(df_dynamic_forecast, current_prediction)

    # # Initialize the DataFrame
    # test_date_times = df['date_time'].reset_index(drop=True)

    # # Initialize the DataFrame without setting 'date_time' as the index
    # df_result = pd.DataFrame({
    #     'date_time': test_date_times,
    #     'actual_traffic_volume': y.reset_index(drop=True),
    #     'lag_1': np.nan,
    #     'lag_2': np.nan,
    #     'lag_3': np.nan,
    #     'forecasted_traffic_volume': df_dynamic_forecast['forecasted_traffic_volume'].reset_index(drop=True)
    # })

    # # Set the initial lagged values from the historical data
    # df_result.loc[0, 'lag_1'] = df_raw.iloc[-4]['traffic_volume']  # Most recent record
    # df_result.loc[0, 'lag_2'] = df_raw.iloc[-5]['traffic_volume']  # Second most recent record
    # df_result.loc[0, 'lag_3'] = df_raw.iloc[-6]['traffic_volume']  # Third most recent record

    # # Update the lagged values with the forecasted values in each step
    # for i in range(1, len(df_result)):
    #     df_result.loc[i, 'lag_1'] = df_result.loc[i - 1, 'actual_traffic_volume']
    #     df_result.loc[i, 'lag_2'] = df_result.loc[i - 1, 'lag_1']
    #     df_result.loc[i, 'lag_3'] = df_result.loc[i - 1, 'lag_2']

    # for i in range(1, len(df_result)):
    #     if pd.isna(df_result.loc[i, 'date_time']):
    #         df_result.loc[i, 'date_time'] = df_result.loc[i-1, 'date_time'] + pd.Timedelta(hours=1)

    # df_result = pd.concat([X, df_result], axis=1)

    # columns_to_drop = ['date_time', 'actual_traffic_volume', 'lag_1', 'lag_2', 'lag_3', 'forecasted_traffic_volume']
    # df_result = df_result.drop(columns=columns_to_drop)
    
    # X = df_result
    X = X.tail(3)
    print(df_result)
    # Create a SHAP Tree Explainer for the XGBoost model
    explainer = shap.TreeExplainer(joblib.load('best_xgboost_model.joblib'))

    # Calculate SHAP values - this might take some time for larger datasets
    shap_values = explainer.shap_values(X)

    return explainer, shap_values, X

def _force_plot_html(explainer, shap_values, ind, X):
    force_plot = shap.force_plot(explainer.expected_value, shap_values[ind, :], X.iloc[ind, :], matplotlib=False, figsize=(20, 8))
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html

@app.route('/xai')
def xai():
    explainer, shap_values, X = give_shap_plot()

    shap_plots = {}
    num_plots = min(3, len(X)) 
    for i in range(num_plots):
        ind = i
        shap_plots[i] = _force_plot_html(explainer, shap_values, ind, X)

    try:
        return render_template('xai.html', shap_plots=shap_plots)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "An error occurred while rendering the template."

def lime():
    global X, explainer, model
    # Define the query to fetch the last three rows
    query = "SELECT * FROM ( SELECT * FROM trainsql ORDER BY date_time DESC LIMIT 13 ) AS subquery ORDER BY date_time ASC;"
    #query = 'SELECT * FROM trainsql ORDER BY date_time ASC LIMIT 6'
    # Execute the query and load the result into a DataFrame
    df_raw = pd.read_sql(query, db)

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

    X = X.tail(3)
    # Create a LIME explainer object
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X), 
        feature_names=X.columns, 
        class_names=['traffic_volume'], 
        mode='regression'
    )

@app.route('/lime')
def lime_explanation():
    lime()  # Call the lime function to set up the necessary variables

    # Choose an instance to explain
    num_instances = 3
    lime_explanations = {}

    for i in range(num_instances):
        instance_index = i  # You may want to adjust this based on your dataset
        instance = X.iloc[instance_index]

        # Generate LIME explanation for this instance
        exp = explainer.explain_instance(
            data_row=instance, 
            predict_fn=model.predict
        )

        # Get the HTML representation of the LIME explanation
        lime_output = exp.as_html()

        # Store the LIME explanation in the dictionary
        lime_explanations[i] = lime_output

    try:
        return render_template('lime.html', lime_explanations=lime_explanations)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "An error occurred while rendering the template."


if __name__ == '__main__':
    app.run(debug=True)