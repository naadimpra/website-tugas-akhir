from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
import math
import datetime
from datetime import datetime, timedelta

app = Flask(__name__)

# Configure MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="test_traffic"
)
cursor = db.cursor()

# Number of rows to display per page
ROWS_PER_PAGE = 10

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

@app.route('/')
def index():
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
    return render_template('index.html', data=data, columns=columns, page=page, rowsPerPage=rows_per_page, totalPages=total_pages,
                           totalRows=total_rows, filter_column=filter_column, filter_value=filter_value, sort_column=sort_column,
                           sort_order=sort_order, column_labels=column_labels)

@app.route('/insert', methods=['GET', 'POST'])
def insert():
    table_name = "trainsql"
    
    if request.method == 'POST':
        # Retrieve form data
        date_time = datetime.datetime.strptime(request.form['date_time'], '%Y-%m-%d %H:%M:%S')
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

        # Insert data into the MySQL table
        query = f"""
            INSERT INTO {table_name} 
            (date_time, is_holiday, air_pollution_index, humidity, wind_speed, wind_direction,
            visibility_in_miles, dew_point, temperature, rain_p_h, snow_p_h, clouds_all,
            weather_type, weather_description, traffic_volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (date_time, is_holiday, air_pollution_index, humidity, wind_speed, wind_direction,
                  visibility_in_miles, dew_point, temperature, rain_p_h, snow_p_h, clouds_all,
                  weather_type, weather_description, traffic_volume)
        print (values)
        cursor.execute(query, values)
        db.commit()

        # Redirect to the index page after successful insertion
        return redirect(url_for('index'))

    # If the request method is GET, render the insert.html template
    return render_template('insert.html')

@app.route('/chart')
def chart():
    # Connect to your database and fetch the most recent 24 records
    table_name = "trainsql"

    # Retrieve the 24 most recent records from MySQL table
    query = f"""
        SELECT date_time, traffic_volume
        FROM (
            SELECT date_time, traffic_volume
            FROM {table_name}
            ORDER BY date_time DESC
            LIMIT 24
        ) AS recent_data
        ORDER BY date_time ASC
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Extract relevant data for the chart (e.g., date_time and traffic_volume)
    chart_data = [(entry[0], entry[1]) for entry in data]

    return render_template('chart.html', chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)