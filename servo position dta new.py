import pandas as pd
import numpy as np
import datetime
import pytz
from pvlib import solarposition

# --- Location ---
latitude = 28.6139  # Greater Noida
longitude = 77.2090
timezone = pytz.timezone('Asia/Kolkata')

# --- Time Range for Dataset ---
start_date = datetime.datetime(2024, 1, 1, 7, 0, 0, tzinfo=timezone)
end_date = datetime.datetime(2024, 12, 31, 19, 0, 0, tzinfo=timezone)
time_interval = datetime.timedelta(minutes=30)

# --- List to Store Data ---
data = []
current_time = start_date

while current_time <= end_date:
    times = pd.DatetimeIndex([current_time], tz=timezone)
    solpos = solarposition.get_solarposition(times, latitude, longitude)

    if not solpos.empty:
        azimuth = solpos['azimuth'][0]

        # Simulate environmental data
        temperature = 20 + 15 * np.sin(2 * np.pi * current_time.timetuple().tm_yday / 365 + np.pi / 2) + 5 * np.random.randn()
        humidity = 60 + 20 * np.cos(2 * np.pi * current_time.timetuple().tm_yday / 365 + np.pi) + 10 * np.random.randn()
        humidity = np.clip(humidity, 0, 100) # Ensure humidity is within a realistic range
        wind_speed = 2 + 3 * np.random.rand()
        wind_direction = np.random.randint(0, 360)

        # Calculate optimal angle based on azimuth for a horizontal axis tracker
        optimal_angle = np.clip(90 + (azimuth - 180) * 0.5, 0, 180)

        # For two servos controlling the same axis, we'll make the target angles very similar
        optimal_angle_1 = optimal_angle + np.random.normal(0, 1)
        optimal_angle_2 = optimal_angle - np.random.normal(0, 1)

        data.append([
            current_time,
            temperature,
            humidity,
            wind_speed,
            wind_direction,
            optimal_angle_1,
            optimal_angle_2,
            solpos['elevation'][0],
            azimuth
        ])

    current_time += time_interval

# --- Create DataFrame ---
df = pd.DataFrame(data, columns=['Timestamp', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Wind Direction (degrees)', 'Optimal Angle Panel 1', 'Optimal Angle Panel 2', 'Solar Elevation', 'Solar Azimuth'])

# --- Save to CSV ---
df.to_csv('servo_position_dataset.csv', index=False)

print("servo_position_dataset.csv generated successfully for a single-axis tracker (azimuth tracking).")