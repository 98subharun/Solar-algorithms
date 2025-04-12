import pandas as pd
import numpy as np
import serial
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import datetime
import pytz
from pvlib import solarposition
import pvlib
import pandas as pd 
import joblib  # For saving the scaler

# --- Serial Port Configuration ---
SERIAL_PORT = 'COM6'  # Replace with your ESP32's serial port
BAUD_RATE = 115200

# --- File Names for Model and Scaler ---
MODEL_FILE = 'solar_tracker_model.h5'
SCALER_FILE = 'feature_scaler.joblib'

# --- Define Features and Targets for Training ---
FEATURES = [
    'time_of_day_sin',
    'time_of_day_cos',
    'day_of_year_sin',
    'day_of_year_cos',
    'Temperature (°C)',
    'Humidity (%)',
    'Wind Speed (m/s)',
    'Wind Direction (degrees)',
    'Solar Azimuth'
]
TARGETS = ['Optimal Angle Panel 1', 'Optimal Angle Panel 2']

# --- Function to Create Neural Network Model ---
def create_model(input_shape, num_targets):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(num_targets, activation='linear')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# --- Function to Predict Angles ---
def predict_angles(model, scaler, current_time, current_day_of_year, temperature, humidity, wind_speed, wind_direction, solar_azimuth):
    max_seconds = 24 * 3600
    sin_time = np.sin(2 * np.pi * (current_time.hour * 3600 + current_time.minute * 60 + current_time.second) / max_seconds)
    cos_time = np.cos(2 * np.pi * (current_time.hour * 3600 + current_time.minute * 60 + current_time.second) / max_seconds)
    max_day = 365.25
    sin_day = np.sin(2 * np.pi * current_day_of_year / max_day)
    cos_day = np.cos(2 * np.pi * current_day_of_year / max_day)

    new_data = np.array([sin_time, cos_time, sin_day, cos_day, temperature, humidity, wind_speed, wind_direction, solar_azimuth]).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data)
    predicted_angles = model.predict(new_data_scaled)
    return predicted_angles[0][0], predicted_angles[0][1]

if __name__ == "__main__":
    # --- Training Section ---
    print("--- Neural Network Training ---")
    trained_model = None
    trained_scaler = None
    try:
        data = pd.read_csv('servo_position_dataset.csv')

        data['timestamp'] = pd.to_datetime(data['Timestamp'])
        data['time_of_day_seconds'] = data['timestamp'].dt.hour * 3600 + data['timestamp'].dt.minute * 60 + data['timestamp'].dt.second
        max_seconds = 24 * 3600
        data['sin_time'] = np.sin(2 * np.pi * data['time_of_day_seconds'] / max_seconds)
        data['cos_time'] = np.cos(2 * np.pi * data['time_of_day_seconds'] / max_seconds)

        data['day_of_year'] = data['timestamp'].dt.dayofyear
        max_day = 365.25
        data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / max_day)
        data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / max_day)

        X = data[FEATURES].values
        y = data[TARGETS].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = create_model(input_shape=(X_train.shape[1],), num_targets=len(TARGETS))

        epochs = 50
        batch_size = 32
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nMean Squared Error on Test Set: {loss:.4f}")
        print(f"Mean Absolute Error on Test Set: {mae:.4f}")

        model.save(MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print(f"\nModel training complete. Model saved to {MODEL_FILE} and scaler to {SCALER_FILE}.")

        trained_model = model
        trained_scaler = scaler

    except FileNotFoundError:
        print("Error: servo_position_dataset.csv not found. Please generate this file first.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

    # --- Real-time Prediction with ESP32 Data ---
    print("\n--- Real-time Prediction from ESP32 ---")
    if trained_model is not None and trained_scaler is not None:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
            print(f"Connected to ESP32 on {SERIAL_PORT}")

            greater_noida_tz = pytz.timezone('Asia/Kolkata')

            while True:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if line.startswith("S,"):
                            sensor_data = line[2:].split(',')
                            if len(sensor_data) == 4:
                                try:
                                    temperature = float(sensor_data[0])
                                    humidity = float(sensor_data[1])
                                    wind_speed = float(sensor_data[2])
                                    wind_direction = float(sensor_data[3])

                                    now_utc = datetime.datetime.now(pytz.utc)
                                    now_local = now_utc.astimezone(greater_noida_tz)
                                    current_day = now_local.timetuple().tm_yday

                                    times = pd_pvlib.DatetimeIndex([now_local], tz=greater_noida_tz)
                                    solpos = solarposition.get_solarposition(times, latitude=28.6139, longitude=77.2090)
                                    solar_azimuth = solpos['azimuth'][0]

                                    angle1, angle2 = predict_angles(trained_model, trained_scaler, now_local, current_day, temperature, humidity, wind_speed, wind_direction, solar_azimuth)

                                    angle_command = f"A,{angle1:.2f},{angle2:.2f}\n".encode('utf-8')
                                    ser.write(angle_command)
                                    print(f"Received Sensor Data: Temp={temperature:.2f}, Hum={humidity:.2f}, WindSpeed={wind_speed:.2f}, WindDir={wind_direction:.2f}")
                                    print(f"Calculated Solar Azimuth: {solar_azimuth:.2f}")
                                    print(f"Sent Predicted Angles: Servo1={angle1:.2f}, Servo2={angle2:.2f}")

                                except ValueError as e:
                                    print(f"Error converting sensor data: {e} - Data: {sensor_data}")
                            else:
                                print(f"Incomplete sensor data received: {line}")
                        else:
                            print(f"Received from ESP32: {line}")

                    except serial.SerialException as e:
                        print(f"Serial port error: {e}")
                        break
                    except UnicodeDecodeError as e:
                        print(f"Error decoding data from ESP32: {e}")

                time.sleep(0.1)

        except serial.SerialException as e:
            print(f"Error opening serial port {SERIAL_PORT}: {e}")
        except KeyboardInterrupt:
            print("\nClosing serial port and exiting.")
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()
                print("Serial port closed.")
    else:
        print("\nSkipping real-time prediction as the model was not trained successfully.")
