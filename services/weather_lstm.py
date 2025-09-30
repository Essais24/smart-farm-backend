# services/weather_lstm.py

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import requests_cache
import openmeteo_requests
from retry_requests import retry
import matplotlib.pyplot as plt
import io
import base64

# -----------------------------
# Load Model + Scaler
# -----------------------------
MODEL_PATH = "models/weather_lstm_model.h5"
SCALER_PATH = "models/weather_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Weather API setup
# -----------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# -----------------------------
# Utility functions
# -----------------------------
def dew_point(T, RH):
    return T - ((100 - RH) / 5)

def create_sequences(dataset, time_step=30):
    X = []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step)])
    return np.array(X)

# -----------------------------
# Thresholds for anomalies
# -----------------------------
THRESHOLDS = {
    "drought": {
        "rainfall_mm": 1.0,
        "soil_temp_high": 35
    },
    "thunderstorm": {
        "rainfall_mm": 30
    },
    "heatwave": {
        "air_temp_high": 40
    },
    "frost": {
        "air_temp_low": 2
    }
}

def detect_anomalies(pred_df):
    alerts = []
    for _, row in pred_df.iterrows():
        rain = row['rain (mm)']
        air_temp = row['temperature_2m (°C)']
        soil_temp = row['soil_temperature_18cm (°C)']

        if rain < THRESHOLDS["drought"]["rainfall_mm"] and soil_temp > THRESHOLDS["drought"]["soil_temp_high"]:
            alerts.append((row['date'], "⚠️ Drought Risk"))

        if rain > THRESHOLDS["thunderstorm"]["rainfall_mm"]:
            alerts.append((row['date'], "⛈️ Thunderstorm Risk"))

        if air_temp > THRESHOLDS["heatwave"]["air_temp_high"]:
            alerts.append((row['date'], "🔥 Heatwave Alert"))

        if air_temp < THRESHOLDS["frost"]["air_temp_low"]:
            alerts.append((row['date'], "❄️ Frost Risk"))
    return alerts

# -----------------------------
# Main pipeline
# -----------------------------
def predict_weather_pipeline(bbox, days_since_sowing, points=None, time_step=30):
    """
    bbox: [min_lon, min_lat, max_lon, max_lat]
    days_since_sowing: int
    points: optional list of [lat, lon]
    """
    centroid_lat = (bbox[1] + bbox[3]) / 2
    centroid_lon = (bbox[0] + bbox[2]) / 2

    # time range: last N days
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=days_since_sowing)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": centroid_lat,
        "longitude": centroid_lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "soil_temperature_7_to_28cm",
            "wind_speed_10m",
            "sunshine_duration",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Asia/Kolkata"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["rain (mm)"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["temperature_2m (°C)"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["soil_temperature_18cm (°C)"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["wind_speed_10m (km/h)"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["sunshine_duration"] = hourly.Variables(5).ValuesAsNumpy()
    hourly_data["et0 (mm)"] = hourly.Variables(6).ValuesAsNumpy()

    df_hist = pd.DataFrame(hourly_data)
    df_hist["dew_point"] = df_hist.apply(
        lambda r: dew_point(r["temperature_2m (°C)"], r["relative_humidity_2m"]), axis=1
    )

    # --- Prepare data for LSTM ---
    df_input = pd.DataFrame(
        df_hist[["rain (mm)", "temperature_2m (°C)", "soil_temperature_18cm (°C)"]].tail(24*time_step).values
    )
    X_input_scaled = scaler.transform(df_input)
    X = create_sequences(X_input_scaled, time_step)

    prediction_scaled = model.predict(X)
    prediction = scaler.inverse_transform(prediction_scaled)

    # Future timestamps
    last_time = df_hist["date"].iloc[-1]
    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=len(prediction),
        freq="H"
    )

    # Build results DataFrame
    pred_df = pd.DataFrame({
        "date": future_times,
        "rain (mm)": prediction[:, 1],
        "temperature_2m (°C)": prediction[:, 0],
        "soil_temperature_18cm (°C)": prediction[:, 2]
    })

    air_temp_pred = pred_df["temperature_2m (°C)"]
    soil_temp_pred = pred_df["soil_temperature_18cm (°C)"]
    rain_pred = pred_df["rain (mm)"]

    # Convert datetime to string for x-axis labels
    future_times_str = pred_df["date"].dt.strftime("%Y-%m-%d %H:%M")

    # Create figure
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(future_times_str, air_temp_pred, marker='o', label='Predicted Air Temp (°C)')
    ax.plot(future_times_str, soil_temp_pred, marker='x', label='Predicted Soil Temp (°C)')
    ax.plot(future_times_str, rain_pred, marker='s', label='Predicted Rainfall (mm)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Predicted Values')
    ax.set_title('LSTM Predictions for next 30 Hours')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    plt.close(fig)
    # Detect anomalies
    alerts = detect_anomalies(pred_df)

    return {
        "prediction_plot": img_base64,
        "alerts": [{"time": str(t), "message": msg} for t, msg in alerts]
    }