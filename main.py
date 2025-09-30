from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Tuple

import io, base64
import numpy as np
import folium
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap

# -----------------------------
# Import your service pipelines
# -----------------------------
from services.weather import predict_weather_pipeline
from services.pest_disease import pest_disease_pipeline
from services.irrigation import irrigation_pipeline
from services.nutrient import fertilizer_map

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Initialize app
app = FastAPI(title="Smart Farm Dashboard")

# -----------------------------
# Request Models
# -----------------------------
class WeatherRequest(BaseModel):
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    days_since_sowing: int

class PestRequest(BaseModel):
    weather_data: dict
    indices_data: dict
    crop_stage: int
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

class IrrigationRequest(BaseModel):
    ndvi: list
    ndwi: list
    days_after_sowing: int
    daily_ET0: list
    daily_rain: list
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

class FertilizerRequest(BaseModel):
    ndvi: list
    ndre: list
    sm: list
    das: int
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

# -----------------------------
# Endpoint: Weather Predictions
# -----------------------------
@app.post("/button1_weather", response_class=HTMLResponse)
def run_weather(req: WeatherRequest):
    result = predict_weather_pipeline(
        bbox=req.bbox,
        days_since_sowing=req.days_since_sowing
    )

    # Extract matplotlib figure
    fig = result["figure"]  # Must be a Figure object

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    html = f'<h3>Weather Predictions</h3><img src="data:image/png;base64,{img_base64}" />'
    return html

# -----------------------------
# Endpoint: Pest/Disease Risk Maps
# -----------------------------
@app.post("/button2_pests", response_class=HTMLResponse)
def run_pests(req: PestRequest):
    results = pest_disease_pipeline(
        weather_data=req.weather_data,
        indices_data=req.indices_data,
        crop_stage=req.crop_stage,
        min_lat=req.min_lat,
        min_lon=req.min_lon,
        max_lat=req.max_lat,
        max_lon=req.max_lon
    )

    # Convert Folium maps to HTML
    aphid_html = results["aphid_map"]._repr_html_()
    blast_html = results["blast_map"]._repr_html_()
    sunn_html = results["sunn_map"]._repr_html_()

    combined_html = f"""
    <h3>Aphid Risk Map</h3>{aphid_html}
    <h3>Wheat Blast Risk Map</h3>{blast_html}
    <h3>Sunn Pest Risk Map</h3>{sunn_html}
    """
    return combined_html

# -----------------------------
# Endpoint: Irrigation Map
# -----------------------------
@app.post("/button3_irrigation", response_class=HTMLResponse)
def run_irrigation(req: IrrigationRequest):
    result = irrigation_pipeline(
        ndvi=np.array(req.ndvi),
        ndwi=np.array(req.ndwi),
        days_after_sowing=req.days_after_sowing,
        daily_ET0=np.array(req.daily_ET0),
        daily_rain=np.array(req.daily_rain),
        min_lat=req.min_lat,
        min_lon=req.min_lon,
        max_lat=req.max_lat,
        max_lon=req.max_lon
    )

    irrigation_map = result["irrigation_map"]  # Folium map
    return irrigation_map._repr_html_()

# -----------------------------
# Endpoint: Fertilizer Map
# -----------------------------
@app.post("/button4_fertilizer", response_class=HTMLResponse)
def run_fertilizer(req: FertilizerRequest):
    result = fertilizer_map(
        ndvi=np.array(req.ndvi),
        ndre=np.array(req.ndre),
        sm=np.array(req.sm),
        das=req.das,
        min_lat=req.min_lat,
        min_lon=req.min_lon,
        max_lat=req.max_lat,
        max_lon=req.max_lon
    )

    fertilizer_map_interactive = result["fertilizer_map"]  # Folium map
    return fertilizer_map_interactive._repr_html_()
