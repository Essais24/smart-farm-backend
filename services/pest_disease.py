# services/pest_disease.py

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Risk Factor Functions
# -----------------------------
def temp_factor(T): return np.where((T >= 18) & (T <= 20), 1.0, 0.0)
def rh_factor(RH): return np.where(RH > 85, 1.0, 0.0)
def rain_factor(RF): return np.where(RF < 350, 1.0, 0.0)
def ndvi_factor(ndvi_val, ndvi_mean=0.55):
    drop = (ndvi_mean - ndvi_val) / ndvi_mean
    return np.where(drop > 0.1, 1.0, 0.0)
def redsi_factor(redsi_val, threshold=0.25):
    return np.where(redsi_val > threshold, 1.0, 0.0)
def nir_red_edge_factor(nir, red_edge, nir_thresh=0.6, red_edge_thresh=0.45):
    return np.where((nir < nir_thresh) & (red_edge < red_edge_thresh), 1.0, 0.0)
def sunshine_factor(sunhrs, max_sunhrs=10):
    return np.minimum(sunhrs / max_sunhrs, 1.0)
def crop_stage_factor_general(stage):
    return np.where((stage >= 20) & (stage <= 59), 1.0,
           np.where((stage >= 70) & (stage <= 79), 0.5, 0.0))

# -----------------------------
# Wheat Blast
# -----------------------------
def wheat_blast_temp_factor(temp_c): return np.where((temp_c >= 25) & (temp_c <= 30), 1.0, 0.0)
def wheat_blast_rh_factor(rh_percent): return np.where(rh_percent >= 90, 1.0, 0.0)
def wheat_blast_leafwetness_factor(leafwetness_hours): return np.where(leafwetness_hours > 6, 1.0, 0.0)
def wheat_blast_rain_factor(rain_mm): return np.where(rain_mm > 5, 1.0, 0.0)
def crop_stage_factor_blast(stage):
    return np.where((stage >= 50) & (stage <= 59), 1.0,
           np.where((stage >= 60) & (stage <= 69), 0.8,
           np.where((stage >= 40) & (stage < 50), 0.6, 0.2)))
def calculate_blast_risk(temp_c, rh_percent, leafwetness_hours, rain_mm, crop_stage):
    weights = {"temp": 0.30, "rh": 0.30, "leaf_wetness": 0.15, "rain": 0.10, "crop_stage": 0.15}
    return (weights["temp"] * wheat_blast_temp_factor(temp_c) +
            weights["rh"] * wheat_blast_rh_factor(rh_percent) +
            weights["leaf_wetness"] * wheat_blast_leafwetness_factor(leafwetness_hours) +
            weights["rain"] * wheat_blast_rain_factor(rain_mm) +
            weights["crop_stage"] * crop_stage_factor_blast(crop_stage))

# -----------------------------
# Sunn Pest
# -----------------------------
def sunn_pest_temp_factor(temp_c): return np.where(temp_c >= 12, 1.0, 0.0)
def sunn_pest_rain_factor(rain_mm): return np.where(rain_mm < 50, 1.0, 0.0)
def sunn_pest_relative_humidity_factor(rh_percent): return np.where((rh_percent >= 40) & (rh_percent <= 70), 1.0, 0.0)
def sunn_pest_crop_stage_factor(stage): return np.where((stage >= 50) & (stage <= 69), 1.0, 0.0)
def calculate_sunn_pest_risk(temp_c, rain_mm, rh_percent, crop_stage):
    weights = {"temperature": 0.30, "rainfall": 0.20, "relative_humidity": 0.25, "crop_stage": 0.25}
    return (weights["temperature"] * sunn_pest_temp_factor(temp_c) +
            weights["rainfall"] * sunn_pest_rain_factor(rain_mm) +
            weights["relative_humidity"] * sunn_pest_relative_humidity_factor(rh_percent) +
            weights["crop_stage"] * sunn_pest_crop_stage_factor(crop_stage))

# -----------------------------
# Unified Pixel-wise Risk Function
# -----------------------------
def check_risks_pixelwise(weather_data, indices_data, crop_stage):
    aphid_weights = {
        "temp": 0.15, "rh": 0.15, "rain": 0.10, "ndvi": 0.15,
        "redsi": 0.15, "nir_red_edge": 0.10, "sunshine": 0.05, "crop_stage": 0.15
    }

    aphid_risk = (
        aphid_weights["temp"] * temp_factor(weather_data["temperature"]) +
        aphid_weights["rh"] * rh_factor(weather_data["relative_humidity"]) +
        aphid_weights["rain"] * rain_factor(weather_data["rainfall"]) +
        aphid_weights["ndvi"] * ndvi_factor(indices_data["ndvi"]) +
        aphid_weights["redsi"] * redsi_factor(indices_data["redsi"]) +
        aphid_weights["nir_red_edge"] * nir_red_edge_factor(indices_data["nir"], indices_data["red_edge"]) +
        aphid_weights["sunshine"] * sunshine_factor(weather_data["sunshine_hours"]) +
        aphid_weights["crop_stage"] * crop_stage_factor_general(crop_stage)
    )
    aphid_alert = aphid_risk >= 0.6

    blast_risk = calculate_blast_risk(
        weather_data["temperature"], weather_data["relative_humidity"],
        weather_data["leafwetness_hours"], weather_data["rainfall"], crop_stage
    )
    blast_alert = blast_risk >= 0.6

    sunn_risk = calculate_sunn_pest_risk(
        weather_data["temperature"], weather_data["rainfall"],
        weather_data["relative_humidity"], crop_stage
    )
    sunn_alert = sunn_risk >= 0.6

    return {
        "aphid_risk": aphid_risk,
        "aphid_alert": aphid_alert,
        "blast_risk": blast_risk,
        "blast_alert": blast_alert,
        "sunn_risk": sunn_risk,
        "sunn_alert": sunn_alert
    }

# -----------------------------
# Main Pipeline
# -----------------------------
def pest_disease_pipeline(weather_data, indices_data, crop_stage):
    risks = check_risks_pixelwise(weather_data, indices_data, crop_stage)

    aphid_area = int(np.sum(risks["aphid_alert"]))
    blast_area = int(np.sum(risks["blast_alert"]))
    sunn_area = int(np.sum(risks["sunn_alert"]))

    return {
        "aphid_risk_map": risks["aphid_risk"].tolist(),
        "blast_risk_map": risks["blast_risk"].tolist(),
        "sunn_risk_map": risks["sunn_risk"].tolist(),
        "alerts": {
            "aphid_area": aphid_area,
            "blast_area": blast_area,
            "sunn_area": sunn_area
        }
    }
