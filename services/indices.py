# services/indices.py

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sentinelhub import SHConfig, BBox, CRS, DataCollection
from eolearn.core import EOTask, FeatureType
from eolearn.io import SentinelHubInputTask

# ✅ Config for SentinelHub
config = SHConfig()
config.sh_client_id = "YOUR_CLIENT_ID"
config.sh_client_secret = "YOUR_CLIENT_SECRET"


# ----- EO-Learn Tasks -----
class SclCloudMaskTask(EOTask):
    def execute(self, eopatch):
        scl_band = eopatch.data['BANDS'][..., -1]
        cloud_classes = [3, 8, 9, 10, 11]  # Cloud shadow, clouds, cirrus, snow
        cloud_mask = np.isin(scl_band, cloud_classes)
        eopatch.mask['CLM'] = np.expand_dims(cloud_mask, axis=-1).astype(bool)
        return eopatch


class CalculateIndicesTask(EOTask):
    def execute(self, eopatch):
        np.seterr(divide='ignore', invalid='ignore')
        bands = eopatch.data['BANDS'][..., :-1]

        blue, green, red, re, nir, swir1, swir2 = (
            bands[..., 0], bands[..., 1], bands[..., 2],
            bands[..., 3], bands[..., 4], bands[..., 5], bands[..., 6]
        )

        # --- Vegetation indices ---
        ndvi = (nir - red) / (nir + red)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        ndre = (nir - re) / (nir + re)

        # --- Moisture indices ---
        msi = swir1 / nir
        ndwi = (nir - swir1) / (nir + swir1)

        # --- Soil organic carbon & OM ---
        bare_soil_mask = ndvi < 0.25
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
        soc = np.clip(2.25 - 1.75 * bsi, 0, 10)
        soc[~bare_soil_mask] = np.nan
        om = soc * 1.724

        # --- Soil moisture ---
        NDWI_min, NDWI_max = 0.1, 0.6
        SM_min, SM_max = 0.05, 0.4
        soil_moisture = np.clip((ndwi - NDWI_min) / (NDWI_max - NDWI_min), 0, 1)
        soil_moisture = soil_moisture * (SM_max - SM_min) + SM_min

        final_results = np.stack(
            [ndvi, evi, savi, ndre, msi, ndwi, soc, om, soil_moisture],
            axis=-1
        )
        eopatch.data['INDICES'] = final_results
        return eopatch


# ----- Main Pipeline Function -----
def analyze_indices_pipeline(bbox_coords, time_range):
    """
    bbox_coords: [min_lon, min_lat, max_lon, max_lat]
    time_range: (start_date, end_date) as strings
    """

    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

    load_data_task = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L2A,
        bands=['B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'B12', 'SCL'],
        bands_feature=(FeatureType.DATA, 'BANDS'),
        resolution=10,
        maxcc=0.5,
        config=config
    )

    scl_mask_task = SclCloudMaskTask()
    calculate_indices_task = CalculateIndicesTask()

    try:
        eopatch = load_data_task.execute(bbox=bbox, time_interval=time_range)
        if not eopatch.timestamp:
            return {"error": "No satellite images found for given bbox/time_range"}

        # Cloud mask + Indices
        eopatch = scl_mask_task.execute(eopatch)
        eopatch = calculate_indices_task.execute(eopatch)

        # Pick best (least cloudy) date
        cloud_coverage = eopatch.mask['CLM'].mean(axis=(1, 2, 3))
        best_idx = np.argmin(cloud_coverage)
        best_timestamp = eopatch.timestamp[best_idx]

        indices_data = eopatch.data['INDICES'][best_idx]
        ndvi, evi, savi, ndre, msi, ndwi, soc, om, sm = np.moveaxis(indices_data, -1, 0)

        return {
            "timestamp": str(best_timestamp),
            "NDVI_mean": float(np.nanmean(ndvi)),
            "EVI_mean": float(np.nanmean(evi)),
            "SAVI_mean": float(np.nanmean(savi)),
            "NDRE_mean": float(np.nanmean(ndre)),
            "MSI_mean": float(np.nanmean(msi)),
            "NDWI_mean": float(np.nanmean(ndwi)),
            "SOC_mean": float(np.nanmean(soc)),
            "OM_mean": float(np.nanmean(om)),
            "SoilMoisture_mean": float(np.nanmean(sm))
        }

    except Exception as e:
        return {"error": str(e)}
