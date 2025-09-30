import numpy as np

def wheat_kc(das, ndvi_pixel):
    """
    Compute pixel-wise Kc for wheat using days after sowing and NDVI
    """
    if das <= 20:
        Kc_min, Kc_max = 0.3, 0.4
        NDVI_min, NDVI_max = 0.1, 0.3
    elif das <= 45:
        Kc_min, Kc_max = 0.4, 1.15
        NDVI_min, NDVI_max = 0.2, 0.6
    elif das <= 80:
        Kc_min, Kc_max = 1.15, 1.2
        NDVI_min, NDVI_max = 0.5, 0.8
    else:  # 81-120 DAS
        Kc_min, Kc_max = 0.7, 1.0
        NDVI_min, NDVI_max = 0.3, 0.6

    Kc_pixel = Kc_min + (Kc_max - Kc_min) * np.clip((ndvi_pixel - NDVI_min) / (NDVI_max - NDVI_min), 0, 1)
    return Kc_pixel

def irrigation_pipeline(ndvi, ndwi, days_after_sowing, daily_ET0, daily_rain, pixel_size=10):
    """
    Returns pixel-wise irrigation requirement maps in mm and liters.
    Args:
        ndvi: 2D array of NDVI values
        ndwi: 2D array of stress factor (0-1)
        days_after_sowing: int
        daily_ET0: 1D array of daily reference evapotranspiration (mm/day)
        daily_rain: 1D array of daily rainfall (mm/day)
        pixel_size: size of Sentinel-2 pixel in meters (default 10m)
    Returns:
        irrigation_maps_mm: 3D array (days x height x width) in mm
        irrigation_maps_L: 3D array (days x height x width) in liters per pixel
        total_irrigation_L_per_day: list of total irrigation in liters for each day
    """
    kc_map = wheat_kc(days_after_sowing, ndvi)
    stress_factor = np.clip(ndwi, 0, 1)

    irrigation_maps_mm = []
    for i, et0 in enumerate(daily_ET0):
        ETc_pixel = et0 * kc_map * stress_factor
        irrigation_pixel = np.maximum(ETc_pixel - daily_rain[i], 0)
        irrigation_maps_mm.append(irrigation_pixel)

    irrigation_maps_mm = np.stack(irrigation_maps_mm, axis=0)

    pixel_area = pixel_size ** 2
    irrigation_maps_L = irrigation_maps_mm * pixel_area
    total_irrigation_L_per_day = [np.sum(day_map) for day_map in irrigation_maps_L]

    return irrigation_maps_mm, irrigation_maps_L, total_irrigation_L_per_day
