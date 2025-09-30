import numpy as np
import matplotlib.pyplot as plt

def normalize_index(index):
    """
    Normalize an index to 0-1 range using 2nd and 98th percentiles to avoid outliers.
    """
    min_val = np.percentile(index, 2)
    max_val = np.percentile(index, 98)
    return np.clip((index - min_val) / (max_val - min_val), 0, 1)

def nutrient_stress(ndvi, ndre, soil_moisture):
    """
    Compute pixel-wise nutrient stress factor (0=no stress, 1=high stress)
    Args:
        ndvi, ndre: 2D arrays of normalized indices
        soil_moisture: 2D array, normalized 0-1
    Returns:
        stress_adjusted: 2D array, nutrient stress factor adjusted for soil moisture
    """
    ndvi_norm = normalize_index(ndvi)
    ndre_norm = normalize_index(ndre)

    stress_factor = 1 - (0.5 * ndvi_norm + 0.5 * ndre_norm)
    stress_factor = np.clip(stress_factor, 0, 1)

    stress_adjusted = stress_factor * (1 - 0.5 * soil_moisture)
    stress_adjusted = np.clip(stress_adjusted, 0, 1)

    return stress_adjusted

def max_nitrogen_wheat(das):
    """
    Returns the max nitrogen dose (kg/ha) for wheat based on days after sowing.
    """
    if das <= 20:
        return 30  # Basal N
    elif das <= 45:
        return 40  # Top-dress N
    elif das <= 80:
        return 35  # Stem elongation / booting
    elif das <= 110:
        return 20  # Heading / grain filling
    else:
        return 0   # Late stage / maturity

def fertilizer_requirement(ndvi, ndre, soil_moisture, das):
    """
    Compute pixel-wise nitrogen requirement in kg/ha
    """
    stress_adjusted = nutrient_stress(ndvi, ndre, soil_moisture)
    max_nitrogen = max_nitrogen_wheat(das)
    return max_nitrogen * stress_adjusted

def plot_fertilizer_map(fertilizer_pixel, das):
    """
    Plot pixel-wise nitrogen requirement map.
    """
    plt.figure(figsize=(8,6))
    plt.imshow(fertilizer_pixel, cmap='YlGn')
    plt.colorbar(label='Nitrogen requirement (kg/ha)')
    plt.title(f'Pixel-wise Nitrogen Requirement for Wheat (DAS={das})')
    plt.axis('off')
    plt.show()
