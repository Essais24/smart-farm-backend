@app.get("/button2_indices")
def run_indices(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_date: str,
    end_date: str
):
    bbox = [min_lon, min_lat, max_lon, max_lat]
    return analyze_indices_pipeline(bbox, (start_date, end_date))