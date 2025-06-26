import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path
from datetime import datetime

# ---- CONFIG ----
CSV_PATH = "../data/VIIRS_FIRMS/modis_2024_India.csv"
ERA5_TEMPLATE = "../data/era5_input.grib"
MASK_SAVE_DIR = "../data/viirs_masks"
RESOLUTION = 0.1  # degrees (match ERA5)
CONFIDENCE_THRESHOLD = 50

Path(MASK_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# ---- Load ERA5 Grid as Template ----
era5 = xr.open_dataset(ERA5_TEMPLATE, engine="cfgrib", backend_kwargs={"indexpath": ""})
lat = era5.latitude.values
lon = era5.longitude.values

# Create bounds from lat/lon grid
min_lon, max_lon = lon.min(), lon.max()
min_lat, max_lat = lat.min(), lat.max()

width = int((max_lon - min_lon) / RESOLUTION)
height = int((max_lat - min_lat) / RESOLUTION)
transform = from_origin(min_lon, max_lat, RESOLUTION, RESOLUTION)

# ---- Load Fire CSV ----
df = pd.read_csv(CSV_PATH)
df = df[df["confidence"] >= CONFIDENCE_THRESHOLD]
df["datetime"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4), format="%Y-%m-%d %H%M")
df["day"] = df["datetime"].dt.strftime("%Y%m%d")

# ---- Group By Day and Rasterize ----
for i, (day, group) in enumerate(df.groupby("day")):
    points = [Point(xy) for xy in zip(group.longitude, group.latitude)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    shapes = ((geom, 1) for geom in gdf.geometry)

    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Save as .npy with shape (1, H, W)
    np.save(f"{MASK_SAVE_DIR}/{i:04}.npy", mask[np.newaxis, :, :])

    print(f"✅ Saved mask for {day} as {i:04}.npy")

print("🎯 All masks saved to:", MASK_SAVE_DIR)

