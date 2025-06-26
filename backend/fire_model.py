# U-Net model loading and prediction logic
import os
import xarray as xr

def merge_grib_parts(part1_path="../data/fire_dataset_part1.grib",
                     part2_path="../data/fire_dataset_part2.grib",
                     merged_path="../data/era5_input.grib"):
    """Merges two GRIB parts into a full GRIB file."""
    if os.path.exists(merged_path):
        print(f"✅ Merged GRIB file already exists: {merged_path}")
        return merged_path

    with open(merged_path, "wb") as outfile:
        for part in [part1_path, part2_path]:
            with open(part, "rb") as infile:
                outfile.write(infile.read())

    print(f"✅ GRIB file reconstructed at: {merged_path}")
    return merged_path


def load_era5_grib_dataset():
    """Loads ERA5 GRIB1 file into an xarray Dataset for training or inference."""
    grib_path = merge_grib_parts()

    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}
        )
        print("✅ ERA5 GRIB file loaded successfully.")
        return ds
    except Exception as e:
        print("❌ Failed to load GRIB file:", e)
        return None

if __name__ == "__main__":
    ds = load_era5_grib_dataset()
    if ds is not None:
        print("Variables available in dataset:", list(ds.data_vars))
        print("Dataset dimensions:", ds.dims)

