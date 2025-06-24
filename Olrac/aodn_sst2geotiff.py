import xarray as xr
import s3fs
import os
from datetime import datetime, timedelta
from pyproj import Geod
import numpy as np
import rioxarray
import rasterio


# Mount the S3 bucket as a FS
fs = s3fs.S3FileSystem(anon=True)

def find_file(fs, desired_date=None):
    imos_s3_bucket_prefix = 'imos-data'
    if desired_date:
        year = int(desired_date[:4])
    else:
        desired_date = datetime.today().strftime('%Y%m%d')
        year = datetime.today().year

    data_dir = f'IMOS/SRS/SST/ghrsst/L3SM-6d/dn/{year}/'
    fs_data_files = fs.ls(f'{imos_s3_bucket_prefix}/{data_dir}')
    filenames = [os.path.basename(file) for file in fs_data_files]

    file_found = False
    desired_date_dt = datetime.strptime(desired_date, '%Y%m%d')

    while not file_found:
        formatted_date = desired_date_dt.strftime('%Y%m%d')
        required_files = [file for file in filenames if file.startswith(formatted_date)]
        
        if required_files:
            file_found = True
            print(f'File Found: {required_files[0]}')
            return f'{imos_s3_bucket_prefix}/{data_dir}{required_files[0]}', formatted_date
        else:
            desired_date_dt -= timedelta(days=1)

    return None, None

def extract_product_name(filename):
    parts = filename.split('-')
    product_name = f"{parts[-3]}-{parts[-2]}-{parts[-1].split('.')[0]}"
    return product_name

def create_bounding_box(lon, lat, distance_km):
    # Create a Geod object
    geod = Geod(ellps='WGS84')

    # Calculate the north-south distance
    _, lat_min, _ = geod.fwd(lon, lat, 180, distance_km * 1000)
    _, lat_max, _ = geod.fwd(lon, lat, 0, distance_km * 1000)

    # Calculate the east-west distance
    lon_min, _, _ = geod.fwd(lon, lat, 270, distance_km * 1000)
    lon_max, _, _ = geod.fwd(lon, lat, 90, distance_km * 1000)

    return lon_min, lat_min, lon_max, lat_max

def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename
    base, ext = os.path.splitext(base_filename)
    i = 1
    while True:
        new_filename = f"{base}({i}){ext}"
        if not os.path.exists(new_filename):
            return new_filename
        i += 1

def export_geotiff(dataset, desired_date, filename, bounds):
    product_name = extract_product_name(filename)
    # Define your bounding box
    lon_min, lat_min, lon_max, lat_max = bounds

    # Convert sea surface temperature from Kelvin to Celsius
    dataset -= 273.15

    # Write CRS
    rio_dataset = dataset.rio.write_crs("EPSG:4326")

    # Extract and update metadata
    metadata = dataset.attrs.copy()
    metadata['units'] = 'Celsius'

    # Export to GeoTIFF
    output_file = f"SST_{desired_date}_{product_name}_{lat_min:.2f}_{lat_max:.2f}_{lon_min:.2f}_{lon_max:.2f}.tif"
    
    # Get a unique filename if the file already exists
    unique_output_file = get_unique_filename(output_file)

    # Export to GeoTIFF without compression
    rio_dataset.rio.to_raster(unique_output_file)

    # Reopen the GeoTIFF file to update the metadata
    with rasterio.open(unique_output_file, 'r+') as dst:
        dst.update_tags(**metadata)

    print(f"GeoTIFF saved as {unique_output_file}")


def process_sst_data(desired_date=None, lon=144.064, lat=-39.858, distance_km=50, shapefile=None):
    # Generate the bounding box
    bounds = create_bounding_box(lon, lat, distance_km)
    print(f"Bounding box: {bounds}")

    # Find the desired file
    s3path, formatted_date = find_file(fs, desired_date)

    if s3path:
        # Open the NetCDF file from the S3 bucket using fs.open
        with fs.open(s3path, 'rb') as f:
            # Open the dataset with xarray
            print("1. Opening Dataset...")
            data = xr.open_dataset(f, engine='h5netcdf', use_cftime=True)[['lat', 'lon', 'sea_surface_temperature']]

            # Extract the necessary variables
            latitudes = data['lat']
            longitudes = data['lon']
            sea_surface_temperature = data['sea_surface_temperature']

            # Subset the dataset to the bounding box
            subset_data = sea_surface_temperature.where((latitudes >= bounds[1]) & (latitudes <= bounds[3]) & (longitudes >= bounds[0]) & (longitudes <= bounds[2]), drop=True)

            # Drop NaN values
            subset_data = subset_data.dropna(dim='lat', how='all')
            subset_data = subset_data.dropna(dim='lon', how='all')

            print("2. Exporting to Geotiff...")
            # Export the masked data to a GeoTIFF
            export_geotiff(subset_data, formatted_date, s3path, bounds)

    else:
        print("No file found for the specified date.")
