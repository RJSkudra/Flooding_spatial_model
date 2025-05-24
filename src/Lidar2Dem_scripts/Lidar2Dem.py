"""
LiDAR to DEM processing script.
Handles LAS file merging, region selection, DEM generation, and caching.
"""

import os
import sys
import logging
import rasterio
import json
from src.utils.path_utils import get_dir
from src.FloodGIF_scripts.visualization_utils import select_region_of_interest
from src.utils.dem_utils import load_dem, save_dem_as_jpeg
import shutil
import numpy as np
from scipy.interpolate import griddata
# Import custom rendering configuration
import matplotlib
from src.config.rendering_config import setup_rendering
from src.Lidar2Dem_scripts.las_utils import merge_las_files, extract_las_metadata
from src.Lidar2Dem_scripts.pdal_pipeline import run_pdal_pipeline
from src.Lidar2Dem_scripts.georeference import ensure_georeferencing

def fill_dem_nans(dem_array):
    x, y = np.meshgrid(np.arange(dem_array.shape[1]), np.arange(dem_array.shape[0]))
    valid_mask = ~np.isnan(dem_array)
    filled = dem_array.copy()
    filled[np.isnan(dem_array)] = griddata(
        (x[valid_mask], y[valid_mask]),
        dem_array[valid_mask],
        (x[np.isnan(dem_array)], y[np.isnan(dem_array)]),
        method='cubic'
    )
    # Fallback for any remaining NaNs
    still_nan = np.isnan(filled)
    if np.any(still_nan):
        filled[still_nan] = griddata(
            (x[valid_mask], y[valid_mask]),
            dem_array[valid_mask],
            (x[still_nan], y[still_nan]),
            method='nearest'
        )
    return filled

def main():
    print("[DEBUG] Lidar2Dem: Turning LIDAR data to DEM...")
    # Configure matplotlib rendering before importing pyplot
    config = setup_rendering(matplotlib)

    # Ensure QT is used as the matplotlib backend
    try:
        matplotlib.use('QtAgg')
    except Exception as backend_error:
        matplotlib.use('Agg')

    # Magic numbers/constants
    DEFAULT_RESOLUTION = 0.25
    CACHE_DIRNAME = "tmp"
    RAW_DEM_FILENAME = "raw_dtm.tif"

    logger = logging.getLogger(__name__)

    # Definējam lidar_data direktoriju
    lidar_data_dir = os.path.join(get_dir("dati"), "lidar_dati")

    # Atrodam visus LAS failus lidar_data direktorijā
    las_files = []
    try:
        if os.path.exists(lidar_data_dir):
            for file in os.listdir(lidar_data_dir):
                if file.lower().endswith('.las'):
                    las_path = os.path.join(lidar_data_dir, file)
                    las_files.append(las_path)
    except Exception as e:
        logger.error(f"LAS failu meklēšanas kļūda: {e}")



    # Pārliecināmies, ka mums ir vismaz viens fails apstrādei
    if not las_files:
        raise FileNotFoundError("Nav atrasti LAS faili norādītajās vietās")

    # Definējam ceļu apvienotajam LAS failam
    merged_las_file = os.path.join(os.path.dirname(lidar_data_dir), "temp_merged.las")

    # Apvienojam LAS failus pirms vizualizācijas, ja atrasti vairāki faili
    if len(las_files) > 1:
        las_file_for_selection = merge_las_files(las_files, merged_las_file)
    else:
        las_file_for_selection = las_files[0]

    # --- Region selection and output file naming ---
    print("[INFO] Attempting to open region selection GUI...")
    selected_region = select_region_of_interest(las_file_for_selection)
    print(f"[INFO] Region selection GUI closed successfully. Selected region: {selected_region}")
    if not selected_region or not all(k in selected_region for k in ["minx", "maxx", "miny", "maxy"]):
        print("[ERROR] Invalid region selected. Exiting.")
        return
    print("[DEBUG] Proceeding to DEM generation...")

    # Generate a unique string for extents
    ext_str = f"{selected_region['minx']:.2f}_{selected_region['miny']:.2f}_{selected_region['maxx']:.2f}_{selected_region['maxy']:.2f}"
    DEM_OUTPUT_FILENAME = f"hydro_dem_output_{ext_str}.tif"
    output_file = os.path.join(get_dir("dati"), "dem_faili", DEM_OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Definējam ceļu pagaidu failiem tajā pašā mapē kā izvades fails
    temp_dem_file = os.path.join(os.path.dirname(output_file), "temp_dtm.tif")
    raw_dem_file = os.path.join(os.path.dirname(output_file), RAW_DEM_FILENAME)

    try:
        print(f"[DEBUG] Entering DEM caching/generation logic. Output file: {output_file}")
        # --- CACHING LOGIC START ---
        # Generate a unique cache filename based on extents
        cache_dir = os.path.join(get_dir("dati"), CACHE_DIRNAME)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"dem_cache_{ext_str}.tif")

        if os.path.exists(cache_file):
            print(f"[DEBUG] Cache file found: {cache_file}. Copying to output.")
            shutil.copy2(cache_file, output_file)
            las_metadata = extract_las_metadata(las_file_for_selection)
            ensure_georeferencing(output_file, las_metadata)
            # Generate JPEG preview after copying from cache
            dem_data, dem_array, _ = load_dem(output_file)
            # Fill DEM holes with NaN before saving preview
            dem_array = np.where(dem_array == -9999, np.nan, dem_array)
            jpeg_preview_path = os.path.join(os.path.dirname(output_file), "dem_preview.jpeg")
            # Calculate extent for precise axis coordinates
            bounds = dem_data.bounds
            extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
            save_dem_as_jpeg(dem_array, "terrain", jpeg_preview_path, title="DEM Preview", extent=extent)
        else:
            print("[DEBUG] No cache file found. Running PDAL pipeline...")
            las_metadata = extract_las_metadata(las_file_for_selection)
            if las_metadata:
                las_metadata['bounds']['minx'] = selected_region['minx']
                las_metadata['bounds']['miny'] = selected_region['miny']
                las_metadata['bounds']['maxx'] = selected_region['maxx']
                las_metadata['bounds']['maxy'] = selected_region['maxy']
            crop_filter = {
                "type": "filters.crop",
                "bounds": f"([{selected_region['minx']},{selected_region['maxx']}],[{selected_region['miny']},{selected_region['maxy']}])"
            }
            pipeline_json = {
                "pipeline": [
                    *[{"type": "readers.las", "filename": file} for file in las_files],
                    *([] if len(las_files) <= 1 else [{"type": "filters.merge"}]),
                    crop_filter,
                    {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 2.2},
                    {"type": "filters.elm", "cell": 1.0, "threshold": 0.3},
                    {"type": "filters.smrf", "window": 12, "slope": 0.2, "threshold": 0.15, "cell": 0.5, "ignore": "Classification[7:7]"},
                    {"type": "filters.pmf", "max_window_size": 25, "slope": 0.2, "initial_distance": 0.1, "max_distance": 2.0},
                    {"type": "filters.range", "limits": "Classification[2:2]"},
                    {"type": "writers.gdal", "resolution": DEFAULT_RESOLUTION, "output_type": "idw", "radius": 2.5, "power": 2.0, "window_size": 6, "filename": raw_dem_file, "gdalopts": "COMPRESS=DEFLATE,PREDICTOR=2,ZLEVEL=9", "nodata": -9999, "default_srs": "EPSG:3059"}
                ]
            }
            print(f"[DEBUG] PDAL pipeline JSON: {json.dumps(pipeline_json, indent=2)}")
            try:
                run_pdal_pipeline(pipeline_json)
                print(f"[DEBUG] PDAL pipeline finished. Checking for output: {raw_dem_file}")
            except Exception as pdal_exc:
                print(f"[ERROR] PDAL pipeline failed: {pdal_exc}")
                raise
            print("PDAL caurules apstrāde pabeigta.")
            print(f"Apstrādājam DEM failu: {raw_dem_file}")
            ensure_georeferencing(raw_dem_file, las_metadata)
            print("Georeference faila nodrošināšana pabeigta.")
            try:
                # Load DEM as array
                with rasterio.open(raw_dem_file) as src:
                    dem_array = src.read(1)
                    profile = src.profile

                nodata = profile.get('nodata', -9999)
                dem_array = np.where(dem_array == nodata, np.nan, dem_array)

                # Fill holes
                dem_array_filled = fill_dem_nans(dem_array)

                # Save filled DEM to output_file and cache_file
                with rasterio.open(output_file, 'w', **profile) as dst:
                    out_array = np.where(np.isnan(dem_array_filled), nodata, dem_array_filled)
                    dst.write(out_array, 1)
                ensure_georeferencing(output_file, las_metadata)

                with rasterio.open(cache_file, 'w', **profile) as dst:
                    dst.write(out_array, 1)
                # Generate JPEG preview after DEM creation
                
                dem_data, dem_array, _ = load_dem(output_file)
                # Fill DEM holes with NaN before saving preview
                dem_array = np.where(dem_array == -9999, np.nan, dem_array)
                jpeg_preview_path = os.path.join(os.path.dirname(output_file), "dem_preview.jpeg")
                bounds = dem_data.bounds
                extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
                save_dem_as_jpeg(dem_array, "terrain", jpeg_preview_path, title="DEM Preview", extent=extent)
            except Exception as copy_error:
                logger.error(f"DEM kopēšanas kļūda: {copy_error}")
                try:
                    with open(raw_dem_file, 'rb') as src_file, open(output_file, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                    ensure_georeferencing(output_file, las_metadata)
                    shutil.copy2(raw_dem_file, cache_file)
                    # Generate JPEG preview after fallback copy
                    dem_data, dem_array, _ = load_dem(output_file)
                    # Fill DEM holes with NaN before saving preview
                    dem_array = np.where(dem_array == -9999, np.nan, dem_array)
                    jpeg_preview_path = os.path.join(os.path.dirname(output_file), "dem_preview.jpeg")
                    bounds = dem_data.bounds
                    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
                    save_dem_as_jpeg(dem_array, "terrain", jpeg_preview_path, title="DEM Preview", extent=extent)
                except Exception as alt_copy_error:
                    logger.error(f"Alternatīvās kopēšanas kļūda: {alt_copy_error}")
            if len(las_files) > 1 and os.path.exists(merged_las_file):
                try:
                    os.remove(merged_las_file)
                except Exception:
                    pass
        # --- CACHING LOGIC END ---
    except Exception as e:
        print(f"Apstrādes kļūda: {e}")
        print(f"[ERROR] Region selection GUI failed to open or crashed: {e}")
        if 'temp_dem_file' in locals() and os.path.exists(temp_dem_file):
            try:
                os.remove(temp_dem_file)
            except:
                pass
        
if __name__ == '__main__':
    main()