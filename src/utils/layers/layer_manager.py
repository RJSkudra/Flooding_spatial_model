"""
Layer management utilities for WMS, shapefile, and LiDAR overlays.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import geopandas as gpd
import pyproj
import requests
import io
import matplotlib.image as mpimg

logger = logging.getLogger(__name__)

def load_wms_layers(config_path: str) -> Dict[str, Any]:
    """
    Load WMS layers configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing WMS layer configurations.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        layers = json.load(f)
    return layers

def get_plot_order(layers_dict: Dict[str, Any]) -> list:
    """
    Determine the order in which layers should be plotted.

    Args:
        layers_dict (Dict[str, Any]): Dictionary containing layer configurations.

    Returns:
        list: List of layer keys in the order they should be plotted.
    """
    return layers_dict.get("plot_order", [k for k in layers_dict if k != "plot_order"])

def preload_layers(
    layers_dict: Dict[str, Any],
    plot_order: list,
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    width: int = 1200,
    height: int = 1080
) -> Tuple[Dict[str, np.ndarray], Dict[str, gpd.GeoDataFrame]]:
    """
    Preload layers by fetching WMS images and filtering shapefiles.

    Args:
        layers_dict (Dict[str, Any]): Dictionary containing layer configurations.
        plot_order (list): List of layer keys in the order they should be plotted.
        minx (float): Minimum x-coordinate of the bounding box.
        maxx (float): Maximum x-coordinate of the bounding box.
        miny (float): Minimum y-coordinate of the bounding box.
        maxy (float): Maximum y-coordinate of the bounding box.
        width (int, optional): Width of the WMS image. Defaults to 1200.
        height (int, optional): Height of the WMS image. Defaults to 1080.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, gpd.GeoDataFrame]]: 
        Dictionary of WMS images and dictionary of shapefile geometries.
    """
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tmp'))
    os.makedirs(cache_dir, exist_ok=True)
    wms_images = {}
    shp_geoms = {}
    bbox = f"{minx},{miny},{maxx},{maxy}"
    for key in plot_order:
        if key not in layers_dict:
            continue
        layer = layers_dict[key]
        bbox_str = f"{minx:.2f}_{miny:.2f}_{maxx:.2f}_{maxy:.2f}"
        if layer.get('type') == 'shapefile':
            shp_cache_file = os.path.join(cache_dir, f"shp_{key}_{bbox_str}.gpkg")
            try:
                if os.path.exists(shp_cache_file):
                    gdf = gpd.read_file(shp_cache_file)
                    shp_geoms[key] = gdf
                else:
                    shp_path = layer['path']
                    if not os.path.isabs(shp_path) and not shp_path.startswith('/'):
                        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
                        shp_path = os.path.normpath(os.path.join(project_root, shp_path.lstrip('./').lstrip('/')))
                    gdf = gpd.read_file(shp_path)
                    print(f"[DEBUG] {key} - Loaded features before cropping: {len(gdf)}")
                    if 'filter' in layer:
                        field = layer['filter']['field']
                        values = layer['filter']['values']
                        gdf = gdf[gdf[field].isin(values)]
                        print(f"[DEBUG] {key} - Features after filter: {len(gdf)}")
                    gdf = gdf.cx[minx:maxx, miny:maxy]
                    print(f"[DEBUG] {key} - Features after cropping to bbox ({minx}, {maxx}, {miny}, {maxy}): {len(gdf)}")
                    # Fix: Drop 'fid' field if present to avoid GPKG error
                    if 'fid' in gdf.columns:
                        print(f"[DEBUG] {key} - Dropping 'fid' field before saving to GPKG to avoid error.")
                        gdf = gdf.drop(columns=['fid'])
                    gdf.to_file(shp_cache_file, driver='GPKG')
                    shp_geoms[key] = gdf
            except Exception as e:
                logger.error(f"Error loading SHP layer '{key}': {e}")
                shp_geoms[key] = None
        elif layer.get('type') == 'lidar':
            pass  # LIDAR handled in main code
        elif layer.get('type') == 'arcgis_tile':
            # ArcGIS tile layer handling with correct EPSG:3059 tile calculation
            wms_cache_file = os.path.join(cache_dir, f"wms_{key}_{bbox_str}.npy")
            try:
                if os.path.exists(wms_cache_file):
                    img = np.load(wms_cache_file)
                    wms_images[key] = img
                else:
                    from PIL import Image
                    # ArcGIS tile info from service metadata
                    tile_size = 512
                    z = 13  # or make dynamic if needed
                    origin_x = -5120900
                    origin_y = 3998100
                    # Level 13 resolution from metadata
                    resolution = 0.5291677250021167
                    # Calculate tile indices for min/max
                    def lks97_to_tile(x, y, z):
                        tx = int((x - origin_x) / (tile_size * resolution))
                        ty = int((origin_y - y) / (tile_size * resolution))
                        return tx, ty
                    tx_min, ty_max = lks97_to_tile(minx, miny, z)
                    tx_max, ty_min = lks97_to_tile(maxx, maxy, z)
                    print(f"[DEBUG] EPSG:3059 bbox: {minx},{miny},{maxx},{maxy}")
                    print(f"[DEBUG] Tile x range: {tx_min} to {tx_max}, y range: {ty_min} to {ty_max}, zoom: {z}")
                    # Ensure correct order
                    tx_min, tx_max = min(tx_min, tx_max), max(tx_min, tx_max)
                    ty_min, ty_max = min(ty_min, ty_max), max(ty_min, ty_max)
                    # Download and stitch tiles
                    tiles_x = tx_max - tx_min + 1
                    tiles_y = ty_max - ty_min + 1
                    print(f"[DEBUG] Downloading {tiles_x} x {tiles_y} tiles...")
                    stitched = Image.new('RGBA', (tiles_x * tile_size, tiles_y * tile_size))
                    for ix, tx in enumerate(range(tx_min, tx_max + 1)):
                        for iy, ty in enumerate(range(ty_min, ty_max + 1)):
                            tile_url = layer["url"].replace("{z}", str(z)).replace("{x}", str(tx)).replace("{y}", str(ty))
                            print(f"[DEBUG] Fetching tile: {tile_url}")
                            resp = requests.get(tile_url, timeout=10)
                            if resp.status_code == 200:
                                tile_img = Image.open(io.BytesIO(resp.content)).convert('RGBA')
                                stitched.paste(tile_img, (ix * tile_size, iy * tile_size))
                            else:
                                print(f"[DEBUG] Tile not found: {tile_url} (status {resp.status_code})")
                    # Calculate pixel coordinates of bbox in stitched image
                    def lks97_to_pixel(x, y, z, tx_min, ty_min):
                        px = ((x - origin_x) / resolution) - (tx_min * tile_size)
                        py = ((origin_y - y) / resolution) - (ty_min * tile_size)
                        return int(px), int(py)
                    px_min, py_max = lks97_to_pixel(minx, miny, z, tx_min, ty_min)
                    px_max, py_min = lks97_to_pixel(maxx, maxy, z, tx_min, ty_min)
                    left, upper = min(px_min, px_max), min(py_min, py_max)
                    right, lower = max(px_min, px_max), max(py_min, py_max)
                    cropped = stitched.crop((left, upper, right, lower)).resize((width, height), Image.LANCZOS)
                    img = np.array(cropped)
                    print(f"[DEBUG] ArcGIS tile image shape: {img.shape}, dtype: {img.dtype}")
                    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tmp'))
                    stitched.save(os.path.join(debug_dir, f"arcgis_stitched_{key}_epsg3059.png"))
                    cropped.save(os.path.join(debug_dir, f"arcgis_cropped_{key}_epsg3059.png"))
                    np.save(wms_cache_file, img)
                    wms_images[key] = img
            except Exception as e:
                logger.error(f"Error loading ArcGIS tile layer '{key}': {e}")
                wms_images[key] = None
        elif layer.get('type') == 'arcgis_feature':
            # ArcGIS feature layer (vector, for calculations)
            shp_cache_file = os.path.join(cache_dir, f"shp_{key}_{bbox_str}.gpkg")
            try:
                gdf = None
                if os.path.exists(shp_cache_file):
                    try:
                        gdf = gpd.read_file(shp_cache_file)
                        shp_geoms[key] = gdf
                    except Exception as read_err:
                        print(f"[DEBUG] Failed to read cached GPKG for {key}: {read_err}. Deleting and re-downloading.")
                        os.remove(shp_cache_file)
                        gdf = None
                if gdf is None:
                    from shapely.geometry import shape
                    # ArcGIS query endpoint
                    base_url = "https://www.melioracija.lv/proxy/93B0862CB25D4851A5872F802F65FE0D/MKGISCacheLKS/MapServer"
                    if 'url' in layer:
                        # Only strip if the url ends with /<number> (layer id)
                        import re
                        m = re.match(r'(.+/MapServer)(/\\d+)?$', layer['url'])
                        if m:
                            base_url = m.group(1)
                        else:
                            base_url = layer['url']
                        print(f"[DEBUG] Using custom ArcGIS base_url for {key}: {base_url}")
                    layer_id = layer["layer_id"]
                    query_url = f"{base_url}/{layer_id}/query"  # <-- Add this line
                    bbox_str_query = f"{minx},{miny},{maxx},{maxy}"
                    params = {
                        "geometry": bbox_str_query,
                        "geometryType": "esriGeometryEnvelope",
                        "spatialRel": "esriSpatialRelIntersects",
                        "outFields": "*",
                        "returnGeometry": "true",
                        "f": "geojson",
                        "inSR": 3059,  # Tell the server the bbox is in EPSG:3059
                        "outSR": 3059  # Tell the server to return in EPSG:3059
                    }
                    print(f"[DEBUG] Querying ArcGIS feature layer '{key}' at {query_url} with params: {params}")
                    resp = requests.get(query_url, params=params, timeout=20)
                    print(f"[DEBUG] ArcGIS response status for {key}: {resp.status_code}")
                    resp.raise_for_status()
                    data = resp.json()
                    print(f"[DEBUG] ArcGIS response for {key} contains keys: {list(data.keys())}")
                    if "features" in data and data["features"]:
                        print(f"[DEBUG] {key} - {len(data['features'])} features returned from ArcGIS service.")
                        gdf = gpd.GeoDataFrame.from_features(data["features"])
                        # If your ArcGIS returns EPSG:3059, just set CRS:
                        gdf = gdf.set_crs("EPSG:3059", allow_override=True)
                        # Extract diameter if present
                        diameter_field = None
                        for col in gdf.columns:
                            if "diam" in col.lower():
                                diameter_field = col
                                break
                        # Only add 'diameter' if it doesn't already exist
                        if diameter_field and 'diameter' not in gdf.columns:
                            print(f"[DEBUG] {key} - Found diameter field: {diameter_field}")
                            gdf["diameter"] = gdf[diameter_field]
                        else:
                            print(f"[DEBUG] {key} - No diameter field found or already present.")
                        # Drop duplicate 'diameter' column if both 'Diameter' and 'diameter' exist
                        if 'Diameter' in gdf.columns and 'diameter' in gdf.columns:
                            print(f"[DEBUG] {key} - Dropping duplicate 'diameter' column before saving to GPKG.")
                            gdf = gdf.drop(columns=['diameter'])
                        try:
                            gdf.to_file(shp_cache_file, driver='GPKG')
                        except Exception as write_err:
                            print(f"[DEBUG] Failed to write GPKG for {key}: {write_err}. Deleting file if exists.")
                            if os.path.exists(shp_cache_file):
                                os.remove(shp_cache_file)
                            raise write_err
                        shp_geoms[key] = gdf
                    else:
                        print(f"[DEBUG] {key} - No features found in extent.")
                        shp_geoms[key] = None
            except Exception as e:
                logger.error(f"Error loading ArcGIS feature layer '{key}': {e}")
                shp_geoms[key] = None
        else:
            wms_cache_file = os.path.join(cache_dir, f"wms_{key}_{bbox_str}.npy")
            try:
                if os.path.exists(wms_cache_file):
                    img = np.load(wms_cache_file)
                    wms_images[key] = img
                else:
                    wms_params = {
                        "service": "WMS",
                        "request": "GetMap",
                        "layers": layer["layers"],
                        "styles": layer["styles"],
                        "format": layer["format"],
                        "version": layer["version"],
                        "crs": layer["crs"],
                        "bbox": bbox,
                        "width": str(width),
                        "height": str(height),
                        "transparent": "true"
                    }
                    wms_url = layer["url"]
                    resp = requests.get(wms_url, params=wms_params, timeout=20)
                    resp.raise_for_status()
                    img = mpimg.imread(io.BytesIO(resp.content), format=layer["format"].split("/")[-1])
                    np.save(wms_cache_file, img)
                    wms_images[key] = img
            except Exception as e:
                logger.error(f"Error loading WMS layer '{key}': {e}")
                wms_images[key] = None
    return wms_images, shp_geoms
