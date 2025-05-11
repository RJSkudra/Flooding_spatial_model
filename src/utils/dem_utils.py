"""
DEM utility functions for loading, validating, modifying, and saving DEM data.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from shapely.geometry import Polygon, LineString
from typing import Tuple, Optional, Any
from pyproj import CRS

logger = logging.getLogger(__name__)

LATVIAN_CRS = CRS.from_epsg(3059)
VALID_X_RANGE = (300000, 760000)
VALID_Y_RANGE = (170000, 430000)

def load_dem(dem_path: str) -> Tuple[Any, np.ndarray, Affine]:
    """
    Load DEM data from a file.

    Args:
        dem_path (str): Path to the DEM file.

    Returns:
        Tuple containing the rasterio dataset, DEM array, and transform.
    """
    dem_data = rasterio.open(dem_path)
    dem_array = dem_data.read(1)
    dem_transform = dem_data.transform
    dem_crs = dem_data.crs
    # Handle NoData values
    nodata = dem_data.nodata
    if nodata is not None:
        dem_array = np.where(dem_array == nodata, np.nan, dem_array)
    if dem_crs.to_epsg() != LATVIAN_CRS.to_epsg():
        logger.warning(f"DEM CRS ({dem_crs.to_string()}) does not match LKS-97 (EPSG:3059)")
    return dem_data, dem_array, dem_transform

def validate_lks97_coords(x: float, y: float, ax: Optional[Any] = None, is_pixel_coords: bool = True) -> bool:
    """
    Validate coordinates against LKS-97 bounds.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        ax (Optional[Any]): Matplotlib axis for pixel coordinates validation.
        is_pixel_coords (bool): Whether the coordinates are pixel-based.

    Returns:
        bool: True if coordinates are valid, False otherwise.
    """
    if is_pixel_coords and ax is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]
    else:
        return VALID_X_RANGE[0] <= x <= VALID_X_RANGE[1] and VALID_Y_RANGE[0] <= y <= VALID_Y_RANGE[1]

def pixel_to_lks97(x: float, y: float, dem_transform: Affine, dem_array: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Convert pixel coordinates to LKS-97 coordinates.

    Args:
        x (float): X pixel coordinate.
        y (float): Y pixel coordinate.
        dem_transform (Affine): DEM transform.
        dem_array (np.ndarray): DEM array.

    Returns:
        Optional[Tuple[float, float]]: LKS-97 coordinates or None if invalid.
    """
    try:
        col = int((x - dem_transform[2]) / dem_transform[0])
        row = int((y - dem_transform[5]) / dem_transform[4])
        if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
            x_lks97, y_lks97 = rasterio.transform.xy(dem_transform, row, col)
            return x_lks97, y_lks97
        else:
            return None
    except Exception as e:
        logger.error(f"Error converting pixel to LKS-97 coordinates: {e}")
        return None

def modify_dem_with_dam(dem_array: np.ndarray, dam_polygon: Polygon, dam_height: float, dem_transform: Affine) -> np.ndarray:
    """
    Modify DEM array with dam polygon.

    Args:
        dem_array (np.ndarray): DEM array.
        dam_polygon (Polygon): Dam polygon geometry.
        dam_height (float): Height of the dam.
        dem_transform (Affine): DEM transform.

    Returns:
        np.ndarray: Modified DEM array.
    """
    dam_mask = geometry_mask([dam_polygon], transform=dem_transform, invert=True, out_shape=dem_array.shape)
    dem_array[dam_mask] = np.maximum(dem_array[dam_mask], dem_array.max() + dam_height)
    return dem_array

def modify_dem_with_dam_line(dem_array: np.ndarray, dam_points: Tuple[Tuple[float, float], Tuple[float, float]], dam_height: float, dem_transform: Affine) -> np.ndarray:
    """
    Modify DEM array with dam line. The dam height is set to the highest surrounding DEM value near the two dam points.

    Args:
        dem_array (np.ndarray): DEM array.
        dam_points (Tuple[Tuple[float, float], Tuple[float, float]]): Dam line points.
        dam_height (float): Height of the dam (ignored, calculated from DEM).
        dem_transform (Affine): DEM transform.

    Returns:
        np.ndarray: Modified DEM array.
    """
    if len(dam_points) != 2:
        logger.warning("Dam line must consist of exactly two points.")
        return dem_array

    # Find the highest DEM value in a 3x3 window around each dam point
    surrounding_heights = []
    for point in dam_points:
        # Convert world coordinates to DEM array indices
        col = int(round((point[0] - dem_transform[2]) / dem_transform[0]))
        row = int(round((point[1] - dem_transform[5]) / dem_transform[4]))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < dem_array.shape[0] and 0 <= nc < dem_array.shape[1]:
                    surrounding_heights.append(dem_array[nr, nc])
    if surrounding_heights:
        dam_height = max(surrounding_heights)
    else:
        dam_height = dem_array.max()  # fallback

    dam_line_geom = LineString(dam_points).buffer(3)
    dam_mask = geometry_mask([dam_line_geom], transform=dem_transform, invert=True, out_shape=dem_array.shape)
    dem_array[dam_mask] = np.maximum(dem_array[dam_mask], dam_height)
    return dem_array

def save_dem_as_jpeg(dem_array: np.ndarray, topo_cmap: str, output_path: str, title: str = None, extent: tuple = None) -> None:
    """
    Save DEM array as a JPEG image.

    Args:
        dem_array (np.ndarray): DEM array.
        topo_cmap (str): Colormap for the image.
        output_path (str): Path to save the image.
        title (str, optional): Title for the plot. Defaults to a standard title.
        extent (tuple, optional): (left, right, bottom, top) for axis coordinates.
    """
    import matplotlib.pyplot as plt
    dem_array = dem_array.astype(float)
    plt.figure(figsize=(10, 10))
    if extent is not None:
        plt.imshow(dem_array, cmap=topo_cmap, extent=extent, origin='upper')
    else:
        plt.imshow(dem_array, cmap=topo_cmap)
    plt.colorbar(label='Augstums (m)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Rediģētais DEM ar Aizsprosta Modifikācijām')
    if extent is not None:
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
    plt.savefig(output_path, dpi=300, format='jpg')
    plt.close()

def save_temp_dem(modified_dem_array: np.ndarray, dem_transform: Affine, LATVIAN_CRS: CRS, temp_dem_path: str) -> None:
    """
    Save modified DEM array as a temporary GeoTIFF file.

    Args:
        modified_dem_array (np.ndarray): Modified DEM array.
        dem_transform (Affine): DEM transform.
        LATVIAN_CRS (CRS): Coordinate reference system.
        temp_dem_path (str): Path to save the temporary DEM file.
    """
    temp_dem_dir = os.path.dirname(temp_dem_path)
    if not os.path.exists(temp_dem_dir):
        os.makedirs(temp_dem_dir)
    with rasterio.open(
        temp_dem_path,
        'w',
        driver='GTiff',
        height=modified_dem_array.shape[0],
        width=modified_dem_array.shape[1],
        count=1,
        dtype=modified_dem_array.dtype,
        crs=LATVIAN_CRS.to_wkt(),
        transform=dem_transform,
    ) as temp_dem:
        temp_dem.write(modified_dem_array, 1)
