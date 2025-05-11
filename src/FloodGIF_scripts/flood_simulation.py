"""
Flood simulation utilities for DEM-based water spread modeling.
"""

import logging
import numpy as np
import rasterio
from heapq import heappush, heappop
from shapely.geometry import LineString
from typing import Tuple, Any
from collections import deque
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape
from numba import njit
from numba.typed import List

logger = logging.getLogger(__name__)

# Constants
STEPS = 500

def simulate_flood(dem_array: np.ndarray, source_point: Tuple[float, float], dem_transform: Any) -> np.ndarray:
    """
    Simulates the spread of flood water from a source point over a DEM array.

    Args:
        dem_array (np.ndarray): The digital elevation model array.
        source_point (Tuple[float, float]): The coordinates of the flood source point.
        dem_transform (Any): The transformation object for the DEM.

    Returns:
        np.ndarray: A boolean array indicating flooded areas.
    """
    flood_array = np.zeros_like(dem_array, dtype=bool)
    visited = np.zeros_like(dem_array, dtype=bool)
    try:
        row, col = rasterio.transform.rowcol(dem_transform, source_point[0], source_point[1])
    except Exception as e:
        logger.error(f"Error determining source point location: {e}")
        return flood_array
    if not (0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]):
        return flood_array
    queue = []
    heappush(queue, (dem_array[row, col], row, col))
    max_water_level = dem_array[row, col]
    while queue or np.any(~visited):
        if not queue:
            # Find the next minimum elevation around the flooded area
            next_water_level = float('inf')
            for r, c in zip(*np.where(flood_array)):  # Iterate over all flooded cells
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check neighbors
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < dem_array.shape[0]
                        and 0 <= nc < dem_array.shape[1]
                        and not visited[nr, nc]  # Only consider unvisited cells
                        and dem_array[nr, nc] > max_water_level  # Higher than current water level
                    ):
                        next_water_level = min(next_water_level, dem_array[nr, nc])
                
                # If no valid neighbors are found, break the loop
            if next_water_level == float('inf'):
                break

            # Update the max water level
            max_water_level = next_water_level
            continue

        elevation, r, c = heappop(queue)
        if visited[r, c]:
            continue

    visited[r, c] = True
    flood_array[r, c] = True
    max_water_level = max(max_water_level, elevation)

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if (
            0 <= nr < dem_array.shape[0]
            and 0 <= nc < dem_array.shape[1]
            and not visited[nr, nc]
            and dem_array[nr, nc] <= max_water_level
        ):
            heappush(queue, (dem_array[nr, nc], nr, nc))
    return flood_array



# def simulate_flood_old(dem_array: np.ndarray, source_point: Tuple[float, float], dem_transform: Any) -> np.ndarray:
#     """
#     Simulates the spread of flood water from a source point over a DEM array.

#     Args:
#         dem_array (np.ndarray): The digital elevation model array.
#         source_point (Tuple[float, float]): The coordinates of the flood source point.
#         dem_transform (Any): The transformation object for the DEM.

#     Returns:
#         np.ndarray: A boolean array indicating flooded areas.
#     """
#     flood_array = np.zeros_like(dem_array, dtype=bool)
#     visited = np.zeros_like(dem_array, dtype=bool)

    return flood_frames


def update_flood_frame(frame: int, dem_array: np.ndarray, dem_transform: Any, water_source_point: Tuple[float, float], dam_points: list, ax: Any, show: Any, topo_cmap: Any, new_cmap: Any, progress_bar: Any, water_elevations: list, flooded_areas: list, num_frames: int, flood_array: np.ndarray, visited: np.ndarray, queue: list) -> None:
    """
    Updates the flood simulation visualization for a given frame.

    Args:
        frame (int): The current frame number.
        dem_array (np.ndarray): The digital elevation model array.
        dem_transform (Any): The transformation object for the DEM.
        water_source_point (Tuple[float, float]): The coordinates of the flood source point.
        dam_points (list): List of dam points for visualization.
        ax (Any): The matplotlib axis object.
        show (Any): The function to display the DEM and flood arrays.
        topo_cmap (Any): The colormap for the DEM.
        new_cmap (Any): The colormap for the flood visualization.
        progress_bar (Any): The progress bar object.
        water_elevations (list): List to store water elevation levels.
        flooded_areas (list): List to store flooded area sizes.
        num_frames (int): Total number of frames in the simulation.
        flood_array (np.ndarray): Boolean array indicating flooded areas.
        visited (np.ndarray): Boolean array indicating visited cells.
        queue (list): Priority queue for flood simulation.

    Returns:
        None
    """
    ax.clear()
    show(dem_array, ax=ax, cmap=topo_cmap, transform=dem_transform)
    if frame == 0:
        # Already initialized in main script
        pass
    max_water_level_current_frame = float('-inf')
    for _ in range(STEPS):
        if not queue:
            break
        elevation, r, c = heappop(queue)
        # Only log if debug is enabled (disable for performance)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing cell: elevation={elevation}, row={r}, col={c}, queue_size={len(queue)}")
        if visited[r, c]:
            continue
        visited[r, c] = True
        flood_array[r, c] = True
        max_water_level_current_frame = max(max_water_level_current_frame, elevation)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < dem_array.shape[0]
                and 0 <= nc < dem_array.shape[1]
                and not visited[nr, nc]
            ):
                heappush(queue, (dem_array[nr, nc], nr, nc))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Flooded cells count: {np.sum(flood_array)}")
    if max_water_level_current_frame != float('-inf'):
        water_elevations.append(max_water_level_current_frame)
    elif water_elevations:
        water_elevations.append(water_elevations[-1])
    else:
        water_elevations.append(0)
    cell_area = dem_transform[0] * -dem_transform[4]
    flooded_area = np.sum(flood_array) * cell_area
    flooded_areas.append(flooded_area)
    # Debugging: Log water elevations and flooded areas
    logger.debug(f"Kadrs {frame}: Maksimālais ūdens līmenis = {max_water_level_current_frame}")
    logger.debug(f"Kadrs {frame}: Applūdusī platība = {flooded_area}")
    masked_flood_array = np.ma.masked_where(~flood_array, flood_array)
    show(masked_flood_array, ax=ax, cmap=new_cmap, alpha=0.7, transform=dem_transform)
    if len(dam_points) == 2:
        x_coords, y_coords = zip(*dam_points)
        ax.plot(x_coords, y_coords, color='red', linewidth=2, label='Uzzīmētais Aizsprosts')
    ax.set_title(f'Plūdu Simulācija (Kadrs: {frame})', fontsize=10)
    ax.text(0.5, -0.1, f'Applūdusī platība: {flooded_area:.2f} m³',
            transform=ax.transAxes, fontsize=10, ha='center', va='center')
    if progress_bar is not None:
        progress_bar.update(1)

def export_flood_to_shapefile(flood_array: np.ndarray, dem_transform: Any, output_path: str, crs: str = "EPSG:3059"):
    """
    Export the flooded area mask as a shapefile for use in QGIS.

    Args:
        flood_array (np.ndarray): Boolean array indicating flooded areas.
        dem_transform (Any): The transformation object for the DEM.
        output_path (str): Path to the output shapefile.
        crs (str): Coordinate reference system (default: "EPSG:3059").
    """
    shapes_gen = features.shapes(flood_array.astype(np.uint8), mask=flood_array, transform=dem_transform)
    polygons = [shape(geom) for geom, value in shapes_gen if value == 1]
    if not polygons:
        logger.warning("No flooded areas to export.")
        return
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(output_path)
    logger.info(f"Flooded area exported to {output_path}")