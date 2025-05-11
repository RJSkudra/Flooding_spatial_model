"""
Flood animation and DEM modification entry point.
Handles DEM loading, dam drawing, flood simulation, and animation export.
"""

import os
import sys
import logging
import json
import glob
import numpy as np
import rasterio
import geopandas as gpd
import rasterio.features
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.widgets import PolygonSelector
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import builtins
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.rendering_config import setup_rendering, get_output_path
from utils.dem_utils import load_dem, validate_lks97_coords, modify_dem_with_dam_line, save_dem_as_jpeg
from utils.visual_outputs import create_flood_animation
from FloodGIF_scripts.ui import DrawingWindow
from FloodGIF_scripts.flood_simulation import update_flood_frame
from FloodGIF_scripts.visualization_utils import load_wms_layers
from utils.Flood.layer_manager import load_wms_layers, get_plot_order, preload_layers

# Configure matplotlib rendering before importing pyplot
config = setup_rendering(matplotlib)

# Magic numbers/constants
defaults = {
    'DEFAULT_RESOLUTION': 0.25,
    'ANIMATION_INTERVAL': 100
}
DEFAULT_RESOLUTION = defaults['DEFAULT_RESOLUTION']
ANIMATION_INTERVAL = defaults['ANIMATION_INTERVAL']

logger = logging.getLogger(__name__)

# Globals
polygon_selector = None
fig = None
ax = None
drawn_dam_polygon = None
water_source_point = None
dam_points = []
builtins.num_frames = 100
ani = None
dem_data = None
dem_array = None
dem_transform = None
topo_cmap = None
new_cmap = None
layers_dict = None
plot_order = None
wms_images = None
shp_geoms = None
window = None

# Colormaps
colors = [(0.0, 1.0, 1.0), (0.0, 0.5, 1.0), (0.0, 0.0, 0.5)]
new_cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=256)
topo_colors = [(0.2, 0.2, 0.6), (0.4, 0.7, 0.7), (0.5, 0.8, 0.5), (0.8, 0.8, 0.4), (0.6, 0.4, 0.2), (0.8, 0.8, 0.8)]
topo_cmap = LinearSegmentedColormap.from_list('topo_colormap', topo_colors, N=256)

# Event connection IDs
event_cid = {'dam': None, 'source': None}

# Load WMS layers config (if available)
wms_layers = None
try:
    wms_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils', 'Flood', 'wms_layers.json'))
    if os.path.exists(wms_config_path):
        with open(wms_config_path, 'r', encoding='utf-8') as f:
            wms_layers = json.load(f)
    else:
        # fallback to default loader
        wms_layers = load_wms_layers(wms_config_path)
except Exception as e:
    print(f"[WARN] Could not load WMS layers: {e}")
    wms_layers = None

def on_dam_click(event: matplotlib.backend_bases.MouseEvent) -> None:
    """
    Handles dam drawing click events.
    """
    global dam_points
    if event.inaxes == ax and len(dam_points) < 2:
        lks97_x, lks97_y = event.xdata, event.ydata
        if validate_lks97_coords(lks97_x, lks97_y, ax, is_pixel_coords=False):
            dam_points.append((lks97_x, lks97_y))
            ax.plot(lks97_x, lks97_y, 'rx', markersize=4)
            fig.canvas.draw()
            if len(dam_points) == 2:
                x_coords, y_coords = zip(*dam_points)
                ax.add_line(Line2D(x_coords, y_coords, color='red', linewidth=2))
                fig.canvas.mpl_disconnect(event_cid['dam'])
                fig.canvas.draw()

def on_source_click(event: matplotlib.backend_bases.MouseEvent) -> None:
    """
    Handles water source selection click events.
    """
    global water_source_point
    if event.inaxes == ax:
        lks97_x, lks97_y = event.xdata, event.ydata
        if validate_lks97_coords(lks97_x, lks97_y, ax, is_pixel_coords=False):
            water_source_point = (lks97_x, lks97_y)
            print(f"[INFO] Water source selected at: {water_source_point}")
            ax.plot(lks97_x, lks97_y, 'ro', markersize=8)
            fig.canvas.mpl_disconnect(event_cid['source'])
            fig.canvas.draw()
            

def enable_dam_drawing() -> None:
    """
    Enables dam drawing mode.
    """
    if event_cid['source']:
        fig.canvas.mpl_disconnect(event_cid['source'])
    global dam_points
    dam_points.clear()
    event_cid['dam'] = fig.canvas.mpl_connect('button_press_event', on_dam_click)

def enable_source_selection() -> None:
    """
    Enables water source selection mode.
    """
    if event_cid['dam']:
        fig.canvas.mpl_disconnect(event_cid['dam'])
    global water_source_point
    water_source_point = None
    event_cid['source'] = fig.canvas.mpl_connect('button_press_event', on_source_click)

def update_dem_with_dam() -> None:
    """
    Updates the DEM with the drawn dam.
    """
    global dem_array
    if len(dam_points) == 2:
        dem_array[:] = modify_dem_with_dam_line(dem_array, dam_points, dam_height=60, dem_transform=dem_transform)
        save_dem_as_jpeg(dem_array, topo_cmap, get_output_path('edited_dem_debug.jpeg'))

def create_animation() -> None:
    """
    Creates the flood animation and the combined animation.
    """
    import threading
    if threading.current_thread() is threading.main_thread():
        print("[WARNING] Flood simulation and animation are running in the main thread. This may freeze the UI. Consider running heavy computations in a background thread or precomputing results before animation.")
    if water_source_point and len(dam_points) == 2:
        print(f"[DEBUG] Water source point: {water_source_point}")
        print(f"[DEBUG] Dam points: {dam_points}")
        try:
            row, col = rasterio.transform.rowcol(dem_transform, water_source_point[0], water_source_point[1])
            print(f"[DEBUG] Converted source to DEM row,col: ({row}, {col})")
        except Exception as e:
            print(f"Error converting source point: {e}")
            row, col = 0, 0  # fallback

        # Run the flood simulation ONCE
        from FloodGIF_scripts.flood_simulation import simulate_flood_numba
        num_frames = builtins.num_frames
        flood_frames = simulate_flood_numba(dem_array, row, col, num_frames)

        from utils.visual_outputs import create_flood_animation, create_combined_flood_animation, export_last_frame_as_jpeg

        create_flood_animation(fig, ax, dem_array, dem_transform, flood_frames, water_source_point, dam_points, topo_cmap, new_cmap, update_flood_frame)
        create_combined_flood_animation(fig, ax, dem_array, dem_transform, flood_frames, water_source_point, dam_points, topo_cmap, new_cmap, update_flood_frame)
        # Export the last frame of the combined animation as a JPEG (if needed)
        # export_last_frame_as_jpeg(..., flood_array, ...)

        # Calculate flooded area and export shapefile
        if shp_geoms is not None:
            kudraugsne_gdf = shp_geoms.get("LAD Kūdraugsne")
            deklar_lauki_gdf = shp_geoms.get("deklarētie_lauki_2025")
            area_kudraugsne = calculate_flooded_area_in_layer(flood_frames[-1], kudraugsne_gdf, dem_transform)
            area_deklar_lauki = calculate_flooded_area_in_layer(flood_frames[-1], deklar_lauki_gdf, dem_transform)
            #calculate all of the flooded area
            total_flooded_area = flood_frames[-1].sum() * abs(dem_transform.a * dem_transform.e)
            print(f"Total flooded area: {total_flooded_area:.2f} m²")
            print(f"Flooded area inside LAD Kūdraugsne: {area_kudraugsne:.2f} m²")
            print(f"Flooded area inside deklarētie lauki 2025: {area_deklar_lauki:.2f} m²")
        from FloodGIF_scripts.flood_simulation import export_flood_to_shapefile
        print(f"Exporting flood to shapefile...")
        export_flood_to_shapefile(flood_frames[-1], dem_transform, get_output_path("flooded_area.shp"))


def update_num_frames(new_value: int) -> None:
    """
    Updates the global num_frames variable.

    Args:
        new_value (int): The new value for num_frames.
    """
    builtins.num_frames = new_value
    
    
def calculate_flooded_area_in_layer(flood_mask, polygon_gdf, dem_transform):
    if polygon_gdf is None or polygon_gdf.empty:
        return 0.0
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in polygon_gdf.geometry],
        out_shape=flood_mask.shape,
        transform=dem_transform,
        fill=0,
        dtype='uint8'
    )
    flooded_in_layer = (flood_mask & (mask == 1))
    cell_area = abs(dem_transform.a * dem_transform.e)
    return flooded_in_layer.sum() * cell_area

def main(existing_app=None):
    global window
    print("Flood_GIF.main() called.")
    # DEM selection logic (moved from top level)
    DEM_DIR = './src/dem_data/'
    dem_files = glob.glob(os.path.join(DEM_DIR, '*.tif'))
    if not dem_files:
        QMessageBox.critical(None, "DEM Error", f"No DEM .tif files found in {DEM_DIR}")
        sys.exit(1)
    app = existing_app or QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True
    dem_file, _ = QFileDialog.getOpenFileName(None, "Select DEM file", DEM_DIR, "GeoTIFF Files (*.tif)")
    if not dem_file:
        dem_file = dem_files[0]
    DEM_PATH = dem_file
    global dem_data, dem_array, dem_transform, fig, ax, topo_cmap, new_cmap, layers_dict, plot_order, wms_images, shp_geoms, polygon_selector
    dem_data, dem_array, dem_transform = load_dem(DEM_PATH)
    print(f"[DEBUG] DEM min: {np.nanmin(dem_array):.2f}, max: {np.nanmax(dem_array):.2f}")
    # Get DEM bounds for WMS/shapefile overlays
    bounds = dem_data.bounds
    minx, maxx, miny, maxy = bounds.left, bounds.right, bounds.bottom, bounds.top
    # Load WMS layers config and plot order
    wms_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils', 'Flood', 'wms_layers.json'))
    layers_dict = load_wms_layers(wms_config_path)
    plot_order = layers_dict.get('plot_order', [k for k in layers_dict if k != 'plot_order'])
    # Preload WMS/shapefile/ArcGIS overlays for the DEM extent
    wms_images, shp_geoms = preload_layers(layers_dict, plot_order, minx, maxx, miny, maxy)
    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    polygon_selector = PolygonSelector(ax, lambda v: None, useblit=True)
    polygon_selector.set_active(False)
    window = DrawingWindow(
        fig, ax, dem_array, dem_data, topo_cmap, 
        enable_dam_drawing, enable_source_selection, 
        update_dem_with_dam, create_animation, builtins.num_frames,
        update_num_frames, layers_dict, plot_order, wms_images, shp_geoms
    )
    print("DrawingWindow initialized.")
    window.show()
    print(">>> DrawingWindow should now be visible.")
    app.processEvents()
    print("GUI refreshed.")
    print("Window show() executed.")
    if created_app:
        app.exec_()

if __name__ == '__main__':
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)  # Print top 30 time-consuming calls