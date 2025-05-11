"""
Visualization utilities for flood simulation and WMS/shapefile/LiDAR overlays.
Provides region selection and WMS layer management for interactive DEM workflows.
"""

import os
import sys
import json
import logging
import threading
from typing import Any, Dict, Optional

import numpy as np
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QEventLoop
import laspy
from pyproj import CRS, Transformer
import geopandas as gpd

# Import custom rendering configuration
from config.rendering_config import setup_rendering, get_output_path
from utils.Flood.layer_manager import load_wms_layers, get_plot_order, preload_layers
import matplotlib.patches as mpatches

def plot_half_filled_circle(ax, x, y, size, color, outline_color, outline_width, zorder=10):
    # Right half (filled)
    wedge = mpatches.Wedge((x, y), size, 270, 90, facecolor=color, edgecolor=outline_color, linewidth=outline_width, zorder=zorder)
    ax.add_patch(wedge)
    # Left half (white)
    wedge2 = mpatches.Wedge((x, y), size, 90, 270, facecolor='white', edgecolor=outline_color, linewidth=outline_width, zorder=zorder)
    ax.add_patch(wedge2)
    
from shapely.geometry import LineString

def plot_double_line(ax, coords, color, width, offset, alpha=1.0, zorder=10):
    line = LineString(coords)
    left = line.parallel_offset(offset, 'left')
    right = line.parallel_offset(offset, 'right')
    for seg in [left, right]:
        if seg.is_empty:
            continue
        if seg.geom_type == 'LineString':
            x, y = seg.xy
            ax.plot(x, y, color=color, linewidth=width, alpha=alpha, zorder=zorder)
        elif seg.geom_type == 'MultiLineString':
            for part in seg:
                x, y = part.xy
                ax.plot(x, y, color=color, linewidth=width, alpha=alpha, zorder=zorder)

# Configure matplotlib rendering before importing pyplot
config = setup_rendering(matplotlib)

# Path to the WMS layers configuration file
WMS_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wms_layers.json')

# Default LVM layer configuration (used if file doesn't exist)
DEFAULT_LVM_LAYER = {
    'lvm': {
        'url': 'https://lvmgeoserver.lvm.lv/geoserver/ows',
        'layers': 'public:Orto_LKS',  # Main ortofoto layer
        'name': 'LVM Ortofoto',
        'attribution': '© LVM',
        'format': 'image/png',
        'version': '1.0.0',
        'styles': 'raster',  # Default style for rasters
        'crs': 'EPSG:3059'
    }
}

# Magic numbers/constants
SCATTER_DOWNSAMPLE = 100
WMS_WIDTH = 1200
WMS_HEIGHT = 1080
BUFFER_RATIO = 0.05

logger = logging.getLogger(__name__)


def save_wms_layers(layers: Dict[str, Any]) -> None:
    """Save WMS layer information to config file."""
    try:
        with open(WMS_CONFIG_FILE, 'w') as f:
            json.dump(layers, f, indent=2)
        logger.info(f"Saved WMS layers to {WMS_CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving WMS layers to {WMS_CONFIG_FILE}: {e}")


def select_region_of_interest(las_file_path: str) -> Optional[Dict[str, float]]:
    print("[DEBUG] select_region_of_interest called with:", las_file_path)
    try:
        # --- GUI check ---
        try:
            backend = matplotlib.get_backend().lower()
            print(f"[DEBUG] Matplotlib backend: {backend}")
            fig_test, ax_test = plt.subplots()
            ax_test.plot([0, 1], [0, 1])
            fig_test.canvas.draw()
            plt.close(fig_test)
        except Exception as e:
            logger.error("Matplotlib GUI is not working: %s", str(e))
            logger.error("Please ensure you are running in a graphical environment with a working Qt or Tk backend.")
            print(f"[ERROR] Matplotlib GUI is not working: {e}")
            raise RuntimeError("Matplotlib GUI is not available. Cannot select region of interest.")

        # Load LIDAR data
        las = laspy.read(las_file_path)
        x = las.x
        y = las.y
        z = las.z
        z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))

        # Get bounds for WMS request
        buffer_x = (np.max(x) - np.min(x)) * BUFFER_RATIO
        buffer_y = (np.max(y) - np.min(y)) * BUFFER_RATIO
        minx = np.min(x) - buffer_x
        maxx = np.max(x) + buffer_x
        miny = np.min(y) - buffer_y
        maxy = np.max(y) + buffer_y

        # Use new layer_manager functions
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils', 'Flood', 'wms_layers.json')
        layers_dict = load_wms_layers(config_path)
        plot_order = get_plot_order(layers_dict)
        wms_images, shp_geoms = preload_layers(layers_dict, plot_order, minx, maxx, miny, maxy, WMS_WIDTH, WMS_HEIGHT)

        # --- PyQt5 GUI with Matplotlib FigureCanvas and controls ---
        app = QApplication.instance()
        created_app = False
        if app is None:
            app = QApplication(sys.argv)
            created_app = True
        window = QMainWindow()
        window.setWindowTitle("DEM reģiona izvēle")
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib FigureCanvas
        fig, ax = plt.subplots(figsize=(10, 8))
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, window)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Layer toggles
        layer_visible = {k: False for k in plot_order}
        x_scatter = x[::SCATTER_DOWNSAMPLE]
        y_scatter = y[::SCATTER_DOWNSAMPLE]
        z_norm_scatter = z_norm[::SCATTER_DOWNSAMPLE]

        def update_background():
            ax.clear()
            legend_elements = []  # To store legend elements
            for idx, key in enumerate(plot_order):
                if not layer_visible.get(key, False):
                    continue
                layer = layers_dict[key] if isinstance(layers_dict, dict) and key in layers_dict else {}
                geometry_type = layer.get('geometry_type', None)
                alpha = layer.get('alpha', 1.0)
                zorder = idx + 1  # Ensure increasing zorder for each layer
                # --- Shapefile Layer ---
                if isinstance(layer, dict) and layer.get('type') == 'shapefile' and shp_geoms.get(key) is not None:
                    gdf = shp_geoms[key]
                    fill_color = layer.get('fill_color', 'none')
                    line_color = layer.get('line_color', 'black')
                    legend = layer.get('legend', True)
                    style = {
                        'zorder': zorder,
                        'alpha': alpha
                    }
                    if geometry_type == 'polygon':
                        style['edgecolor'] = line_color
                        style['facecolor'] = fill_color if fill_color else 'none'
                        style['linewidth'] = 1.5
                    elif geometry_type == 'line':
                        style['edgecolor'] = line_color
                        style['facecolor'] = 'none'
                        style['linewidth'] = 2.0
                    else:
                        style['edgecolor'] = line_color
                        style['facecolor'] = fill_color if fill_color else 'none'
                    gdf.plot(ax=ax, **style)
                    if legend:
                        legend_elements.append(
                            patches.Patch(edgecolor=line_color, facecolor=fill_color if (geometry_type=="polygon" and fill_color) else 'none', label=layer.get('name', key), linewidth=style.get('linewidth', 1.5), alpha=alpha)
                        )
                # --- LIDAR Layer ---
                elif isinstance(layer, dict) and layer.get('type') == 'lidar':
                    fill_color = layer.get('fill_color', '#0000FF')
                    legend = layer.get('legend', True)
                    scatter = ax.scatter(x_scatter, y_scatter, c=z_norm_scatter, cmap='viridis', s=0.5, alpha=alpha, zorder=100)
                    if legend:
                        legend_elements.append(
                            patches.Patch(color=fill_color, label=layer.get('name', 'LIDAR Points'), alpha=alpha)
                        )
                # --- WMS Layer ---
                elif key in wms_images and wms_images[key] is not None:
                    legend = layer.get('legend', False)
                    ax.imshow(wms_images[key], extent=[minx, maxx, miny, maxy], alpha=alpha, zorder=zorder)
                    if legend:
                        legend_elements.append(
                            patches.Patch(color='gray', label=layer.get('name', key), alpha=alpha)
                        )
                # --- Custom Drainage Styles ---
                elif key == "drenu_savienojumi" and shp_geoms.get(key) is not None:
                    gdf = shp_geoms[key]
                    for _, row in gdf.iterrows():
                        x, y = row.geometry.x, row.geometry.y
                        plot_half_filled_circle(
                            ax, x, y,
                            size=layer.get("marker_size", 8),
                            color=layer.get("color", "#00FF00"),
                            outline_color=layer.get("marker_outline_color", "#006400"),
                            outline_width=layer.get("marker_outline_width", 1),
                            zorder=zorder
                        )
                elif key in ["drenu_kolektori", "drenu_kolektori_lieli"] and shp_geoms.get(key) is not None:
                    gdf = shp_geoms[key]
                    for _, row in gdf.iterrows():
                        coords = list(row.geometry.coords)
                        plot_double_line(
                            ax, coords,
                            color=layer.get("line_color", "#006400"),
                            width=layer.get("line_width", 3),
                            offset=layer.get("parallel_offset", 2),
                            alpha=layer.get("alpha", 1.0),
                            zorder=zorder
                        )
                # --- Default plotting for other layers ---
                elif shp_geoms.get(key) is not None:
                    gdf = shp_geoms[key]
                    gdf.plot(ax=ax, alpha=alpha, zorder=zorder)
            ax.set_title('Izvēlieties reģionu, velkot taisnstūri ar peli')
            ax.set_xlabel('X koordināte (m)')
            ax.set_ylabel('Y koordināte (m)')
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            if legend_elements:
                # Place legend outside the plot on the right
                ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
            canvas.draw_idle()

        # Layer toggle checkboxes
        layer_box = QHBoxLayout()
        checkboxes = {}
        for key in plot_order:
            cb = QCheckBox(key)
            cb.setChecked(layer_visible[key])
            def make_toggle(k):
                return lambda state: (layer_visible.__setitem__(k, bool(state)), update_background())
            cb.stateChanged.connect(make_toggle(key))
            checkboxes[key] = cb
            layer_box.addWidget(cb)

        # Add label for WMS layer toggles
        wms_label = QLabel("WMS/Slāņu pārslēgšana:")
        layout.addWidget(wms_label)
        layout.addLayout(layer_box)

        # RectangleSelector for region selection
        selected = {}
        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            selected['minx'] = min(x1, x2)
            selected['maxx'] = max(x1, x2)
            selected['miny'] = min(y1, y2)
            selected['maxy'] = max(y1, y2)
            print(f"[DEBUG] Region selected: {selected}")
        rect_selector = RectangleSelector(
            ax, onselect, useblit=True,
            button=[1],  # left mouse button
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )

        # Confirm button
        confirm_btn = QPushButton("Apstiprināt izvēli")
        def on_confirm():
            if selected:
                print(f"[DEBUG] Confirming selection: {selected}")
                # begin the DEM generation process with the selected region
                window.close()
            else:
                print("[DEBUG] No region selected yet.")
        confirm_btn.clicked.connect(on_confirm)
        layout.addWidget(confirm_btn)

        update_background()
        window.show()

        if created_app:
            app.exec_()
        else:
            loop = QEventLoop()
            def on_window_destroyed():
                print("[DEBUG] Window destroyed, quitting event loop.")
                loop.quit()
            window.destroyed.connect(on_window_destroyed)
            loop.exec_()

        if selected:
            return selected
        else:
            print("[DEBUG] No region was selected.")
            return None
    except Exception as e:
        print(f"[ERROR] Exception in select_region_of_interest: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_water_elevation_plot(water_elevations, flooded_areas, output_path):
    """
    Generate a 1D plot showing water elevation changes over time.

    Parameters:
        water_elevations (list): List of water elevation values for each frame.
        flooded_areas (list): List of flooded area sizes for each frame.
        output_path (str): Path to save the generated plot.
    """
    import matplotlib.pyplot as plt

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot water elevations
    ax1.plot(water_elevations, color='blue', label='Ūdens līmenis')
    ax1.set_xlabel('Kadrs')
    ax1.set_ylabel('Ūdens līmenis (m)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a twin axis to plot flooded areas
    ax2 = ax1.twinx()
    ax2.plot(flooded_areas, color='green', label='Aplūstošā platība')
    ax2.set_ylabel('Aplūstošā platība (kv. m)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title and legend
    plt.title('Ūdens līmenis un aplūstošā platība laika gaitā')
    fig.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

    print(f"Water elevation plot saved to {output_path}")