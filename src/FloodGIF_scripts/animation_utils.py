import builtins
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from src.FloodGIF_scripts.visualization_utils import generate_water_elevation_plot
from src.config.rendering_config import get_output_path

def create_flood_animation(fig, ax, dem_array, dem_transform, water_source_point, dam_points, topo_cmap, new_cmap, update_flood_frame_func):
    num_frames = getattr(builtins, 'num_frames', 100)
    flood_array = np.zeros_like(dem_array, dtype=bool)
    visited = np.zeros_like(dem_array, dtype=bool)
    from heapq import heappush
    try:
        row, col = int((water_source_point[1] - dem_transform[5]) / dem_transform[4]), int((water_source_point[0] - dem_transform[2]) / dem_transform[0])
    except Exception:
        row, col = 0, 0
    queue = []
    heappush(queue, (dem_array[row, col], row, col))
    water_elevations = []
    flooded_areas = []
    from rasterio.plot import show as rasterio_show

    # Precompute DEM and hillshade backgrounds
    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)
    dem_min = np.nanmin(dem_plot_array)
    dem_max = np.nanmax(dem_plot_array)
    extent = [
        dem_transform[2],
        dem_transform[2] + dem_transform[0] * dem_plot_array.shape[1],
        dem_transform[5] + dem_transform[4] * dem_plot_array.shape[0],
        dem_transform[5]
    ]
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_plot_array, vert_exag=3)

    progress_bar = tqdm(total=num_frames+1, desc="Generating GIF Animation")

    def frame_func(frame):
        ax.clear()
        # Draw DEM as background
        im_dem = ax.imshow(dem_plot_array, cmap=topo_cmap, extent=extent, origin='upper', alpha=0.7, vmin=dem_min, vmax=dem_max, zorder=0)
        # Draw hillshade as overlay
        ax.imshow(hillshade, cmap='gray', extent=extent, origin='upper', alpha=0.3, zorder=1)
        # Update flood and overlays
        update_flood_frame_func(
            frame, dem_plot_array, dem_transform, water_source_point, dam_points, ax,
            rasterio_show, topo_cmap, new_cmap, progress_bar, water_elevations, flooded_areas, num_frames, flood_array, visited, queue
        )
        # Keep UI responsive
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

    ani = FuncAnimation(fig, frame_func, frames=num_frames, repeat=False)
    gif_output_path = get_output_path('flood_simulation.gif')
    ani.save(gif_output_path, writer=PillowWriter(fps=5))
    progress_bar.close()
    print(f"Animation saved: {gif_output_path}")
    # Generate 1D water elevation plot after animation
    elevation_plot_path = get_output_path('water_elevation_changes.jpeg')
    generate_water_elevation_plot(water_elevations, flooded_areas, elevation_plot_path)
    return gif_output_path, elevation_plot_path