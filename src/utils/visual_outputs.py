"""
Animation utilities for flood simulation visualization.
"""

import logging
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from src.config.rendering_config import get_output_path
from typing import Any
import builtins
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from rasterio.plot import show as rasterio_show
import os
from concurrent.futures import ProcessPoolExecutor
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage import measure

logger = logging.getLogger(__name__)

def create_flood_animation(
    fig: Any, ax: Any, dem_array: np.ndarray, dem_transform: np.ndarray, flood_frames: np.ndarray,
    water_source_point: tuple, dam_points: list,
    topo_cmap: Any, new_cmap: Any, update_flood_frame_func: Any
) -> tuple:
    """
    Create an animation for flood simulation visualization.

    Args:
        fig (Any): Matplotlib figure object.
        ax (Any): Matplotlib axis object.
        dem_array (np.ndarray): Digital elevation model array.
        dem_transform (np.ndarray): Transformation parameters for DEM.
        flood_array (np.ndarray): Boolean array of flooded cells (from simulate_flood_numba).
        water_source_point (tuple): Coordinates of the water source point.
        dam_points (list): List of dam points.
        topo_cmap (Any): Colormap for topography.
        new_cmap (Any): Colormap for flood visualization.
        update_flood_frame_func (Any): Function to update flood frames.

    Returns:
        tuple: Path to the generated GIF and water elevation plot.
    """
    FPS = 5

    # Update colormaps for DEM and water visualization
    dem_colors = [(0.2, 0.2, 0.6), (0.4, 0.7, 0.7), (0.5, 0.8, 0.5), (0.8, 0.8, 0.4), (0.6, 0.4, 0.2), (0.8, 0.8, 0.8)]
    topo_cmap = LinearSegmentedColormap.from_list('topo_colormap', dem_colors, N=256)

    water_colors = [(0.0, 0.0, 0.5), (0.0, 0.0, 0.8), (0.0, 0.5, 1.0), (0.0, 1.0, 1.0)]
    new_cmap = LinearSegmentedColormap.from_list('water_colormap', water_colors, N=256)
    
    # Use a blue colormap for flooded areas (from light blue to deep blue)
    flood_colors = [(0.7, 0.85, 1.0), (0.0, 0.2, 0.8)]  # light blue to deep blue
    flood_cmap = LinearSegmentedColormap.from_list('flood_blue', flood_colors, N=256)

    # Mask -9999 values as np.nan for plotting
    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)

    water_elevations = []
    flooded_areas = []
    num_frames = getattr(builtins, 'num_frames', 100)
    progress_bar = tqdm(total=num_frames+1, desc="Generating GIF Animation")

    def frame_func(frame, flood_frames):
        ax.clear()
        dem_min = np.nanmin(dem_plot_array)
        dem_max = np.nanmax(dem_plot_array)
        # Draw DEM as background with correct scaling
        im_dem = rasterio_show(dem_plot_array, ax=ax, cmap=topo_cmap, transform=dem_transform, zorder=0, alpha=0.7, vmin=dem_min, vmax=dem_max)
        # Add DEM colorbar
        import matplotlib as mpl
        sm_dem = mpl.cm.ScalarMappable(cmap=topo_cmap, norm=mpl.colors.Normalize(vmin=dem_min, vmax=dem_max))
        if hasattr(ax, 'figure'):
            if not hasattr(ax, '_dem_colorbar') or ax._dem_colorbar is None:
                ax._dem_colorbar = ax.figure.colorbar(sm_dem, ax=ax, orientation='vertical', fraction=0.025, pad=0.02, label='DEM augstums (m)')
        # Optionally add hillshade for better visualization
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(dem_plot_array, vert_exag=3)
        ax.imshow(hillshade, cmap='gray', alpha=0.3, extent=[dem_transform[2], dem_transform[2] + dem_transform[0] * dem_plot_array.shape[1], dem_transform[5] + dem_transform[4] * dem_plot_array.shape[0], dem_transform[5]], zorder=1)
        update_flood_frame_func(
            frame, dem_plot_array, dem_transform, water_source_point, dam_points, ax,
            rasterio_show, topo_cmap, new_cmap, progress_bar, water_elevations, flooded_areas, num_frames, flood_frames[frame], None, None
        )
        # Show flooded areas in blue
        flood_mask = np.ma.masked_where(~flood_frames[frame], flood_frames[frame])
        ax.imshow(flood_mask, cmap=flood_cmap, alpha=0.6, extent=[dem_transform[2], dem_transform[2] + dem_transform[0] * dem_array.shape[1], dem_transform[5] + dem_transform[4] * dem_array.shape[0], dem_transform[5]], zorder=2)
        # Add flood colorbar
        sm_flood = mpl.cm.ScalarMappable(cmap=flood_cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        if hasattr(ax, 'figure'):
            if not hasattr(ax, '_flood_colorbar') or ax._flood_colorbar is None:
                ax._flood_colorbar = ax.figure.colorbar(sm_flood, ax=ax, orientation='vertical', fraction=0.025, pad=0.08, label='Nopludinātais laukums')

    ani = FuncAnimation(fig, lambda frame: frame_func(frame, flood_frames), frames=np.arange(0, num_frames), repeat=False)
    gif_output_path = get_output_path('flood_simulation.gif')
    ani.save(gif_output_path, writer=PillowWriter(fps=FPS))
    progress_bar.close()
    logger.info(f"Animation saved: {gif_output_path}")

    # Generate 1D water elevation plot after animation
    elevation_plot_path = get_output_path('water_elevation_changes.jpeg')
    generate_water_elevation_plot(water_elevations, flooded_areas, elevation_plot_path)
    logger.info(f"Water elevation plot saved: {elevation_plot_path}")
    return gif_output_path, elevation_plot_path

def create_combined_flood_animation(
    fig: Any, ax: Any, dem_array: np.ndarray, dem_transform: np.ndarray, 
    flood_frames: np.ndarray, water_source_point: tuple, dam_points: list, 
    topo_cmap: Any, new_cmap: Any, update_flood_frame_func: Any
) -> str:
    """
    Create a combined animation for flood simulation visualization with a 2D plot, water elevation graph, and flooded area graph.

    Args:
        fig (Any): Matplotlib figure object.
        ax (Any): Matplotlib axis object.
        dem_array (np.ndarray): Digital elevation model array.
        dem_transform (np.ndarray): Transformation parameters for DEM.
        flood_array (np.ndarray): Boolean array of flooded cells (from simulate_flood_numba).
        water_source_point (tuple): Coordinates of the water source point.
        dam_points (list): List of dam points.
        topo_cmap (Any): Colormap for topography.
        new_cmap (Any): Colormap for flood visualization.
        update_flood_frame_func (Any): Function to update flood frames.

    Returns:
        str: Path to the generated combined GIF.
    """
    from matplotlib.gridspec import GridSpec

    FPS = 5

    # Create a new figure with subplots for 2D plot, water elevation graph, and flooded area graph
    combined_fig = plt.figure(figsize=(10, 15))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], figure=combined_fig)

    ax_2d = combined_fig.add_subplot(gs[0])
    ax_elevation = combined_fig.add_subplot(gs[1])
    ax_flooded_area = combined_fig.add_subplot(gs[2])

    # Mask -9999 values as np.nan for plotting
    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)

    # Use a blue colormap for flooded areas (from light blue to deep blue)
    flood_colors = [(0.7, 0.85, 1.0), (0.0, 0.2, 0.8)]  # light blue to deep blue
    flood_cmap = LinearSegmentedColormap.from_list('flood_blue', flood_colors, N=256)

    water_elevations = []
    flooded_areas = []
    num_frames = getattr(builtins, 'num_frames', 100)
    progress_bar = tqdm(total=num_frames+1, desc="Generating Combined GIF Animation")

    def frame_func(frame):
        ax_2d.clear()
        dem_min = np.nanmin(dem_plot_array)
        dem_max = np.nanmax(dem_plot_array)
        rasterio_show(dem_plot_array, ax=ax_2d, cmap=topo_cmap, transform=dem_transform, zorder=0, alpha=0.7, vmin=dem_min, vmax=dem_max)
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(dem_plot_array, vert_exag=3)
        ax_2d.imshow(hillshade, cmap='gray', alpha=0.3, extent=[dem_transform[2], dem_transform[2] + dem_transform[0] * dem_plot_array.shape[1], dem_transform[5] + dem_transform[4] * dem_plot_array.shape[0], dem_transform[5]], zorder=1)
        update_flood_frame_func(
            frame, dem_plot_array, dem_transform, water_source_point, dam_points, ax_2d,
            rasterio_show, topo_cmap, new_cmap, progress_bar, water_elevations, flooded_areas, num_frames, flood_frames[frame], None, None
        )
        # Show flooded areas in blue
        flood_mask = np.ma.masked_where(~flood_frames[frame], flood_frames[frame])
        ax_2d.imshow(flood_mask, cmap=flood_cmap, alpha=0.6, extent=[dem_transform[2], dem_transform[2] + dem_transform[0] * dem_array.shape[1], dem_transform[5] + dem_transform[4] * dem_array.shape[0], dem_transform[5]], zorder=2)

        # Plot water elevation changes on the graph
        ax_elevation.clear()
        ax_elevation.plot(water_elevations, color='blue', label='Ūdens līmenis')
        ax_elevation.set_xlabel('Kadrs')
        ax_elevation.set_ylabel('Ūdens līmenis (m)', color='blue')
        ax_elevation.tick_params(axis='y', labelcolor='blue')
        ax_elevation.set_title('Ūdens līmenis')

        # Plot flooded area changes on the graph
        ax_flooded_area.clear()
        ax_flooded_area.plot(flooded_areas, color='green', label='Nopludinātais laukums')
        ax_flooded_area.set_xlabel('Kadrs')
        ax_flooded_area.set_ylabel('Nopludinātais laukums (kvadrātmetri)', color='green')
        ax_flooded_area.tick_params(axis='y', labelcolor='green')
        ax_flooded_area.set_title('Nopludinātais laukums')

    ani = FuncAnimation(combined_fig, frame_func, frames=np.arange(0, num_frames), repeat=False)
    combined_gif_output_path = get_output_path('combined_flood_simulation.gif')
    ani.save(combined_gif_output_path, writer=PillowWriter(fps=FPS))
    progress_bar.close()
    logger.info(f"Combined animation saved: {combined_gif_output_path}")

    # Adjust layout to fix title spacing
    combined_fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Export the last frame as a JPEG
    jpeg_output_path = get_output_path('combined_flood_simulation_last_frame.jpeg')
    combined_fig.savefig(jpeg_output_path, dpi=150, format='jpeg')
    logger.info(f"Last frame exported as JPEG: {jpeg_output_path}")

    return combined_gif_output_path

def export_last_frame_as_jpeg(
    fig: Any, ax_2d: Any, ax_elevation: Any, ax_flooded_area: Any, 
    dem_array: np.ndarray, dem_transform: np.ndarray, water_source_point: tuple, 
    dam_points: list, topo_cmap: Any, new_cmap: Any, update_flood_frame_func: Any
) -> str:
    """
    Export the last frame of the combined flood simulation as a JPEG.

    Args:
        fig (Any): Matplotlib figure object.
        ax_2d (Any): Matplotlib axis for 2D plot.
        ax_elevation (Any): Matplotlib axis for water elevation graph.
        ax_flooded_area (Any): Matplotlib axis for flooded area graph.
        dem_array (np.ndarray): Digital elevation model array.
        dem_transform (np.ndarray): Transformation parameters for DEM.
        water_source_point (tuple): Coordinates of the water source point.
        dam_points (list): List of dam points.
        topo_cmap (Any): Colormap for topography.
        new_cmap (Any): Colormap for flood visualization.
        update_flood_frame_func (Any): Function to update flood frames.

    Returns:
        str: Path to the exported JPEG file.
    """
    # Mask -9999 values as np.nan for plotting
    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)

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
    num_frames = getattr(builtins, 'num_frames', 100)

    # Update the last frame without a progress bar
    ax_2d.clear()
    dem_min = np.nanmin(dem_plot_array)
    dem_max = np.nanmax(dem_plot_array)
    rasterio_show(dem_plot_array, ax=ax_2d, cmap=topo_cmap, transform=dem_transform, zorder=0, alpha=0.7, vmin=dem_min, vmax=dem_max)
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_plot_array, vert_exag=3)
    ax_2d.imshow(hillshade, cmap='gray', alpha=0.3, extent=[dem_transform[2], dem_transform[2] + dem_transform[0] * dem_plot_array.shape[1], dem_transform[5] + dem_transform[4] * dem_plot_array.shape[0], dem_transform[5]], zorder=1)
    update_flood_frame_func(
        num_frames, dem_plot_array, dem_transform, water_source_point, dam_points, ax_2d,
        rasterio_show, topo_cmap, new_cmap, None, water_elevations, flooded_areas, num_frames, flood_array, visited, queue
    )

    # Plot water elevation changes on the graph
    ax_elevation.clear()
    ax_elevation.plot(water_elevations, color='blue', label='Ūdens līmenis')
    ax_elevation.set_xlabel('Kadrs')
    ax_elevation.set_ylabel('Ūdens līmenis (m)', color='blue')
    ax_elevation.tick_params(axis='y', labelcolor='blue')
    ax_elevation.set_title('Ūdens līmeņa izmaiņas laika gaitā')

    # Plot flooded area changes on the graph
    ax_flooded_area.clear()
    ax_flooded_area.plot(flooded_areas, color='green', label='Nopludinātais laukums')
    ax_flooded_area.set_xlabel('Kadrs')
    ax_flooded_area.set_ylabel('Nopludinātais laukums (kvadrātmetri)', color='green')
    ax_flooded_area.tick_params(axis='y', labelcolor='green')
    ax_flooded_area.set_title('Nopludinātā laukuma izmaiņas laika gaitā')

    # Save the last frame as a JPEG
    jpeg_output_path = get_output_path('combined_flood_simulation_last_frame.jpeg')
    fig.savefig(jpeg_output_path, dpi=150, format='jpeg')
    logger.info(f"Last frame exported as JPEG: {jpeg_output_path}")

    return jpeg_output_path

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
    ax2.plot(flooded_areas, color='green', label='Nopludinātais laukums')
    ax2.set_ylabel('Nopludinātais laukums (kvadrātmetri)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title and legend
    plt.title('Ūdens līmenis un nopludinātais laukums laika gaitā')
    fig.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

    print(f"Water elevation plot saved to {output_path}")

def export_final_water_depth_png(water_depth, dem_transform, output_path):
    """
    Export the final water depth as a PNG image.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    extent = [dem_transform[2], dem_transform[2] + dem_transform[0] * water_depth.shape[1],
              dem_transform[5] + dem_transform[4] * water_depth.shape[0], dem_transform[5]]
    im = plt.imshow(water_depth, cmap='Blues', extent=extent, origin='upper')
    plt.colorbar(im, label='Ūdens dziļums (m)')
    plt.title('Gala ūdens dziļums')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def export_dem_water_drainage_png(dem_array, dem_transform, water_depth, drainage_gdfs, output_path):

    # Mask -9999 values as np.nan for plotting
    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    vmin = float(np.nanmin(dem_plot_array))
    vmax = float(np.nanmax(dem_plot_array))
    rasterio_show(dem_plot_array, ax=ax, transform=dem_transform, cmap='gist_earth', vmin=vmin, vmax=vmax, zorder=1)
    # Water overlay
    extent = [dem_transform[2], dem_transform[2] + dem_transform[0] * dem_array.shape[1],
              dem_transform[5] + dem_transform[4] * dem_array.shape[0], dem_transform[5]]
    ax.imshow(water_depth, cmap='Blues', alpha=0.7, extent=extent, origin='upper', zorder=2)
    # --- Connected flooded regions (keep only for PNG) ---
    flooded = water_depth > 0.01
    labeled, num = ndi.label(flooded)
    for region in range(1, num + 1):
        mask = labeled == region
        if np.sum(mask) < 20:
            continue
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            y, x = contour[:, 0], contour[:, 1]
            x_coords = dem_transform[2] + x * dem_transform[0]
            y_coords = dem_transform[5] + y * dem_transform[4]
            ax.plot(x_coords, y_coords, color='red', linewidth=1.5, zorder=20)
    # Drainage tiles (list of GeoDataFrames)
    colors = ['#0000FF', '#00FFFF', '#FF00FF', '#FFA500']
    for gdf, color in zip(drainage_gdfs, colors):
        if gdf is not None and not gdf.empty:
            gdf.plot(ax=ax, color=color, linewidth=2, alpha=0.9, zorder=10)
    ax.set_title('DEM ar ūdeni un drenāžas elementiem')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def create_water_depth_gif(water_depth_frames, dem_transform, output_path, fps=5):
    """
    Create a GIF showing the progression of water depth over time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [dem_transform[2], dem_transform[2] + dem_transform[0] * water_depth_frames[0].shape[1],
              dem_transform[5] + dem_transform[4] * water_depth_frames[0].shape[0], dem_transform[5]]
    def update(frame):
        ax.clear()
        im = ax.imshow(water_depth_frames[frame], cmap='Blues', extent=extent, origin='upper')
        ax.set_title(f'Ūdens dziļums (Kadrs {frame})')
        return [im]
    ani = FuncAnimation(fig, update, frames=len(water_depth_frames), blit=False)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def create_dem_water_drainage_gif(dem_array, dem_transform, water_depth_frames, drainage_gdfs, output_path, fps=5):
    """
    Create a GIF showing the progression of DEM+water+drainage tiles over time,
    with outlines of contiguous flooded regions.
    """
    # Mask -9999 values as np.nan for plotting
    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)

    fig, ax = plt.subplots(figsize=(10, 8))
    vmin = float(np.nanmin(dem_plot_array))
    vmax = float(np.nanmax(dem_plot_array))
    extent = [dem_transform[2], dem_transform[2] + dem_transform[0] * dem_plot_array.shape[1],
              dem_transform[5] + dem_transform[4] * dem_plot_array.shape[0], dem_transform[5]]
    colors = ['#0000FF', '#00FFFF', '#FF00FF', '#FFA500']

    progress_bar = tqdm(total=len(water_depth_frames), desc="Creating DEM+Water+Drainage GIF")

    def update(frame):
        ax.clear()
        rasterio_show(dem_plot_array, ax=ax, transform=dem_transform, cmap='gist_earth', vmin=vmin, vmax=vmax, zorder=1)
        ax.imshow(water_depth_frames[frame], cmap='Blues', alpha=0.7, extent=extent, origin='upper', zorder=2)
        # --- Drainage overlays ---
        for gdf, color in zip(drainage_gdfs, colors):
            if gdf is not None and not gdf.empty:
                gdf.plot(ax=ax, color=color, linewidth=2, alpha=0.9, zorder=10)
        ax.set_title(f'DEM + Ūdens + Drenāža (Kadrs {frame})')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        progress_bar.update(1)
        return []

    ani = FuncAnimation(fig, update, frames=len(water_depth_frames), blit=False)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    progress_bar.close()
    plt.close(fig)
    
def render_dem_water_drainage_frame(args):
    frame, dem_plot_array, dem_transform, water_depth_frames, drainage_gdfs, colors, extent, vmin, vmax, tmp_dir = args
    import matplotlib.pyplot as plt
    from rasterio.plot import show as rasterio_show
    import numpy as np
    import scipy.ndimage as ndi
    from skimage import measure

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.clear()
    rasterio_show(dem_plot_array, ax=ax, transform=dem_transform, cmap='gist_earth', vmin=vmin, vmax=vmax, zorder=1)
    ax.imshow(water_depth_frames[frame], cmap='Blues', alpha=0.7, extent=extent, origin='upper', zorder=2)

    ax.grid(True, linestyle='--', alpha=0.5)
    for gdf, color in zip(drainage_gdfs, colors):
        if gdf is not None and not gdf.empty:
            gdf.plot(ax=ax, color=color, linewidth=2, alpha=0.9, zorder=10)
    ax.set_title(f'DEM + Ūdens + Drenāža (kadrs {frame})')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    out_path = os.path.join(tmp_dir, f"frame_{frame:04d}.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path

def create_dem_water_drainage_gif_parallel(dem_array, dem_transform, water_depth_frames, drainage_gdfs, output_path, fps=5):
    import tempfile
    import shutil

    dem_plot_array = np.where(dem_array == -9999, np.nan, dem_array)
    vmin = float(np.nanmin(dem_plot_array))
    vmax = float(np.nanmax(dem_plot_array))
    extent = [dem_transform[2], dem_transform[2] + dem_transform[0] * dem_plot_array.shape[1],
              dem_transform[5] + dem_transform[4] * dem_plot_array.shape[0], dem_transform[5]]
    colors = ['#0000FF', '#00FFFF', '#FF00FF', '#FFA500']

    tmp_dir = tempfile.mkdtemp()
    try:
        args_list = [
            (frame, dem_plot_array, dem_transform, water_depth_frames, drainage_gdfs, colors, extent, vmin, vmax, tmp_dir)
            for frame in range(len(water_depth_frames))
        ]
        with ProcessPoolExecutor() as executor:
            frame_paths = list(executor.map(render_dem_water_drainage_frame, args_list))
        # Assemble GIF
        images = [imageio.imread(fp) for fp in sorted(frame_paths)]
        imageio.mimsave(output_path, images, fps=fps)
    finally:
        shutil.rmtree(tmp_dir)