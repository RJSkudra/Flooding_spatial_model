"""
UI utilities for flood simulation and DEM editing.
"""

import builtins
import logging
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QCheckBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from typing import Any

logger = logging.getLogger(__name__)

class DrawingWindow(QMainWindow):
    def __init__(
        self,
        fig: Any,
        ax: Any,
        dem_array: np.ndarray,
        dem_data: Any,
        topo_cmap: str,
        enable_dam_drawing: Any,
        enable_source_selection: Any,
        update_dem_with_dam: Any,
        create_animation: Any,
        num_frames: int,
        update_num_frames=None,
        layers_dict=None,
        plot_order=None,
        wms_images=None,
        shp_geoms=None
    ) -> None:
        """
        Initialize the DrawingWindow.

        Args:
            fig: Matplotlib figure object.
            ax: Matplotlib axes object.
            dem_array: DEM array data.
            dem_data: DEM metadata.
            topo_cmap: Colormap for DEM visualization.
            enable_dam_drawing: Callback for enabling dam drawing.
            enable_source_selection: Callback for enabling source selection.
            update_dem_with_dam: Callback for updating DEM with dam.
            create_animation: Callback for creating animation.
            num_frames: Number of frames for the simulation.
            update_num_frames: Optional callback for updating the number of frames.
            layers_dict: Dictionary of layers for visualization.
            plot_order: Order of layers to be plotted.
            wms_images: Dictionary of WMS images for layers.
            shp_geoms: Dictionary of shapefile geometries for layers.
        """
        super().__init__()
        self.setWindowTitle("Plūdu simulācija")
        self.setGeometry(100, 100, 800, 600)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        canvas = FigureCanvasQTAgg(fig)
        main_layout.addWidget(canvas)
        self.layers_dict = layers_dict
        self.plot_order = plot_order
        self.wms_images = wms_images
        self.shp_geoms = shp_geoms
        self.ax = ax
        self.fig = fig
        self.dem_array = dem_array
        self.dem_data = dem_data
        self.topo_cmap = topo_cmap
        ax.clear()
        from rasterio.plot import show
        show(dem_array, ax=ax, cmap=topo_cmap, transform=dem_data.transform)
        ax.set_xlim(dem_data.bounds.left, dem_data.bounds.right)
        ax.set_ylim(dem_data.bounds.bottom, dem_data.bounds.top)
        x_ticks = np.linspace(dem_data.bounds.left, dem_data.bounds.right, 5)
        y_ticks = np.linspace(dem_data.bounds.bottom, dem_data.bounds.top, 5)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{x:.0f}" for x in x_ticks])
        ax.set_yticklabels([f"{y:.0f}" for y in y_ticks])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title('DEM Vizualizācija (LKS-97 Koordinātes)', fontsize=12)
        ax.set_xlabel('Horizontāle (m)')
        ax.set_ylabel('Vertikāle (m)')
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(dem_array, vert_exag=3)
        ax.imshow(hillshade, cmap='gray', alpha=0.3, extent=[
            dem_data.bounds.left, dem_data.bounds.right,
            dem_data.bounds.bottom, dem_data.bounds.top
        ])
        instructions_label = QLabel("Izvēlamies 2 punktus aizsprostam, 1 punktu ūdens avotam!")
        main_layout.addWidget(instructions_label)
        # Add WMS layer toggles and legend if provided
        if self.layers_dict and self.plot_order:
            wms_label = QLabel("WMS/Slāņu pārslēgšana un leģenda:")
            main_layout.addWidget(wms_label)
            self.wms_checkboxes = {}
            wms_box = QHBoxLayout()
            self.layer_visible = {k: True for k in self.plot_order}
            legend_elements = []
            import matplotlib.patches as mpatches
            for key in self.plot_order:
                layer = self.layers_dict.get(key, {})
                cb = QCheckBox(layer.get('name', key))
                cb.setChecked(True)
                cb.stateChanged.connect(self.make_toggle(key))
                self.wms_checkboxes[key] = cb
                wms_box.addWidget(cb)
                # Add to legend if requested
                if layer.get('legend', False):
                    color = layer.get('fill_color') or layer.get('line_color') or '#888888'
                    patch = mpatches.Patch(color=color, label=layer.get('name', key), alpha=layer.get('alpha', 1.0))
                    legend_elements.append(patch)
            main_layout.addLayout(wms_box)
            self.update_background()
        frames_layout = QHBoxLayout()
        main_layout.addLayout(frames_layout)
        frames_label = QLabel("Simulācijas kadru skaits:")
        frames_layout.addWidget(frames_label)
        self.frames_entry = QLineEdit(str(num_frames))
        frames_layout.addWidget(self.frames_entry)
        dam_button = QPushButton("Zīmēt Aizsprostu")
        dam_button.clicked.connect(enable_dam_drawing)
        main_layout.addWidget(dam_button)
        source_button = QPushButton("Iestatīt Ūdens Avotu")
        source_button.clicked.connect(enable_source_selection)
        main_layout.addWidget(source_button)
        finish_button = QPushButton("Apstiprināt izvēli")
        finish_button.clicked.connect(self.on_close)
        main_layout.addWidget(finish_button)
        self.update_dem_with_dam = update_dem_with_dam
        self.create_animation = create_animation
        self.num_frames = num_frames

        

    def make_toggle(self, key):
        def toggle(state):
            self.layer_visible[key] = bool(state)
            self.update_background()
        return toggle

    def update_background(self):
        ax = self.ax
        ax.clear()
        legend_elements = []
        # Draw overlays in plot_order
        for idx, key in enumerate(self.plot_order):
            if not self.layer_visible.get(key, False):
                continue
            layer = self.layers_dict.get(key, {})
            # WMS/ArcGIS tile
            if self.wms_images and key in self.wms_images and self.wms_images[key] is not None:
                ax.imshow(self.wms_images[key], extent=[self.dem_data.bounds.left, self.dem_data.bounds.right, self.dem_data.bounds.bottom, self.dem_data.bounds.top], alpha=layer.get('alpha', 1.0), zorder=idx+1)
            # Shapefile
            elif self.shp_geoms and key in self.shp_geoms and self.shp_geoms[key] is not None:
                gdf = self.shp_geoms[key]
                if not gdf.empty:
                    style = {
                        'zorder': idx+1,
                        'alpha': layer.get('alpha', 1.0),
                        'edgecolor': layer.get('line_color', 'black'),
                        'facecolor': layer.get('fill_color', 'none')
                    }
                    gdf.plot(ax=ax, **style)
                    # Add to legend if requested
                    if layer.get('legend', False):
                        import matplotlib.patches as mpatches
                        color = layer.get('fill_color') or layer.get('line_color') or '#888888'
                        patch = mpatches.Patch(color=color, label=layer.get('name', key), alpha=layer.get('alpha', 1.0))
                        legend_elements.append(patch)
        # Always show DEM and hillshade
            from rasterio.plot import show
            im = show(self.dem_array, ax=ax, cmap=self.topo_cmap, transform=self.dem_data.transform, zorder=0, alpha=0.7)
            from matplotlib.colors import LightSource
            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(self.dem_array, vert_exag=3)
            ax.imshow(hillshade, cmap='gray', alpha=0.3, extent=[self.dem_data.bounds.left, self.dem_data.bounds.right, self.dem_data.bounds.bottom, self.dem_data.bounds.top], zorder=100)
            ax.set_xlim(self.dem_data.bounds.left, self.dem_data.bounds.right)
            ax.set_ylim(self.dem_data.bounds.bottom, self.dem_data.bounds.top)
            ax.set_title('DEM Vizualizācija (LKS-97 Koordinātes)', fontsize=12)
            ax.set_xlabel('Horizontāle (m)')
            ax.set_ylabel('Vertikāle (m)')
        # Add the legend to the plot
        if legend_elements:
            leg = ax.legend(
                handles=legend_elements,
                loc='center right',
                bbox_to_anchor=(-0.25, 0.5),  # Negative value puts it outside the axes on the left
                borderaxespad=0
            )
            # Colorbar for DEM
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(cmap=self.topo_cmap, norm=mpl.colors.Normalize(
            vmin=float(np.nanmin(self.dem_array)),
            vmax=float(np.nanmax(self.dem_array))
        ))
        sm._A = []
        cbar_ax = self.fig.add_axes([0.87, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
        self.fig.colorbar(sm, cax=cbar_ax, label='DEM augstums (m)')

        self.fig.canvas.draw_idle()

    def on_close(self) -> None:
        """
        Handle the close event, update the number of frames, and trigger callbacks.
        """
        try:
            self.num_frames = int(self.frames_entry.text())
            if self.num_frames <= 0:
                self.num_frames = 100
        except ValueError:
            self.num_frames = 100
        builtins.num_frames = self.num_frames  # Ensure global num_frames is updated
        self.update_dem_with_dam()
        self.create_animation()
        self.close()
        logger.info("Drawing window closed.")