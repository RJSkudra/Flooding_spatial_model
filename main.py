"""
Main interface for selecting between DEM generation and flood simulation.
"""

import os
import sys
# Add the src directory to sys.path for correct imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from FloodGIF_scripts.Flood_GIF import main as flood_gif_main
from Lidar2Dem_scripts.Lidar2Dem import main as lidar2dem_main
from RainfallFloodSim_scripts.RainfallFloodSimUI import main as rainfall_flood_main
from utils.Flood.wms_layer_editor import WmsLayerEditorWindow
from RainfallFloodSim_scripts.RainfallFloodSimUI import RainfallFloodSimUI
import builtins

class MainInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galvenā sadaļa")

        # Main layout
        layout = QVBoxLayout()

        # Buttons
        self.dem_button = QPushButton("Ģenerēt DEM")
        self.simulation_button = QPushButton("Ģenerēt plūdu simulāciju")
        #self.rainfall_sim_button = QPushButton("Lietus plūdu simulācija")
        self.wms_editor_button = QPushButton("Rediģēt ģeotelpisko datu slāņus")

        # Connect buttons to actions
        self.dem_button.clicked.connect(self.run_lidar2dem)
        self.simulation_button.clicked.connect(self.run_flood_gif)
        #self.rainfall_sim_button.clicked.connect(self.run_rainfall_flood_sim)
        self.wms_editor_button.clicked.connect(self.run_wms_editor)

        # Add buttons to layout
        layout.addWidget(self.dem_button)
        layout.addWidget(self.simulation_button)
        #layout.addWidget(self.rainfall_sim_button)
        layout.addWidget(self.wms_editor_button)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_lidar2dem(self):
        """Run the Lidar2Dem module."""
        lidar2dem_main()

    def run_flood_gif(self):
        print("Flood simulation button clicked.")
        flood_gif_main(QApplication.instance())

    def run_rainfall_flood_sim(self):
        """Run the rainfall-based flood simulation module."""
        self.rainfall_window = RainfallFloodSimUI()
        self.rainfall_window.show()

    def run_wms_editor(self):
        self.wms_editor_window = WmsLayerEditorWindow()
        self.wms_editor_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainInterface()
    main_window.show()
    sys.exit(app.exec_())