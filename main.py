"""
Main interface for selecting between DEM generation and flood simulation.
"""

import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from src.FloodGIF_scripts.Flood_GIF import main as flood_gif_main
from src.Lidar2Dem_scripts.Lidar2Dem import main as lidar2dem_main
from src.utils.layers.layer_editor import WmsLayerEditorWindow

def ensure_data_folders():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dati")
    subfolders = [
        "dem_faili",
        "izejas_dati",
        "lidar_dati",
        "shp_dati",
        "tmp"
    ]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for sub in subfolders:
        sub_path = os.path.join(base_dir, sub)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

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
        self.layer_editor_button = QPushButton("Rediģēt ģeotelpisko datu slāņus")

        # Connect buttons to actions
        self.dem_button.clicked.connect(self.run_lidar2dem)
        self.simulation_button.clicked.connect(self.run_flood_gif)
        self.layer_editor_button.clicked.connect(self.run_layer_editor)

        # Add buttons to layout
        layout.addWidget(self.dem_button)
        layout.addWidget(self.simulation_button)
        layout.addWidget(self.layer_editor_button)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_lidar2dem(self):
        lidar2dem_main()

    def run_flood_gif(self):
        flood_gif_main(QApplication.instance())

    def run_layer_editor(self):
        self.layer_editor_window = WmsLayerEditorWindow()
        self.layer_editor_window.show()


if __name__ == "__main__":
    ensure_data_folders()
    app = QApplication(sys.argv)
    main_window = MainInterface()
    main_window.show()
    sys.exit(app.exec_())