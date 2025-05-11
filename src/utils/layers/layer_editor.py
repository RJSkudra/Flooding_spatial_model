"""
WMS Layer Editor GUI for editing and previewing wms_layers.json.
"""

import os
import json
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, QColorDialog, QCheckBox, QListWidget, QListWidgetItem, QMessageBox, QFormLayout, QFileDialog, QDialog, QDialogButtonBox, QRadioButton
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.FloodGIF_scripts.visualization_utils import plot_double_line, plot_half_filled_circle

WMS_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wms_layers.json')

class WmsLayerEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ģeotelpisko datu slāņu redaktors")
        self.setGeometry(200, 200, 1200, 800)
        self.layers = self.load_layers()
        self.current_layer_key = None
        self.init_ui()
        
    def closeEvent(self, event):
        # Properly close the Matplotlib figure to avoid updates after deletion
        try:
            if hasattr(self, 'fig'):
                import matplotlib.pyplot as plt
                plt.close(self.fig)
        except Exception:
            pass
        super().closeEvent(event)

    def load_layers(self):
        try:
            with open(WMS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"plot_order": []}

    def save_layers(self):
        try:
            with open(WMS_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.layers, f, indent=2)
            QMessageBox.information(self, "Saglabāts", "Slāņi veiksmīgi saglabāti.")
        except Exception as e:
            QMessageBox.critical(self, "Kļūda", f"Neizdevās saglabāt: {e}")
            
    def move_layer_up(self):
        row = self.layer_list.currentRow()
        if row > 0:
            key = self.layer_list.item(row).text()
            plot_order = self.layers.get("plot_order", [])
            plot_order[row], plot_order[row-1] = plot_order[row-1], plot_order[row]
            self.layers["plot_order"] = plot_order
            self.refresh_layer_list()
            self.layer_list.setCurrentRow(row-1)

    def move_layer_down(self):
        row = self.layer_list.currentRow()
        plot_order = self.layers.get("plot_order", [])
        if 0 <= row < len(plot_order)-1:
            key = self.layer_list.item(row).text()
            plot_order[row], plot_order[row+1] = plot_order[row+1], plot_order[row]
            self.layers["plot_order"] = plot_order
            self.refresh_layer_list()
            self.layer_list.setCurrentRow(row+1)
            
    def edit_layer_source(self):
        item = self.layer_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Nav izvēlēts", "Lūdzu, izvēlieties slāni.")
            return
        key = item.text()
        layer = self.layers.get(key, {})
        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Rediģēt slāņa avotu")
        layout = QVBoxLayout(dialog)

        if layer.get("type") == "wms":
            form = QFormLayout()
            url_edit = QLineEdit(layer.get("url", ""))
            layers_edit = QLineEdit(layer.get("layers", ""))
            format_edit = QLineEdit(layer.get("format", ""))
            version_edit = QLineEdit(layer.get("version", ""))
            styles_edit = QLineEdit(layer.get("styles", ""))
            crs_edit = QLineEdit(layer.get("crs", ""))
            filter_field_edit = QLineEdit(layer.get("filter", {}).get("field", ""))
            filter_values_edit = QLineEdit(
                ", ".join(layer.get("filter", {}).get("values", []))
            )
            form.addRow("URL", url_edit)
            form.addRow("Layers", layers_edit)
            form.addRow("Format", format_edit)
            form.addRow("Version", version_edit)
            form.addRow("Styles", styles_edit)
            form.addRow("CRS", crs_edit)
            form.addRow("Filtra lauks", filter_field_edit)
            form.addRow("Filtra vērtības (atdalītas ar komatu)", filter_values_edit)
            layout.addLayout(form)
        elif layer.get("type") == "shapefile":
            form = QFormLayout()
            path_edit = QLineEdit(layer.get("path", ""))
            browse_btn = QPushButton("Izvēlēties...")
            filter_field_edit = QLineEdit(layer.get("filter", {}).get("field", ""))
            filter_values_edit = QLineEdit(
                ", ".join(layer.get("filter", {}).get("values", []))
            )
            def browse_shp():
                path, _ = QFileDialog.getOpenFileName(dialog, "Izvēlieties lokālu failu", "", "Shapefiles (*.shp)")
                if path:
                    path_edit.setText(path)
            browse_btn.clicked.connect(browse_shp)
            form.addRow("Ceļš", path_edit)
            form.addRow("", browse_btn)
            form.addRow("Filtra lauks", filter_field_edit)
            form.addRow("Filtra vērtības (atdalītas ar komatu)", filter_values_edit)
            layout.addLayout(form)
        else:
            QMessageBox.information(self, "Nav atbalstīts", "Šo slāņa tipu nevar rediģēt šādi.")
            return

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apstiprināt")
        buttons.button(QDialogButtonBox.Cancel).setText("Atcelt")
        layout.addWidget(buttons)

        def accept():
            if layer.get("type") == "wms":
                layer["url"] = url_edit.text().strip()
                layer["layers"] = layers_edit.text().strip()
                layer["format"] = format_edit.text().strip()
                layer["version"] = version_edit.text().strip()
                layer["styles"] = styles_edit.text().strip()
                layer["crs"] = crs_edit.text().strip()
                filter_field = filter_field_edit.text().strip()
                filter_values = [v.strip() for v in filter_values_edit.text().split(",") if v.strip()]
                if filter_field and filter_values:
                    layer["filter"] = {"field": filter_field, "values": filter_values}
                elif "filter" in layer:
                    del layer["filter"]
            elif layer.get("type") == "shapefile":
                path = path_edit.text().strip()
                if not path:
                    QMessageBox.warning(dialog, "Trūkst ceļa", "Lūdzu, izvēlieties shapefile.")
                    return
                config_dir = os.path.dirname(WMS_CONFIG_FILE)
                rel_path = os.path.relpath(path, config_dir)
                layer["path"] = rel_path
                filter_field = filter_field_edit.text().strip()
                filter_values = [v.strip() for v in filter_values_edit.text().split(",") if v.strip()]
                if filter_field and filter_values:
                    layer["filter"] = {"field": filter_field, "values": filter_values}
                elif "filter" in layer:
                    del layer["filter"]
            self.layers[key] = layer
            dialog.accept()
            QMessageBox.information(self, "Saglabāts", "Slāņa avots atjaunināts.")

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec_()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Layer list and controls
        left_layout = QVBoxLayout()
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.on_layer_selected)
        left_layout.addWidget(QLabel("Slāņi:"))
        left_layout.addWidget(self.layer_list)
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Pievienot slāni")
        add_btn.clicked.connect(self.add_layer)
        remove_btn = QPushButton("Noņemt slāni")
        remove_btn.clicked.connect(self.remove_layer)
        move_up_btn = QPushButton("↑")
        move_down_btn = QPushButton("↓")
        move_up_btn.clicked.connect(self.move_layer_up)
        move_down_btn.clicked.connect(self.move_layer_down)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(move_up_btn)
        btn_layout.addWidget(move_down_btn)
        left_layout.addLayout(btn_layout)
        save_btn = QPushButton("Saglabāt izmaiņas")
        save_btn.clicked.connect(self.save_layers)
        left_layout.addWidget(save_btn)
        edit_source_btn = QPushButton("Rediģēt avotu")
        edit_source_btn.clicked.connect(self.edit_layer_source)
        btn_layout.addWidget(edit_source_btn)
        main_layout.addLayout(left_layout, 2)

        # Center: Layer property editor
        center_layout = QVBoxLayout()
        self.form = QFormLayout()
        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["wms", "shapefile", "lidar"])
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["point", "line", "polygon", "none"])
        self.alpha_edit = QLineEdit()
        self.legend_check = QCheckBox("Rādīt leģendā")
        self.fill_color_btn = QPushButton("Iestatīt aizpildījuma krāsu")
        self.fill_color_btn.clicked.connect(lambda: self.pick_color(self.fill_color_btn))
        self.line_color_btn = QPushButton("Iestatīt līnijas krāsu")
        self.line_color_btn.clicked.connect(lambda: self.pick_color(self.line_color_btn))
        self.fill_color = None
        self.line_color = None
        self.form.addRow("Nosaukums", self.name_edit)
        self.form.addRow("Tips", self.type_combo)
        self.form.addRow("Ģeometrija", self.geometry_combo)
        self.form.addRow("Caurredzamība (0-1)", self.alpha_edit)
        self.form.addRow(self.legend_check)
        self.form.addRow(self.fill_color_btn)
        self.form.addRow(self.line_color_btn)
        center_layout.addLayout(self.form)
        update_btn = QPushButton("Saglabāt slāņa īpašības")
        update_btn.clicked.connect(self.update_layer_properties)
        center_layout.addWidget(update_btn)
        main_layout.addLayout(center_layout, 2)

        # Right: Preview
        right_layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        right_layout.addWidget(QLabel("Slāņa priekšskatījums:"))
        right_layout.addWidget(self.canvas)
        preview_btn = QPushButton("Priekšskatīt slāni")
        preview_btn.clicked.connect(self.preview_layer)
        right_layout.addWidget(preview_btn)
        main_layout.addLayout(right_layout, 3)

        self.refresh_layer_list()

    def refresh_layer_list(self):
        self.layer_list.clear()
        for key in self.layers.get("plot_order", []):
            item = QListWidgetItem(key)
            self.layer_list.addItem(item)

    def add_layer(self):
        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Pievienot jaunu slāni")
        layout = QVBoxLayout(dialog)
        type_form = QFormLayout()
        wms_radio = QRadioButton("Tīmekļa slānis")
        shp_radio = QRadioButton("Lokāls slānis")
        wms_radio.setChecked(True)
        type_form.addRow(wms_radio)
        type_form.addRow(shp_radio)
        layout.addLayout(type_form)
        wms_fields = {}
        shp_fields = {}
        # WMS fields
        wms_form = QFormLayout()
        wms_fields['name'] = QLineEdit()
        wms_fields['url'] = QLineEdit("https://lvmgeoserver.lvm.lv/geoserver/ows")
        wms_fields['layers'] = QLineEdit()
        wms_fields['format'] = QLineEdit("image/png")
        wms_fields['version'] = QLineEdit("1.0.0")
        wms_fields['styles'] = QLineEdit("raster")
        wms_fields['crs'] = QLineEdit("EPSG:3059")
        wms_fields['filter_field'] = QLineEdit()
        wms_fields['filter_values'] = QLineEdit()
        for k, v in wms_fields.items():
            if k == 'name':
                wms_form.addRow("Nosaukums", v)
            elif k in ['filter_field', 'filter_values']:
                label = "Filtra lauks" if k == 'filter_field' else "Filtra vērtības (atdalītas ar komatu)"
                wms_form.addRow(label, v)
            else:
                wms_form.addRow(k.capitalize(), v)
        # Shapefile fields
        shp_form = QFormLayout()
        shp_fields['name'] = QLineEdit()
        shp_fields['path'] = QLineEdit()
        shp_browse = QPushButton("Izvēlēties...")
        def browse_shp():
            path, _ = QFileDialog.getOpenFileName(dialog, "Izvēlieties lokālu failu", "", "Shapefiles (*.shp)")
            if path:
                shp_fields['path'].setText(path)
        shp_browse.clicked.connect(browse_shp)
        shp_form.addRow("Nosaukums", shp_fields['name'])
        shp_form.addRow("Ceļš", shp_fields['path'])
        shp_form.addRow("", shp_browse)
        shp_fields['filter_field'] = QLineEdit()
        shp_fields['filter_values'] = QLineEdit()
        shp_form.addRow("Filtra lauks", shp_fields['filter_field'])
        shp_form.addRow("Filtra vērtības (atdalītas ar komatu)", shp_fields['filter_values'])
        # Stacked layout
        wms_form_widget = QWidget()
        wms_form_widget.setLayout(wms_form)
        shp_form_widget = QWidget()
        shp_form_widget.setLayout(shp_form)
        layout.addWidget(wms_form_widget)
        layout.addWidget(shp_form_widget)
        wms_form_widget.show()
        shp_form_widget.hide()
        def toggle_forms():
            if wms_radio.isChecked():
                wms_form_widget.show()
                shp_form_widget.hide()
            else:
                wms_form_widget.hide()
                shp_form_widget.show()
        wms_radio.toggled.connect(toggle_forms)
        shp_radio.toggled.connect(toggle_forms)
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apstiprināt")
        buttons.button(QDialogButtonBox.Cancel).setText("Atcelt")
        layout.addWidget(buttons)
        def accept():
            if wms_radio.isChecked():
                key = wms_fields['name'].text().strip()
                if not key:
                    QMessageBox.warning(dialog, "Trūkst nosaukuma", "Lūdzu, ievadiet nosaukumu slānim.")
                    return
                if key in self.layers:
                    QMessageBox.warning(dialog, "Eksistē", "Slāņa atslēga jau pastāv.")
                    return
                filter_field = wms_fields['filter_field'].text().strip()
                filter_values = [v.strip() for v in wms_fields['filter_values'].text().split(',') if v.strip()]
                self.layers[key] = {
                    "type": "wms",
                    "name": key,
                    "url": wms_fields['url'].text().strip(),
                    "layers": wms_fields['layers'].text().strip(),
                    "format": wms_fields['format'].text().strip(),
                    "version": wms_fields['version'].text().strip(),
                    "styles": wms_fields['styles'].text().strip(),
                    "crs": wms_fields['crs'].text().strip(),
                    "geometry_type": None,
                    "fill_color": None,
                    "line_color": None,
                    "alpha": 1.0,
                    "legend": False
                }
                if filter_field and filter_values:
                    self.layers[key]["filter"] = {"field": filter_field, "values": filter_values}
            else:
                key = shp_fields['name'].text().strip()
                if not key:
                    QMessageBox.warning(dialog, "Trūkst nosaukums", "Lūdzu, ievadiet nosaukumu shapefile slānim.")
                    return
                if key in self.layers:
                    QMessageBox.warning(dialog, "Eksistē", "Slāņa atslēga jau eksistē.")
                    return
                path = shp_fields['path'].text().strip()
                if not path:
                    QMessageBox.warning(dialog, "Trūkst ceļa", "Lūdzu, izvēlieties shapefile.")
                    return
                # Convert to relative path for portability
                config_dir = os.path.dirname(WMS_CONFIG_FILE)
                rel_path = os.path.relpath(path, config_dir)
                filter_field = shp_fields['filter_field'].text().strip()
                filter_values = [v.strip() for v in shp_fields['filter_values'].text().split(',') if v.strip()]
                self.layers[key] = {
                    "type": "shapefile",
                    "name": key,
                    "path": rel_path,
                    "geometry_type": None,
                    "fill_color": None,
                    "line_color": None,
                    "alpha": 1.0,
                    "legend": False
                }
                if filter_field and filter_values:
                    self.layers[key]["filter"] = {"field": filter_field, "values": filter_values}
            self.layers.setdefault("plot_order", []).append(key)
            dialog.accept()
            self.refresh_layer_list()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec_()

    def remove_layer(self):
        item = self.layer_list.currentItem()
        if not item:
            return
        key = item.text()
        if key in self.layers:
            del self.layers[key]
        if key in self.layers.get("plot_order", []):
            self.layers["plot_order"].remove(key)
        self.refresh_layer_list()
        self.current_layer_key = None

    def pick_color(self, btn):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            btn.setStyleSheet(f"background:{hex_color}")
            if btn == self.fill_color_btn:
                self.fill_color = hex_color
            elif btn == self.line_color_btn:
                self.line_color = hex_color

    def on_layer_selected(self, item):
        key = item.text()
        self.current_layer_key = key
        layer = self.layers.get(key, {})
        self.name_edit.setText(layer.get("name", key))
        self.type_combo.setCurrentText(layer.get("type", "wms"))
        self.geometry_combo.setCurrentText(str(layer.get("geometry_type", "none")))
        self.alpha_edit.setText(str(layer.get("alpha", 1.0)))
        self.legend_check.setChecked(bool(layer.get("legend", False)))
        self.fill_color = layer.get("fill_color", None)
        self.line_color = layer.get("line_color", None)
        self.fill_color_btn.setStyleSheet(f"background:{self.fill_color if self.fill_color else '#ffffff'}")
        self.line_color_btn.setStyleSheet(f"background:{self.line_color if self.line_color else '#000000'}")
        # Show WMS connection fields if WMS
        if layer.get("type") == "wms":
            if not hasattr(self, 'wms_conn_fields'):
                self.wms_conn_fields = {}
                self.wms_conn_form = QFormLayout()
                for field in ["url", "layers", "format", "version", "styles", "crs"]:
                    self.wms_conn_fields[field] = QLineEdit()
                    self.wms_conn_form.addRow(field.capitalize(), self.wms_conn_fields[field])
                #self.form.addRow(QLabel("WMS Connection Settings:"))
                #for field in self.wms_conn_fields.values():
                #    self.form.addRow(field)
            for field in ["url", "layers", "format", "version", "styles", "crs"]:
                val = layer.get(field, "")
                self.wms_conn_fields[field].setText(str(val) if val is not None else "")
                self.wms_conn_fields[field].show()
        else:
            if hasattr(self, 'wms_conn_fields'):
                for field in self.wms_conn_fields.values():
                    field.hide()

    def update_layer_properties(self):
        key = self.current_layer_key
        if not key:
            return
        layer = self.layers.get(key, {})
        layer["name"] = self.name_edit.text()
        layer["type"] = self.type_combo.currentText()
        layer["geometry_type"] = self.geometry_combo.currentText()
        try:
            layer["alpha"] = float(self.alpha_edit.text())
        except Exception:
            layer["alpha"] = 1.0
        layer["legend"] = self.legend_check.isChecked()
        layer["fill_color"] = self.fill_color
        layer["line_color"] = self.line_color
        # Save WMS connection fields if WMS
        if layer["type"] == "wms" and hasattr(self, 'wms_conn_fields'):
            for field in ["url", "layers", "format", "version", "styles", "crs"]:
                layer[field] = self.wms_conn_fields[field].text().strip()
        self.layers[key] = layer
        QMessageBox.information(self, "Updated", f"Layer '{key}' updated.")

    def preview_layer(self):
        key = self.current_layer_key
        if not key:
            return
        layer = self.layers.get(key, {})
        self.ax.clear()
        geom = layer.get("geometry_type", "none")
        alpha = float(layer.get("alpha", 1.0))
        fill = layer.get("fill_color", "#cccccc")
        line = layer.get("line_color", "#000000")
        if geom == "polygon":
            poly = mpatches.Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]), closed=True, facecolor=fill, edgecolor=line, alpha=alpha)
            self.ax.add_patch(poly)
            self.ax.set_xlim(-0.5, 1.5)
            self.ax.set_ylim(-0.5, 1.5)
        elif geom == "line":
            self.ax.plot([0,1], [0,1], color=line, alpha=alpha, linewidth=3)
            self.ax.set_xlim(-0.5, 1.5)
            self.ax.set_ylim(-0.5, 1.5)
        elif geom == "point":
            self.ax.scatter([0.5], [0.5], color=fill, alpha=alpha, s=200, edgecolor=line)
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        if key == "drenu_savienojumi" and layer.get("marker_style") == "half_filled":
            plot_half_filled_circle(self.ax, 0.5, 0.5, size=0.4, color=fill, outline_color=line, outline_width=2)
        elif key in ["drenu_kolektori", "drenu_kolektori_lieli"] and layer.get("line_style") == "double":
            plot_double_line(self.ax, [(0, 0.5), (1, 0.5)], color=line, width=3, offset=0.05, alpha=alpha)
        else:
            self.ax.text(0.5, 0.5, "", ha='center', va='center')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        if layer.get("legend", False):
            self.ax.legend([layer.get("name", key)])
        self.ax.set_title(f"Priekšskatījums: {layer.get('name', key)}")
        self.canvas.draw()
