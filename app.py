import sys
import json
import os
import subprocess
import shutil
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFormLayout, QLineEdit, QPushButton, QComboBox, QFileDialog, 
    QLabel, QMessageBox, QGroupBox, QCheckBox, QTabWidget, 
    QDoubleSpinBox, QSpinBox, QScrollArea, QFrame, QColorDialog, 
    QStyle, QDialog, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette, QAction, QKeySequence
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Circle
from matplotlib import cm
import matplotlib.animation as animation

# --- Custom UI Widgets ---

class ColorButton(QPushButton):
    """A button that shows a color swatch and opens a color picker."""
    def __init__(self, color):
        super().__init__()
        self.color = color
        self.set_color(color)
        self.clicked.connect(self.pick_color)
        self.setFixedWidth(40)

    def pick_color(self):
        c = QColorDialog.getColor(QColor(self.color))
        if c.isValid():
            self.set_color(c.name())

    def set_color(self, color):
        self.color = color
        self.setStyleSheet(f"background-color: {color}; border: 1px solid #555; border-radius: 3px;")

    def get_color(self):
        return self.color

class FunctionLayerWidget(QGroupBox):
    """A collapsible widget to hold all settings for a single function layer."""
    def __init__(self, index, default_func, default_color, default_label, remove_callback, duplicate_callback, get_global_bounds, move_up_callback, move_down_callback):
        super().__init__(f"Function {index}")
        self.setCheckable(True)
        self.setChecked(True)
        self.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 10px; } "
                           "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; } "
                           "QGroupBox::indicator { width: 12px; height: 12px; }")
        
        self._index = index
        self.get_global_bounds = get_global_bounds
        
        self.content_widget = QWidget()
        layout = QVBoxLayout(self.content_widget)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(5, 5, 5, 5)
        
        self.func_input = QLineEdit(default_func)
        self.func_input.setToolTip("Use 'x' for domain, 't' for time. e.g., np.sin(x - t)")
        self.label_input = QLineEdit(default_label)
        
        self.anim_combo = QComboBox()
        self.anim_combo.addItems([
            "Draw Left-to-Right", "Draw Right-to-Left", "Fade In", "Slide Up from X-axis",
            "Curtain Drop (From Top)", "Scale from Center", "Horizontal Stretch",
            "Vertical Stretch (Scale Y)", "Snake (Trailing Segment)", "Bounce Drop (Elastic)",
            "Axis Projection (Shadows)", "Rotate Around Origin (Spin)", "Scribble (Hand-drawn)",
            "Scatter Plot (Dots)", "Reverse Erase (Un-draw)", "Typewriter (Chunk Reveal)",
            "Mirror Flip (Over X-axis)", "Color Cycle (Rainbow Draw)", "Pulse (Heartbeat)",
            "Step Function (Staircase)", "Area Under Curve (Integral)", "Shade Between Curves (f & g)",
            "Tangent Line (Derivative)", "Morphing (Next Layer)", "Parametric (Next Layer)", 
            "Slope Field (Diff Eq)", "Vector Field (Arrows)", "Circle Mask Reveal", "Flash/Blink",
            "Unit Circle (Radius Vector)", "Sine Unwrap (Circle to Wave)", "Cosine Unwrap (Circle to Wave)",
            "None (Static)"
        ])
        
        self.color_btn = ColorButton(default_color)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 10.0)
        self.width_spin.setValue(3.0)
        
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Solid", "Dashed", "Dotted", "DashDot"])
        
        self.dot_check = QCheckBox("Dot")
        self.dot_check.setChecked(True)
        self.dot_color_btn = ColorButton("red")
        
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 1000)
        self.delay_spin.setValue(0)
        self.delay_spin.setToolTip("Frames to wait before this layer starts animating.")
        
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-10000, 10000)
        self.x_min_spin.setValue(-10)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-10000, 10000)
        self.x_max_spin.setValue(10)
        
        form_layout.addRow("Function:", self.func_input)
        form_layout.addRow("Legend Label:", self.label_input)
        form_layout.addRow("Animation:", self.anim_combo)
        
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("Delay:"))
        adv_row.addWidget(self.delay_spin)
        adv_row.addWidget(QLabel("X Min:"))
        adv_row.addWidget(self.x_min_spin)
        adv_row.addWidget(QLabel("X Max:"))
        adv_row.addWidget(self.x_max_spin)
        self.sync_btn = QPushButton("Sync")
        self.sync_btn.setToolTip("Sync local domain to global bounds.")
        self.sync_btn.clicked.connect(self.sync_domain)
        adv_row.addWidget(self.sync_btn)
        form_layout.addRow("Timing/Domain:", adv_row)
        
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Color:"))
        style_row.addWidget(self.color_btn)
        style_row.addWidget(QLabel("Width:"))
        style_row.addWidget(self.width_spin)
        style_row.addWidget(QLabel("Style:"))
        style_row.addWidget(self.style_combo)
        style_row.addWidget(self.dot_check)
        style_row.addWidget(QLabel("Dot:"))
        style_row.addWidget(self.dot_color_btn)
        form_layout.addRow("Style:", style_row)
        
        layout.addLayout(form_layout)
        
        btn_row = QHBoxLayout()
        self.up_btn = QPushButton("Up")
        self.up_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.up_btn.clicked.connect(lambda: move_up_callback(self))
        
        self.down_btn = QPushButton("Down")
        self.down_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        self.down_btn.clicked.connect(lambda: move_down_callback(self))
        
        self.dup_btn = QPushButton("Dup")
        self.dup_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.dup_btn.clicked.connect(lambda: duplicate_callback(self))
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.remove_btn.clicked.connect(lambda: remove_callback(self))
        
        btn_row.addWidget(self.up_btn)
        btn_row.addWidget(self.down_btn)
        btn_row.addWidget(self.dup_btn)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.content_widget)
        
        self.toggled.connect(self.toggle_visibility)

    def toggle_visibility(self, checked):
        self.content_widget.setVisible(checked)

    def sync_domain(self):
        xmin, xmax = self.get_global_bounds()
        self.x_min_spin.setValue(xmin)
        self.x_max_spin.setValue(xmax)

    def update_index(self, index):
        self._index = index
        self.setTitle(f"Function {index}")

    def to_dict(self):
        return {
            "func": self.func_input.text(),
            "label": self.label_input.text(),
            "anim": self.anim_combo.currentText(),
            "color": self.color_btn.get_color(),
            "width": self.width_spin.value(),
            "style": self.style_combo.currentText(),
            "dot": self.dot_check.isChecked(),
            "dot_color": self.dot_color_btn.get_color(),
            "delay": self.delay_spin.value(),
            "x_min": self.x_min_spin.value(),
            "x_max": self.x_max_spin.value()
        }

    def from_dict(self, data):
        self.func_input.setText(data.get("func", "x"))
        self.label_input.setText(data.get("label", ""))
        self.anim_combo.setCurrentText(data.get("anim", "Draw Left-to-Right"))
        self.color_btn.set_color(data.get("color", "#1f77b4"))
        self.width_spin.setValue(data.get("width", 3.0))
        self.style_combo.setCurrentText(data.get("style", "Solid"))
        self.dot_check.setChecked(data.get("dot", True))
        self.dot_color_btn.set_color(data.get("dot_color", "red"))
        self.delay_spin.setValue(data.get("delay", 0))
        self.x_min_spin.setValue(data.get("x_min", -10))
        self.x_max_spin.setValue(data.get("x_max", 10))


class MathStudioPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math Video Studio - Pro Creator Edition")
        self.resize(1400, 950) # Slightly wider for new UI elements

        self.apply_dark_theme()
        self.setup_menu_and_shortcuts()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        scroll.setWidget(controls_container)
        scroll.setFixedWidth(550)

        proj_bar = QHBoxLayout()
        self.save_proj_btn = QPushButton("Save Project")
        self.save_proj_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_proj_btn.clicked.connect(self.save_project)
        self.load_proj_btn = QPushButton("Load Project")
        self.load_proj_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.load_proj_btn.clicked.connect(self.load_project)
        proj_bar.addWidget(self.save_proj_btn)
        proj_bar.addWidget(self.load_proj_btn)
        controls_layout.addLayout(proj_bar)

        self.tabs = QTabWidget()
        controls_layout.addWidget(self.tabs)

        # TAB 1: Functions
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        func_group = QGroupBox("Function Layers")
        func_group_layout = QVBoxLayout(func_group)
        
        self.layers_scroll = QScrollArea()
        self.layers_scroll.setWidgetResizable(True)
        self.layers_scroll.setMinimumHeight(400)
        self.layers_container = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setAlignment(Qt.AlignTop)
        self.layers_scroll.setWidget(self.layers_container)
        
        func_group_layout.addWidget(self.layers_scroll)
        
        layer_btns_row = QHBoxLayout()
        self.add_layer_btn = QPushButton("+ Add Layer")
        self.add_layer_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self.add_layer_btn.clicked.connect(lambda: self.add_layer("x", "", ""))
        
        self.clear_layers_btn = QPushButton("Clear All")
        self.clear_layers_btn.setStyleSheet("color: #ff5555;")
        self.clear_layers_btn.clicked.connect(self.clear_layers)
        
        layer_btns_row.addWidget(self.add_layer_btn)
        layer_btns_row.addWidget(self.clear_layers_btn)
        func_group_layout.addLayout(layer_btns_row)
        
        tab1_layout.addWidget(func_group)

        scene_group = QGroupBox("Global Scene Effects")
        scene_layout = QFormLayout()
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(["None", "Zoom In", "Zoom Out", "Camera Pan (Follow L1)", "Sweeping Bar"])
        self.intersect_check = QCheckBox("Show Flashing Intersections (L1 & L2)")
        
        scene_layout.addRow("Scene Effect:", self.scene_combo)
        scene_layout.addRow(self.intersect_check)
        scene_group.setLayout(scene_layout)
        tab1_layout.addWidget(scene_group)
        
        self.tabs.addTab(tab1, "Functions")

        # TAB 2: Style
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        
        theme_group = QGroupBox("Canvas & Axes")
        theme_layout = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Paper White", "Dark Mode", "Midnight Blue"])
        self.theme_combo.currentIndexChanged.connect(self.setup_plot)
        self.axes_combo = QComboBox()
        self.axes_combo.addItems(["Border Box", "Center Cross (0,0)"])
        self.axes_combo.currentIndexChanged.connect(self.setup_plot)
        self.title_input = QLineEdit(r"$y = \sin(x - t)$") # Default updated to show 't' feature
        
        self.latex_check = QCheckBox("Enable LaTeX Rendering (use $...$)")
        self.latex_check.setChecked(True)
        self.latex_check.stateChanged.connect(self.setup_plot)
        
        self.x_label_input = QLineEdit("")
        self.y_label_input = QLineEdit("")
        self.x_tick_spin = QDoubleSpinBox(); self.x_tick_spin.setRange(0, 1000); self.x_tick_spin.setValue(0)
        self.y_tick_spin = QDoubleSpinBox(); self.y_tick_spin.setRange(0, 1000); self.y_tick_spin.setValue(0)
        
        self.grid_check = QCheckBox("Show Grid Lines")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self.setup_plot)
        self.ticks_check = QCheckBox("Show Axis Numbers (Ticks)")
        self.ticks_check.setChecked(True)
        self.ticks_check.stateChanged.connect(self.setup_plot)
        self.aspect_check = QCheckBox("Equal Aspect Ratio (1:1 Scale)")
        self.aspect_check.setChecked(False)
        self.aspect_check.stateChanged.connect(self.setup_plot)
        
        theme_layout.addRow("Theme:", self.theme_combo)
        theme_layout.addRow("Axes Style:", self.axes_combo)
        theme_layout.addRow("Graph Title:", self.title_input)
        theme_layout.addRow(self.latex_check)
        theme_layout.addRow("X Axis Label:", self.x_label_input)
        theme_layout.addRow("Y Axis Label:", self.y_label_input)
        theme_layout.addRow("X Tick Step:", self.x_tick_spin)
        theme_layout.addRow("Y Tick Step:", self.y_tick_spin)
        theme_layout.addRow(self.grid_check)
        theme_layout.addRow(self.ticks_check)
        theme_layout.addRow(self.aspect_check)
        theme_group.setLayout(theme_layout)
        tab2_layout.addWidget(theme_group)

        font_group = QGroupBox("Font Sizes")
        font_layout = QFormLayout()
        self.title_font_spin = QSpinBox(); self.title_font_spin.setRange(8, 48); self.title_font_spin.setValue(16)
        self.axis_font_spin = QSpinBox(); self.axis_font_spin.setRange(8, 36); self.axis_font_spin.setValue(12)
        self.text_font_spin = QSpinBox(); self.text_font_spin.setRange(8, 36); self.text_font_spin.setValue(10)
        font_layout.addRow("Title:", self.title_font_spin)
        font_layout.addRow("Axes/Ticks:", self.axis_font_spin)
        font_layout.addRow("Dynamic (x,y) Text:", self.text_font_spin)
        font_group.setLayout(font_layout)
        tab2_layout.addWidget(font_group)

        annot_group = QGroupBox("Static Text Annotation (Use {t} for time)")
        annot_layout = QVBoxLayout()
        self.annot_text_input = QLineEdit("Time: {t:.2f}s")
        self.annot_text_input.setPlaceholderText("e.g., Area = 1, or Time = {t:.2f}")
        annot_form = QFormLayout()
        self.annot_x_spin = QDoubleSpinBox(); self.annot_x_spin.setRange(-10000, 10000); self.annot_x_spin.setValue(0.5)
        self.annot_y_spin = QDoubleSpinBox(); self.annot_y_spin.setRange(-10000, 10000); self.annot_y_spin.setValue(1.5)
        self.annot_size_spin = QSpinBox(); self.annot_size_spin.setRange(8, 48); self.annot_size_spin.setValue(14)
        annot_form.addRow("Text:", self.annot_text_input)
        annot_form.addRow("X Pos:", self.annot_x_spin)
        annot_form.addRow("Y Pos:", self.annot_y_spin)
        annot_form.addRow("Font Size:", self.annot_size_spin)
        annot_layout.addLayout(annot_form)
        annot_group.setLayout(annot_layout)
        tab2_layout.addWidget(annot_group)
        
        tab2_layout.addStretch()
        self.tabs.addTab(tab2, "Style")

        # TAB 3: Export
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        
        vid_group = QGroupBox("Video Dimensions & Time")
        vid_layout = QFormLayout()
        self.x_min_spin = QDoubleSpinBox(); self.x_min_spin.setRange(-10000, 10000); self.x_min_spin.setValue(0)
        self.x_max_spin = QDoubleSpinBox(); self.x_max_spin.setRange(-10000, 10000); self.x_max_spin.setValue(6.283)
        self.y_min_spin = QDoubleSpinBox(); self.y_min_spin.setRange(-10000, 10000); self.y_min_spin.setValue(-2)
        self.y_max_spin = QDoubleSpinBox(); self.y_max_spin.setRange(-10000, 10000); self.y_max_spin.setValue(2)
        
        self.frames_spin = QSpinBox(); self.frames_spin.setRange(10, 10000); self.frames_spin.setValue(150)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(10, 120); self.fps_spin.setValue(30)
        
        self.intro_hold_spin = QDoubleSpinBox(); self.intro_hold_spin.setRange(0, 60); self.intro_hold_spin.setValue(1.0); self.intro_hold_spin.setSingleStep(0.5)
        self.outro_hold_spin = QDoubleSpinBox(); self.outro_hold_spin.setRange(0, 60); self.outro_hold_spin.setValue(2.0); self.outro_hold_spin.setSingleStep(0.5)
        
        self.boomerang_check = QCheckBox("Loop Animation (Boomerang Effect)")
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YouTube (16:9)", "Shorts/TikTok (9:16)"])
        self.format_combo.currentIndexChanged.connect(self.setup_plot)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["MP4 Video", "GIF Image"])
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["480p", "720p", "1080p (HD)", "1440p (2K)", "2160p (4K)"])
        
        self.easing_combo = QComboBox()
        self.easing_combo.addItems(["Linear", "Ease In", "Ease Out", "Ease In-Out"])
        
        vid_layout.addRow("Global X Min:", self.x_min_spin)
        vid_layout.addRow("Global X Max:", self.x_max_spin)
        vid_layout.addRow("Global Y Min:", self.y_min_spin)
        vid_layout.addRow("Global Y Max:", self.y_max_spin)
        vid_layout.addRow("Animation Frames:", self.frames_spin)
        vid_layout.addRow("FPS:", self.fps_spin)
        vid_layout.addRow("Intro Hold (sec):", self.intro_hold_spin)
        vid_layout.addRow("Outro Hold (sec):", self.outro_hold_spin)
        vid_layout.addRow(self.boomerang_check)
        vid_layout.addRow("Easing Function:", self.easing_combo)
        vid_layout.addRow("Aspect Format:", self.format_combo)
        vid_layout.addRow("Export Format:", self.export_format_combo)
        vid_layout.addRow("Export Quality:", self.quality_combo)
        vid_group.setLayout(vid_layout)
        tab3_layout.addWidget(vid_group)

        audio_group = QGroupBox("Background Audio / Voiceover (MP4 Only)")
        audio_layout = QHBoxLayout()
        self.audio_btn = QPushButton("Select Audio File")
        self.audio_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.audio_btn.clicked.connect(self.select_audio)
        self.audio_label = QLabel("No audio selected")
        self.audio_label.setStyleSheet("font-style: italic; color: #888;")
        audio_layout.addWidget(self.audio_btn)
        audio_layout.addWidget(self.audio_label)
        audio_group.setLayout(audio_layout)
        tab3_layout.addWidget(audio_group)
        
        tab3_layout.addStretch()
        self.tabs.addTab(tab3, "Export")

        # Bottom Controls
        controls_bottom = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.preview_btn.clicked.connect(self.preview_animation)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop_preview)
        self.stop_btn.setEnabled(False)
        
        self.frame_btn = QPushButton("Frame PNG")
        self.frame_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        self.frame_btn.setToolTip("Export the current preview frame as a PNG image.")
        self.frame_btn.clicked.connect(self.export_frame)
        
        self.render_btn = QPushButton("Render Video")
        self.render_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        self.render_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.render_btn.clicked.connect(self.render_video)
        
        controls_bottom.addWidget(self.preview_btn)
        controls_bottom.addWidget(self.stop_btn)
        controls_bottom.addWidget(self.frame_btn)
        controls_bottom.addWidget(self.render_btn)
        controls_layout.addLayout(controls_bottom)
        
        controls_layout.addStretch()

        # Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.setup_plot()

        main_layout.addWidget(scroll)
        main_layout.addWidget(self.canvas, 1)
        self.ani = None
        self.audio_path = ""
        self.circle_drawn = False
        
        self.layer_widgets = []
        self.add_layer("np.sin(x - t)", "#1f77b4", r"$f(x) = \sin(x - t)$")
        self.add_layer("np.cos(x)", "#ff7f0e", r"$g(x) = \cos(x)$")

        self.statusBar().showMessage("Ready. Use 't' in your functions for time-based animations!")
        self.check_ffmpeg()

    def setup_menu_and_shortcuts(self):
        save_action = QAction("Save Project", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_project)
        self.addAction(save_action)

        load_action = QAction("Load Project", self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self.load_project)
        self.addAction(load_action)

        render_action = QAction("Render", self)
        render_action.setShortcut(QKeySequence("Ctrl+R"))
        render_action.triggered.connect(self.render_video)
        self.addAction(render_action)

        preview_action = QAction("Preview", self)
        preview_action.setShortcut(QKeySequence(Qt.Key_Space))
        preview_action.triggered.connect(self.preview_animation)
        self.addAction(preview_action)

    def check_ffmpeg(self):
        if shutil.which("ffmpeg") is None:
            self.statusBar().showMessage("Warning: FFmpeg not found. MP4 rendering will fail. GIF export will still work.", 10000)

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.black)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)
        self.setStyleSheet("QToolTip { color: #ffffff; background-color: #2b2b2b; border: 1px solid white; }"
                           "QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 10px; }"
                           "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")

    def get_global_bounds(self):
        return self.x_min_spin.value(), self.x_max_spin.value()

    def select_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.aac)")
        if file_path:
            self.audio_path = file_path
            self.audio_label.setText(os.path.basename(file_path))
            self.audio_label.setStyleSheet("font-style: normal; color: #fff;")

    def add_layer(self, default_func="", default_color="", default_label=""):
        if not default_func: default_func = "x"
        if not default_color:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            default_color = colors[len(self.layer_widgets) % len(colors)]
            
        index = len(self.layer_widgets) + 1
        layer = FunctionLayerWidget(index, default_func, default_color, default_label, self.remove_layer, self.duplicate_layer, self.get_global_bounds, self.move_layer_up, self.move_layer_down)
        layer.x_min_spin.setValue(self.x_min_spin.value())
        layer.x_max_spin.setValue(self.x_max_spin.value())
        self.layers_layout.addWidget(layer)
        self.layer_widgets.append(layer)

    def duplicate_layer(self, layer_widget):
        data = layer_widget.to_dict()
        self.add_layer(data["func"], data["color"], data["label"])
        self.layer_widgets[-1].from_dict(data)

    def remove_layer(self, layer_widget):
        if len(self.layer_widgets) <= 1:
            QMessageBox.warning(self, "Warning", "You must have at least one function layer.")
            return
        self.layer_widgets.remove(layer_widget)
        layer_widget.deleteLater()
        for i, w in enumerate(self.layer_widgets):
            w.update_index(i + 1)

    def clear_layers(self):
        reply = QMessageBox.question(self, "Clear All Layers", "Are you sure?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            for w in self.layer_widgets[:]:
                self.layer_widgets.remove(w)
                w.deleteLater()
            self.add_layer("x", "#1f77b4", "")

    def rebuild_layers_ui(self):
        while self.layers_layout.count():
            item = self.layers_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for i, w in enumerate(self.layer_widgets):
            self.layers_layout.addWidget(w)
            w.update_index(i + 1)

    def move_layer_up(self, layer_widget):
        idx = self.layer_widgets.index(layer_widget)
        if idx > 0:
            self.layer_widgets.insert(idx - 1, self.layer_widgets.pop(idx))
            self.rebuild_layers_ui()

    def move_layer_down(self, layer_widget):
        idx = self.layer_widgets.index(layer_widget)
        if idx < len(self.layer_widgets) - 1:
            self.layer_widgets.insert(idx + 1, self.layer_widgets.pop(idx))
            self.rebuild_layers_ui()

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Math Project (*.mathproj)")
        if not file_path: return
        if not file_path.endswith('.mathproj'): file_path += '.mathproj'
        
        data = {
            "settings": {
                "theme": self.theme_combo.currentText(), "axes": self.axes_combo.currentText(),
                "title": self.title_input.text(), "latex": self.latex_check.isChecked(),
                "x_label": self.x_label_input.text(), "y_label": self.y_label_input.text(),
                "x_tick": self.x_tick_spin.value(), "y_tick": self.y_tick_spin.value(),
                "grid": self.grid_check.isChecked(), "ticks": self.ticks_check.isChecked(), "aspect": self.aspect_check.isChecked(),
                "title_font": self.title_font_spin.value(), "axis_font": self.axis_font_spin.value(), "text_font": self.text_font_spin.value(),
                "annot_text": self.annot_text_input.text(), "annot_x": self.annot_x_spin.value(), "annot_y": self.annot_y_spin.value(), "annot_size": self.annot_size_spin.value(),
                "scene": self.scene_combo.currentText(), "intersect": self.intersect_check.isChecked(),
                "x_min": self.x_min_spin.value(), "x_max": self.x_max_spin.value(), "y_min": self.y_min_spin.value(), "y_max": self.y_max_spin.value(),
                "frames": self.frames_spin.value(), "fps": self.fps_spin.value(),
                "intro": self.intro_hold_spin.value(), "outro": self.outro_hold_spin.value(), "boomerang": self.boomerang_check.isChecked(),
                "easing": self.easing_combo.currentText(), "format": self.format_combo.currentText(), "quality": self.quality_combo.currentText(),
                "export_format": self.export_format_combo.currentText(), "audio_path": self.audio_path
            },
            "layers": [w.to_dict() for w in self.layer_widgets]
        }
        with open(file_path, 'w') as f: json.dump(data, f, indent=4)
        self.statusBar().showMessage(f"Project saved to {file_path}", 5000)

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Math Project (*.mathproj)")
        if not file_path: return
        with open(file_path, 'r') as f:
            try: data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}"); return
                
        s = data.get("settings", {})
        self.theme_combo.setCurrentText(s.get("theme", "Paper White"))
        self.axes_combo.setCurrentText(s.get("axes", "Border Box"))
        self.title_input.setText(s.get("title", ""))
        self.latex_check.setChecked(s.get("latex", True))
        self.x_label_input.setText(s.get("x_label", ""))
        self.y_label_input.setText(s.get("y_label", ""))
        self.x_tick_spin.setValue(s.get("x_tick", 0))
        self.y_tick_spin.setValue(s.get("y_tick", 0))
        self.grid_check.setChecked(s.get("grid", True))
        self.ticks_check.setChecked(s.get("ticks", True))
        self.aspect_check.setChecked(s.get("aspect", False))
        self.title_font_spin.setValue(s.get("title_font", 16))
        self.axis_font_spin.setValue(s.get("axis_font", 12))
        self.text_font_spin.setValue(s.get("text_font", 10))
        self.annot_text_input.setText(s.get("annot_text", ""))
        self.annot_x_spin.setValue(s.get("annot_x", 0))
        self.annot_y_spin.setValue(s.get("annot_y", 0))
        self.annot_size_spin.setValue(s.get("annot_size", 14))
        self.scene_combo.setCurrentText(s.get("scene", "None"))
        self.intersect_check.setChecked(s.get("intersect", False))
        self.x_min_spin.setValue(s.get("x_min", 0))
        self.x_max_spin.setValue(s.get("x_max", 6.28))
        self.y_min_spin.setValue(s.get("y_min", -2))
        self.y_max_spin.setValue(s.get("y_max", 2))
        self.frames_spin.setValue(s.get("frames", 150))
        self.fps_spin.setValue(s.get("fps", 30))
        self.intro_hold_spin.setValue(s.get("intro", 1.0))
        self.outro_hold_spin.setValue(s.get("outro", 2.0))
        self.boomerang_check.setChecked(s.get("boomerang", False))
        self.easing_combo.setCurrentText(s.get("easing", "Linear"))
        self.format_combo.setCurrentText(s.get("format", "YouTube (16:9)"))
        self.quality_combo.setCurrentText(s.get("quality", "1080p (HD)"))
        self.export_format_combo.setCurrentText(s.get("export_format", "MP4 Video"))
        
        self.audio_path = s.get("audio_path", "")
        if self.audio_path and os.path.exists(self.audio_path):
            self.audio_label.setText(os.path.basename(self.audio_path))
            self.audio_label.setStyleSheet("font-style: normal; color: #fff;")
        else:
            self.audio_path = ""
            self.audio_label.setText("No audio selected")
            self.audio_label.setStyleSheet("font-style: italic; color: #888;")
        
        for w in self.layer_widgets[:]: self.remove_layer(w)
        for layer_data in data.get("layers", []):
            self.add_layer()
            self.layer_widgets[-1].from_dict(layer_data)
            
        self.setup_plot()
        self.statusBar().showMessage(f"Project loaded from {file_path}", 5000)

    def format_text(self, text, t_val=0.0):
        if not text: return ""
        # Dynamic substitution
        text = text.replace("{t}", f"{t_val:.2f}")
        text = text.replace("{frame}", f"{int(t_val * self.fps_spin.value())}")
        
        if not self.latex_check.isChecked():
            return text.replace("$", r"\$")
        return text

    def setup_plot(self, t_val=0.0):
        self.figure.clear()
        themes = {
            "Paper White": {"bg": "white", "grid": "lightgray", "axes": "black", "text": "black"},
            "Dark Mode": {"bg": "#121212", "grid": "#333333", "axes": "#FFFFFF", "text": "#FFFFFF"},
            "Midnight Blue": {"bg": "#0a192f", "grid": "#172a45", "axes": "#64ffda", "text": "#e6f1ff"}
        }
        t = themes[self.theme_combo.currentText()]
        self.figure.patch.set_facecolor(t["bg"])
        
        if self.format_combo.currentText().startswith("Shorts"): self.figure.set_size_inches(4.5, 8)
        else: self.figure.set_size_inches(8, 4.5)
            
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(t["bg"])
        
        if self.grid_check.isChecked(): self.ax.grid(True, linestyle='--', alpha=0.4, color=t["grid"])
        else: self.ax.grid(False)
            
        if not self.ticks_check.isChecked():
            self.ax.set_xticks([]); self.ax.set_yticks([])
        elif self.x_tick_spin.value() > 0 or self.y_tick_spin.value() > 0:
            if self.x_tick_spin.value() > 0: self.ax.set_xticks(np.arange(self.x_min_spin.value(), self.x_max_spin.value() + self.x_tick_spin.value(), self.x_tick_spin.value()))
            if self.y_tick_spin.value() > 0: self.ax.set_yticks(np.arange(self.y_min_spin.value(), self.y_max_spin.value() + self.y_tick_spin.value(), self.y_tick_spin.value()))
                
        self.ax.tick_params(colors=t["text"], labelsize=self.axis_font_spin.value())
        for spine in self.ax.spines.values(): spine.set_color(t["axes"])
            
        if self.axes_combo.currentText().startswith("Center"):
            self.ax.spines['left'].set_position('zero'); self.ax.spines['bottom'].set_position('zero')
            self.ax.spines['right'].set_color('none'); self.ax.spines['top'].set_color('none')
        else:
            self.ax.spines['left'].set_position(('outward', 0)); self.ax.spines['bottom'].set_position(('outward', 0))
            self.ax.spines['right'].set_visible(True); self.ax.spines['top'].set_visible(True)

        self.ax.set_xlim(self.x_min_spin.value(), self.x_max_spin.value())
        self.ax.set_ylim(self.y_min_spin.value(), self.y_max_spin.value())
        if self.aspect_check.isChecked(): self.ax.set_aspect('equal', adjustable='box')
            
        self.ax.set_title(self.format_text(self.title_input.text(), t_val), color=t["text"], fontsize=self.title_font_spin.value(), pad=20)
        if self.x_label_input.text().strip(): self.ax.set_xlabel(self.format_text(self.x_label_input.text(), t_val), color=t["text"], fontsize=self.axis_font_spin.value())
        if self.y_label_input.text().strip(): self.ax.set_ylabel(self.format_text(self.y_label_input.text(), t_val), color=t["text"], fontsize=self.axis_font_spin.value())
        
        self.dynamic_annot = None
        if self.annot_text_input.text().strip():
            self.dynamic_annot = self.ax.text(self.annot_x_spin.value(), self.annot_y_spin.value(), self.format_text(self.annot_text_input.text(), t_val), color=t["text"], fontsize=self.annot_size_spin.value(), weight='bold')
            
        self.figure.tight_layout()
        self.canvas.draw()

    def eval_func(self, func_str, x, y=None, t=0):
        allowed = {
            "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "pi": np.pi, "e": np.e, 
            "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "abs": np.abs, "t": t,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "arctan2": np.arctan2,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "floor": np.floor, "ceil": np.ceil, "sign": np.sign
        }
        local_vars = {"x": x, **allowed}
        if y is not None: local_vars["y"] = y
        return eval(func_str, {"__builtins__": {}}, local_vars)

    def ease_out_bounce(self, t):
        if t < 1/2.75: return 7.5625*t*t
        elif t < 2/2.75:
            t -= 1.5/2.75; return 7.5625*t*t + 0.75
        elif t < 2.5/2.75:
            t -= 2.25/2.75; return 7.5625*t*t + 0.9375
        else:
            t -= 2.625/2.75; return 7.5625*t*t + 0.984375

    def apply_easing(self, progress, easing_type):
        if progress <= 0: return 0
        if progress >= 1: return 1
        if easing_type == "Linear": return progress
        elif easing_type == "Ease In": return progress ** 3
        elif easing_type == "Ease Out": return 1 - (1 - progress) ** 3
        elif easing_type == "Ease In-Out":
            if progress < 0.5: return 4 * progress ** 3
            else: return 1 - (-2 * progress + 2) ** 3 / 2
        return progress

    def get_style_string(self, style_name):
        return {"Solid": "-", "Dashed": "--", "Dotted": ":", "DashDot": "-."}.get(style_name, "-")

    def init_animation_objects(self):
        layers_data = []
        t_color = "white" if self.theme_combo.currentText() != "Paper White" else "black"
        x_min, x_max = self.x_min_spin.value(), self.x_max_spin.value()
        y_min, y_max = self.y_min_spin.value(), self.y_max_spin.value()
        
        needs_circle = any(layer.anim_combo.currentText() in ["Sine Unwrap (Circle to Wave)", "Cosine Unwrap (Circle to Wave)"] for layer in self.layer_widgets)
        if needs_circle:
            theta = np.linspace(0, 2*np.pi, 100)
            self.ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.5, lw=1)
        
        for i, layer in enumerate(self.layer_widgets):
            x_local = np.linspace(layer.x_min_spin.value(), layer.x_max_spin.value(), 1000)
            line, = self.ax.plot([], [], lw=layer.width_spin.value(), color=layer.color_btn.get_color(), label=self.format_text(layer.label_input.text()) if layer.label_input.text().strip() else f"_child{i}")
            point, = self.ax.plot([], [], 'o', markersize=8.0, color=layer.dot_color_btn.get_color())
            text_item = self.ax.text(0, 0, "", color=t_color, fontsize=self.text_font_spin.value(), weight='bold')
            polygon = Polygon(np.zeros((3,2)), alpha=0.3, color=layer.color_btn.get_color())
            self.ax.add_patch(polygon)
            tangent_line, = self.ax.plot([], [], '--', color=layer.dot_color_btn.get_color(), lw=2)
            proj_x, = self.ax.plot([], [], '--', color='gray', lw=1, alpha=0.5)
            proj_y, = self.ax.plot([], [], '--', color='gray', lw=1, alpha=0.5)
            
            segments = []
            anim_type = layer.anim_combo.currentText()
            if anim_type in ["Slope Field (Diff Eq)", "Vector Field (Arrows)"]:
                for gx in np.linspace(x_min, x_max, 15):
                    for gy in np.linspace(y_min, y_max, 15):
                        try: slope = self.eval_func(layer.func_input.text(), gx, gy)
                        except: slope = 0
                        length = (x_max - x_min) * 0.03
                        dx = length / np.sqrt(1 + slope**2)
                        dy = slope * dx
                        if anim_type == "Vector Field (Arrows)":
                            seg, = self.ax.plot([gx, gx+dx], [gy, gy+dy], color=layer.color_btn.get_color(), lw=1.5, marker='>', markersize=4)
                        else:
                            seg, = self.ax.plot([gx-dx, gx+dx], [gy-dy, gy+dy], color=layer.color_btn.get_color(), lw=1.5)
                        segments.append(seg)
            
            layers_data.append({
                "widget": layer, "x_local": x_local, "line": line, "point": point, "text": text_item,
                "polygon": polygon, "tangent": tangent_line, "proj_x": proj_x, "proj_y": proj_y, "segments": segments
            })
            
        if any(layer.label_input.text().strip() for layer in self.layer_widgets):
            leg = self.ax.legend(loc='upper right', facecolor=t_color, edgecolor=t_color, fontsize=self.axis_font_spin.value())
            for text in leg.get_texts(): text.set_color("black" if t_color == "white" else "white")
            
        return layers_data

    def animation_logic(self, frame, total_frames, anim_frames, intro_frames, layers_data, scene_effect, intersect_check, easing_type, boomerang):
        global_progress = 0.0
        if frame < intro_frames: global_progress = 0.0
        elif frame >= intro_frames + anim_frames: global_progress = 1.0
        else: global_progress = (frame - intro_frames) / anim_frames
            
        if boomerang and 1.0 > global_progress > 0.0:
            if global_progress > 0.5: global_progress = 1.0 - ((global_progress - 0.5) * 2)
            else: global_progress = global_progress * 2
            
        x_min, x_max = self.x_min_spin.value(), self.x_max_spin.value()
        y_min, y_max = self.y_min_spin.value(), self.y_max_spin.value()
        self.ax.set_xlim(x_min, x_max); self.ax.set_ylim(y_min, y_max)
        
        # Calculate current time 't' for dynamic functions
        fps = self.fps_spin.value()
        current_time = max(0, (frame - intro_frames)) / fps
        
        # Update dynamic title and annotation
        self.ax.set_title(self.format_text(self.title_input.text(), current_time))
        if self.dynamic_annot:
            self.dynamic_annot.set_text(self.format_text(self.annot_text_input.text(), current_time))
        
        y_vals_all = []
        for data in layers_data:
            try: y_vals_all.append(self.eval_func(data["widget"].func_input.text(), data["x_local"], t=current_time))
            except: y_vals_all.append(np.zeros_like(data["x_local"]))
        
        cam_x, cam_y = None, None
        
        for i, data in enumerate(layers_data):
            layer = data["widget"]
            anim_type = layer.anim_combo.currentText()
            line = data["line"]; point = data["point"]; text_item = data["text"]
            polygon = data["polygon"]; tangent_line = data["tangent"]
            proj_x = data["proj_x"]; proj_y = data["proj_y"]; segments = data["segments"]
            
            x = data["x_local"]; y_eval = y_vals_all[i]
            
            delay = layer.delay_spin.value()
            local_frame = max(0, frame - intro_frames - delay)
            local_anim_frames = max(1, anim_frames - delay)
            local_progress = min(1.0, local_frame / local_anim_frames)
            
            if boomerang and 1.0 > local_progress > 0.0:
                if local_progress > 0.5: local_progress = 1.0 - ((local_progress - 0.5) * 2)
                else: local_progress = local_progress * 2
                    
            eased_progress = self.apply_easing(local_progress, easing_type)
            
            line.set_visible(anim_type not in ["Slope Field (Diff Eq)", "Vector Field (Arrows)"])
            point.set_visible(layer.dot_check.isChecked() and local_progress > 0 and anim_type not in ["Slope Field (Diff Eq)", "Vector Field (Arrows)"])
            text_item.set_visible(False); polygon.set_visible(False); tangent_line.set_visible(False)
            proj_x.set_visible(False); proj_y.set_visible(False)
            line.set_linestyle(self.get_style_string(layer.style_combo.currentText()))
            line.set_marker('')
            line.set_alpha(1.0 if local_progress > 0 else 0)
            line.set_linewidth(layer.width_spin.value()); line.set_color(layer.color_btn.get_color())
            
            limit = max(1, int(1000 * eased_progress))
            
            if anim_type in ["Slope Field (Diff Eq)", "Vector Field (Arrows)"]:
                show_segs = int(len(segments) * eased_progress)
                for idx, seg in enumerate(segments): seg.set_visible(idx < show_segs)
            elif anim_type == "Draw Left-to-Right":
                line.set_data(x[:limit], y_eval[:limit])
                if limit > 0 and point.get_visible(): point.set_data([x[limit-1]], [y_eval[limit-1]])
            elif anim_type == "Draw Right-to-Left":
                start_idx = min(999, int(1000 * (1 - eased_progress)))
                line.set_data(x[start_idx:], y_eval[start_idx:])
                if point.get_visible(): point.set_data([x[start_idx]], [y_eval[start_idx]])
            elif anim_type == "Fade In":
                line.set_data(x, y_eval); line.set_alpha(eased_progress)
            elif anim_type == "Slide Up from X-axis":
                line.set_data(x, y_eval * eased_progress)
            elif anim_type == "Curtain Drop (From Top)":
                current_y = y_max - (y_max - y_eval) * eased_progress
                line.set_data(x, current_y)
                if point.get_visible(): point.set_data([x[500]], [current_y[500]])
            elif anim_type == "Scale from Center":
                c = 500; s = int(500 * eased_progress); line.set_data(x[c-s:c+s], y_eval[c-s:c+s])
            elif anim_type == "Horizontal Stretch":
                center_x = (x_min + x_max) / 2
                line.set_data(center_x + (x - center_x) * eased_progress, y_eval)
            elif anim_type == "Vertical Stretch (Scale Y)":
                line.set_data(x, y_eval * eased_progress)
            elif anim_type == "Snake (Trailing Segment)":
                seg_len = max(10, int(1000 * 0.25)); head = int(1000 * eased_progress); tail = max(0, head - seg_len)
                line.set_data(x[tail:head], y_eval[tail:head])
                if head > 0 and point.get_visible(): point.set_data([x[head-1]], [y_eval[head-1]])
            elif anim_type == "Bounce Drop (Elastic)":
                scale = self.ease_out_bounce(eased_progress); line.set_data(x, y_eval * scale)
                if point.get_visible(): point.set_data([x[500]], [y_eval[500] * scale])
            elif anim_type == "Axis Projection (Shadows)":
                line.set_data(x[:limit], y_eval[:limit])
                x_curr, y_curr = x[limit-1], y_eval[limit-1]
                proj_x.set_data([x_curr, x_curr], [y_min, y_curr]); proj_x.set_visible(True)
                proj_y.set_data([x_min, x_curr], [y_curr, y_curr]); proj_y.set_visible(True)
                if point.get_visible(): point.set_data([x_curr], [y_curr])
                text_item.set_position((x_curr, y_curr)); text_item.set_text(f"({x_curr:.1f}, {y_curr:.1f})"); text_item.set_visible(True)
            elif anim_type == "Rotate Around Origin (Spin)":
                theta = eased_progress * 2 * np.pi
                line.set_data(x * np.cos(theta) - y_eval * np.sin(theta), x * np.sin(theta) + y_eval * np.cos(theta))
            elif anim_type == "Scribble (Hand-drawn)":
                noise = np.random.normal(0, 0.05 * (1 - eased_progress), limit)
                line.set_data(x[:limit], y_eval[:limit] + noise)
                if point.get_visible(): point.set_data([x[limit-1]], [y_eval[limit-1] + noise[-1]])
            elif anim_type == "Scatter Plot (Dots)":
                line.set_data(x[:limit], y_eval[:limit]); line.set_linestyle('none'); line.set_marker('o')
                line.set_markersize(layer.width_spin.value() * 2)
            elif anim_type == "Reverse Erase (Un-draw)":
                limit = max(1, int(1000 * (1 - eased_progress)))
                line.set_data(x[:limit], y_eval[:limit])
                if point.get_visible(): point.set_data([x[limit-1]], [y_eval[limit-1]])
            elif anim_type == "Typewriter (Chunk Reveal)":
                limit = max(1, (int(1000 * eased_progress) // 20) * 20)
                line.set_data(x[:limit], y_eval[:limit])
            elif anim_type == "Mirror Flip (Over X-axis)":
                line.set_data(x, y_eval * (-1 + 2 * eased_progress))
            elif anim_type == "Color Cycle (Rainbow Draw)":
                line.set_data(x[:limit], y_eval[:limit]); line.set_color(cm.rainbow(eased_progress))
            elif anim_type == "Pulse (Heartbeat)":
                line.set_data(x, y_eval); line.set_linewidth(layer.width_spin.value() + np.sin(frame * 0.3) * 2.5)
            elif anim_type == "Step Function (Staircase)":
                x_step = np.repeat(x[:limit], 2)[1:]; y_step = np.repeat(y_eval[:limit], 2)[:-1]
                line.set_data(x_step, y_step)
            elif anim_type == "Area Under Curve (Integral)":
                line.set_data(x[:limit], y_eval[:limit])
                verts = np.concatenate([np.column_stack([x[:limit], y_eval[:limit]]), [[x[limit-1], 0], [x[0], 0]]])
                polygon.set_xy(verts); polygon.set_visible(True)
                if point.get_visible(): point.set_data([x[limit-1]], [y_eval[limit-1]])
            elif anim_type == "Shade Between Curves (f & g)":
                y1 = y_eval
                y2 = y_vals_all[i+1] if i+1 < len(y_vals_all) else np.zeros_like(x)
                line.set_data(x[:limit], y1[:limit])
                safe_limit = max(1, limit)
                verts = np.concatenate([
                    np.column_stack([x[:safe_limit], y1[:safe_limit]]),
                    np.column_stack([x[:safe_limit][::-1], y2[:safe_limit][::-1]])
                ])
                polygon.set_xy(verts); polygon.set_visible(True)
                if point.get_visible(): point.set_data([x[safe_limit-1]], [y1[safe_limit-1]])
            elif anim_type == "Tangent Line (Derivative)":
                line.set_data(x[:limit], y_eval[:limit])
                x_curr, y_curr = x[limit-1], y_eval[limit-1]; dx = 0.01
                m = (self.eval_func(layer.func_input.text(), x_curr + dx, t=current_time) - self.eval_func(layer.func_input.text(), x_curr - dx, t=current_time)) / (2 * dx)
                tangent_line.set_data([x_min, x_max], m * (np.array([x_min, x_max]) - x_curr) + y_curr); tangent_line.set_visible(True)
                if point.get_visible(): point.set_data([x_curr], [y_curr])
                text_item.set_position((x_curr, y_curr)); text_item.set_text(f"slope={m:.2f}"); text_item.set_visible(True)
            elif anim_type == "Morphing (Next Layer)":
                y_next = y_vals_all[i+1] if i+1 < len(y_vals_all) else y_eval
                y_morph = (1 - eased_progress) * y_eval + eased_progress * y_next
                line.set_data(x, y_morph)
                if point.get_visible(): point.set_data([x[500]], [y_morph[500]])
            elif anim_type == "Parametric (Next Layer)":
                y_next = y_vals_all[i+1] if i+1 < len(y_vals_all) else np.zeros_like(x)
                line.set_data(y_eval[:limit], y_next[:limit])
                if point.get_visible(): point.set_data([y_eval[limit-1]], [y_next[limit-1]])
            elif anim_type == "Circle Mask Reveal":
                cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
                max_dist = np.sqrt((x_max-cx)**2 + (y_max-cy)**2)
                radius = max_dist * eased_progress
                dist = np.sqrt((x - cx)**2 + (y_eval - cy)**2)
                mask = dist <= radius
                line.set_data(x[mask], y_eval[mask])
            elif anim_type == "Flash/Blink":
                if np.sin(frame * 0.5) > 0: line.set_data(x, y_eval)
                else: line.set_data([], [])
            elif anim_type == "Unit Circle (Radius Vector)":
                r = y_eval
                x_cart = r * np.cos(x); y_cart = r * np.sin(x)
                line.set_data(x_cart[:limit], y_cart[:limit])
                x_curr, y_curr = x_cart[limit-1], y_cart[limit-1]
                tangent_line.set_data([0, x_curr], [0, y_curr]); tangent_line.set_visible(True)
                proj_x.set_data([x_curr, x_curr], [0, y_curr]); proj_x.set_visible(True)
                proj_y.set_data([0, x_curr], [y_curr, y_curr]); proj_y.set_visible(True)
                if point.get_visible(): point.set_data([x_curr], [y_curr])
                text_item.set_position((x_curr, y_curr)); text_item.set_text(f"({x_curr:.1f}, {y_curr:.1f})"); text_item.set_visible(True)
            elif anim_type == "Sine Unwrap (Circle to Wave)":
                line.set_data(x[:limit], y_eval[:limit])
                x_curr, y_curr = x[limit-1], y_eval[limit-1]
                cx = np.cos(x_curr); cy = y_curr
                tangent_line.set_data([0, cx], [0, cy]); tangent_line.set_visible(True)
                proj_x.set_data([cx, x_curr], [cy, y_curr]); proj_x.set_visible(True)
                if point.get_visible(): point.set_data([x_curr], [y_curr])
            elif anim_type == "Cosine Unwrap (Circle to Wave)":
                line.set_data(x[:limit], y_eval[:limit])
                x_curr, y_curr = x[limit-1], y_eval[limit-1]
                cx = y_curr; cy = np.sin(x_curr)
                tangent_line.set_data([0, cx], [0, cy]); tangent_line.set_visible(True)
                proj_x.set_data([0, x_curr], [y_curr, y_curr]); proj_x.set_visible(True)
                proj_y.set_data([cx, cx], [0, cy]); proj_y.set_visible(True)
                if point.get_visible(): point.set_data([x_curr], [y_curr])
            elif anim_type == "None (Static)":
                line.set_data(x, y_eval)
                
            if i == 0 and point.get_visible():
                try: cam_x, cam_y = float(point.get_data()[0][0]), float(point.get_data()[1][0])
                except: pass

        if scene_effect == "Zoom In":
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            rx = (x_max - x_min) * (1 + global_progress * 2); ry = (y_max - y_min) * (1 + global_progress * 2)
            self.ax.set_xlim(cx - rx/2, cx + rx/2); self.ax.set_ylim(cy - ry/2, cy + ry/2)
        elif scene_effect == "Zoom Out":
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            rx = max(0.1, (x_max - x_min) * global_progress); ry = max(0.1, (y_max - y_min) * global_progress)
            self.ax.set_xlim(cx - rx/2, cx + rx/2); self.ax.set_ylim(cy - ry/2, cy + ry/2)
        elif scene_effect == "Camera Pan (Follow L1)" and cam_x is not None:
            vw = (x_max - x_min) * 0.15; self.ax.set_xlim(cam_x - vw/2, cam_x + vw/2)
        elif scene_effect == "Sweeping Bar":
            sweep_x = x_min + (x_max - x_min) * global_progress
            if layers_data:
                bar = layers_data[0]["proj_x"]
                bar.set_data([sweep_x, sweep_x], [y_min, y_max]); bar.set_color("yellow")
                bar.set_linewidth(10); bar.set_alpha(0.3); bar.set_visible(True)

        if intersect_check.isChecked() and len(y_vals_all) >= 2 and global_progress > 0.1:
            x_global = np.linspace(x_min, x_max, 1000)
            try:
                y1 = self.eval_func(layers_data[0]["widget"].func_input.text(), x_global, t=current_time)
                y2 = self.eval_func(layers_data[1]["widget"].func_input.text(), x_global, t=current_time)
                idxs = np.where(np.diff(np.sign(y1 - y2)))[0]
                for idx in idxs:
                    self.ax.plot([x_global[idx]], [y1[idx]], 'o', color="red", markersize=10 + 5 * np.sin(frame * 0.5), alpha=0.8)
            except: pass

        artists = []
        for data in layers_data:
            artists.extend([data["line"], data["point"], data["text"], data["polygon"], data["tangent"], data["proj_x"], data["proj_y"]])
            artists.extend(data["segments"])
        return artists

    def calculate_total_frames(self):
        anim_frames = self.frames_spin.value(); fps = self.fps_spin.value()
        intro_frames = int(self.intro_hold_spin.value() * fps)
        outro_frames = int(self.outro_hold_spin.value() * fps)
        return intro_frames + anim_frames + outro_frames, anim_frames, intro_frames

    def stop_preview(self):
        if self.ani:
            self.ani.event_source.stop()
            self.statusBar().showMessage("Preview stopped.", 3000)
            self.stop_btn.setEnabled(False)
            self.preview_btn.setEnabled(True)

    def preview_animation(self):
        self.stop_preview()
        self.setup_plot()
        try:
            fps = self.fps_spin.value()
            total_frames, anim_frames, intro_frames = self.calculate_total_frames()
            scene_effect = self.scene_combo.currentText()
            intersect = self.intersect_check.isChecked()
            easing = self.easing_combo.currentText()
            boomerang = self.boomerang_check.isChecked()
            
            layers_data = self.init_animation_objects()

            def init():
                for data in layers_data:
                    data["line"].set_data([], []); data["point"].set_data([], [])
                    for seg in data["segments"]: seg.set_visible(False)
                return []

            def update(frame):
                return self.animation_logic(frame, total_frames, anim_frames, intro_frames, layers_data, scene_effect, intersect, easing, boomerang)

            self.ani = animation.FuncAnimation(self.figure, update, frames=total_frames, init_func=init,
                                               blit=False, interval=1000/fps, repeat=True)
            self.canvas.draw()
            self.statusBar().showMessage(f"Previewing {total_frames} frames ({total_frames/fps:.1f}s)...")
            self.stop_btn.setEnabled(True)
            self.preview_btn.setEnabled(False)
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}", 5000)

    def export_frame(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Frame", "", "PNG Image (*.png)")
        if not file_path: return
        if not file_path.endswith('.png'): file_path += '.png'
        
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=self.figure.get_facecolor())
            self.statusBar().showMessage(f"Frame exported to {file_path}", 5000)
            QMessageBox.information(self, "Success", f"Frame saved successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save frame:\n{str(e)}")

    def render_video(self):
        export_fmt = self.export_format_combo.currentText()
        ext = ".mp4" if "MP4" in export_fmt else ".gif"
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save {export_fmt}", "", f"{export_fmt} (*{ext})")
        if not file_path: return
        if not file_path.endswith(ext): file_path += ext

        # FFmpeg check for MP4
        if ext == ".mp4" and shutil.which("ffmpeg") is None:
            QMessageBox.critical(self, "FFmpeg Missing", "FFmpeg is required to export MP4 files but was not found on your system.\n\nPlease install FFmpeg or export as GIF instead.")
            return

        self.render_dialog = QDialog(self)
        self.render_dialog.setWindowTitle("Rendering Media")
        self.render_dialog.setModal(True)
        self.render_dialog.setFixedSize(300, 100)
        dialog_layout = QVBoxLayout(self.render_dialog)
        
        lbl = QLabel(f"Rendering {export_fmt}... Please wait.")
        lbl.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(lbl)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)
        dialog_layout.addWidget(progress_bar)
        
        self.statusBar().showMessage(f"Rendering to {file_path}... Please wait.")
        QApplication.processEvents()
        
        def start_render():
            try:
                self.setup_plot()
                fps = self.fps_spin.value()
                total_frames, anim_frames, intro_frames = self.calculate_total_frames()
                scene_effect = self.scene_combo.currentText()
                intersect = self.intersect_check.isChecked()
                easing = self.easing_combo.currentText()
                boomerang = self.boomerang_check.isChecked()
                
                layers_data = self.init_animation_objects()

                def init():
                    for data in layers_data:
                        data["line"].set_data([], []); data["point"].set_data([], [])
                        for seg in data["segments"]: seg.set_visible(False)
                    return []

                def update(frame):
                    return self.animation_logic(frame, total_frames, anim_frames, intro_frames, layers_data, scene_effect, intersect, easing, boomerang)

                ani = animation.FuncAnimation(self.figure, update, frames=total_frames, init_func=init, blit=False, interval=1000/fps)
                
                quality_map = {"480p": 60, "720p": 120, "1080p (HD)": 240, "1440p (2K)": 320, "2160p (4K)": 480}
                dpi = quality_map[self.quality_combo.currentText()]
                self.figure.set_dpi(dpi)
                
                out_path = file_path
                
                if ext == ".gif":
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(out_path, writer=writer)
                else:
                    if self.audio_path: out_path = file_path + ".tmp.mp4"
                    writer = animation.FFMpegWriter(fps=fps, bitrate=10000, extra_args=['-threads', '4'])
                    ani.save(out_path, writer=writer)
                    
                    if self.audio_path:
                        lbl.setText("Merging audio... Please wait.")
                        QApplication.processEvents()
                        cmd = f'ffmpeg -y -i "{out_path}" -i "{self.audio_path}" -c:v copy -c:a aac -shortest "{file_path}"'
                        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        if res.returncode != 0: raise Exception(f"FFmpeg audio merge failed:\n{res.stderr}")
                        os.remove(out_path)
                
                self.render_dialog.close()
                self.figure.set_dpi(100)
                self.setup_plot() # Reset plot back to default dpi view
                self.statusBar().showMessage(f"Success! Saved to {file_path}", 5000)
                QMessageBox.information(self, "Success", f"Media saved successfully to:\n{file_path}")
                
            except Exception as e:
                self.render_dialog.close()
                self.figure.set_dpi(100)
                self.setup_plot()
                self.statusBar().showMessage("Rendering failed.", 5000)
                QMessageBox.critical(self, "Rendering Error", f"Failed to render.\n\nError: {str(e)}")

        QTimer.singleShot(100, start_render)
        self.render_dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MathStudioPro()
    window.show()
    sys.exit(app.exec())