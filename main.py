import sys
import math
import time
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QLineEdit, QComboBox, 
                               QPushButton, QFrame, QCheckBox, QFileDialog, 
                               QDialog, QFormLayout, QDialogButtonBox, QMessageBox)
from PySide6.QtCore import Qt, QTimer, QPointF, QBuffer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QImage

# --- MATH COMPILER ENGINE ---
def compile_formula(expr, N):
    """Safely compiles user string into a Python function."""
    expr = expr.strip()
    
    math_env = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "abs": abs, "pow": pow,
        "exp": math.exp, "log": math.log, "min": min, "max": max,
        "floor": math.floor, "round": round, "pi": math.pi, "e": math.e,
        "N": N
    }

    try:
        if 'return' not in expr:
            expr = "return " + expr
            
        indented_expr = "    " + expr.replace('\n', '\n    ')
        code = f"def _user_formula(n, t, a, b, c):\n{indented_expr}"
        
        local_ns = {}
        exec(code, {"__builtins__": {}, **math_env}, local_ns)
        return local_ns['_user_formula']
    except Exception:
        return None

# --- PRESETS ---
PRESETS = {
    "Parametric Spring (a,b,c)": {
        "x": "a * 50 * cos(n * 0.2 + t)",
        "y": "b * 50 * sin(n * 0.2 + t)",
        "z": "n * c * 0.5 - 400"
    },
    "Trefoil Knot (a,b,c)": {
        "x": "a * 100 * (sin(n*0.05) + 2*sin(2*n*0.05))",
        "y": "a * 100 * (cos(n*0.05) - 2*cos(2*n*0.05))",
        "z": "b * 100 * (-sin(3*n*0.05))"
    },
    "Phyllotaxis (Sunflower)": {
        "x": "4 * sqrt(n) * cos(n * 137.5 * pi / 180)",
        "y": "4 * sqrt(n) * sin(n * 137.5 * pi / 180)",
        "z": "c * 10"
    },
    "Pinecone (3D Spire)": {
        "x": "4 * sqrt(n) * cos(n * 137.5 * pi / 180)",
        "y": "4 * sqrt(n) * sin(n * 137.5 * pi / 180)",
        "z": "n * 0.2 - 150"
    },
    "Fibonacci Sphere": {
        "x": "z_val = (1 - 2*(n/N)) * 100\nreturn sqrt(100*100 - z_val*z_val) * cos(n * 137.5 * pi / 180)",
        "y": "z_val = (1 - 2*(n/N)) * 100\nreturn sqrt(100*100 - z_val*z_val) * sin(n * 137.5 * pi / 180)",
        "z": "(1 - 2*(n/N)) * 100"
    },
    "DNA Double Helix": {
        "x": "50 * cos(n * 0.2 + t)",
        "y": "50 * sin(n * 0.2 + t)",
        "z": "n * 0.5 - 400"
    },
    "3D Rose Curve": {
        "x": "200 * cos(n * 0.1) * cos(n * 0.05 + t)",
        "y": "200 * cos(n * 0.1) * sin(n * 0.05 + t)",
        "z": "100 * sin(n * 0.1 + t)"
    },
    "Lissajous 3D Curve": {
        "x": "150 * sin(a * n * 0.05 + t)",
        "y": "150 * sin(b * n * 0.05)",
        "z": "150 * sin(c * n * 0.05 + t/2)"
    },
    "Möbius Strip": {
        "x": "u = n * 0.1\nv = (n/N) * 4 - 2\nreturn (100 + v * 30 * cos(u/2)) * cos(u)",
        "y": "u = n * 0.1\nv = (n/N) * 4 - 2\nreturn (100 + v * 30 * cos(u/2)) * sin(u)",
        "z": "u = n * 0.1\nv = (n/N) * 4 - 2\nreturn v * 30 * sin(u/2)"
    },
    "Tornado Vortex": {
        "x": "r = n/N * a * 150\nreturn r * cos(n * 0.5 + t)",
        "y": "r = n/N * a * 150\nreturn r * sin(n * 0.5 + t)",
        "z": "(n/N) * 300 - 150"
    },
    "Ocean Waves Grid": {
        "x": "(n % 50) * 10 - 250",
        "y": "floor(n / 50) * 10 - 250",
        "z": "x_val = (n % 50) * 10 - 250\ny_val = floor(n / 50) * 10 - 250\nreturn 30 * sin(x_val * 0.05 + t) + 30 * cos(y_val * 0.05 + t)"
    }
}

# --- 3D CANVAS WIDGET ---
class MathCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #05050a;")
        
        self.points = []
        self.grid_points = []
        self.time = 0.0
        self.fps = 0
        
        # Settings
        self.point_size = 3.0
        self.line_dist = 0
        self.show_grid = True
        self.show_axes = True
        self.auto_rotate = False
        
        # Export state
        self.is_exporting = False
        self.export_progress = 0.0
        
        # Camera state
        self.rotX = 0.5
        self.rotY = 0.0
        self.zoom = 1.0
        
        # Camera Physics
        self.velX = 0.0
        self.velY = 0.0
        
        self.isDragging = False
        self.lastX = 0
        self.lastY = 0

        self.init_grid()

    def init_grid(self):
        self.grid_points = []
        size = 300
        step = 30
        for i in range(-size, size + 1, step):
            for j in range(-size, size + 1, step):
                self.grid_points.append((i, -200, j))

    def project(self, x, y, z):
        cosY = math.cos(self.rotY)
        sinY = math.sin(self.rotY)
        x1 = x * cosY - z * sinY
        z1 = x * sinY + z * cosY

        cosX = math.cos(self.rotX)
        sinX = math.sin(self.rotX)
        y1 = y * cosX - z1 * sinX
        z2 = y * sinX + z1 * cosX

        fov = 600
        scale = (fov / (fov + z2 + 400)) * self.zoom
        
        return {
            "x": x1 * scale + self.width() / 2,
            "y": y1 * scale + self.height() / 2,
            "scale": scale,
            "depth": z2
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(5, 5, 10))
        
        # Draw HUD/FPS
        painter.setPen(QColor(0, 242, 255, 150))
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(15, 20, f"FPS: {self.fps}")
        
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Draw Grid Floor
        if self.show_grid:
            proj_grid = [self.project(*p) for p in self.grid_points]
            grid_pen = QPen(QColor(30, 30, 50, 150), 1.0)
            painter.setPen(grid_pen)
            size = 300
            step = 30
            divs = (size * 2) // step + 1
            
            for i in range(divs):
                p1 = proj_grid[i * divs]
                p2 = proj_grid[(i * divs) + divs - 1]
                painter.drawLine(QPointF(p1['x'], p1['y']), QPointF(p2['x'], p2['y']))
                p3 = proj_grid[i]
                p4 = proj_grid[divs * (divs - 1) + i]
                painter.drawLine(QPointF(p3['x'], p3['y']), QPointF(p4['x'], p4['y']))

        # Draw 3D Axes
        if self.show_axes:
            axes = [
                ((0,0,0), (60,0,0), QColor(255, 50, 50)), 
                ((0,0,0), (0,60,0), QColor(50, 255, 50)), 
                ((0,0,0), (0,0,60), QColor(50, 50, 255))  
            ]
            pen = QPen()
            pen.setWidthF(2.0)
            for start, end, color in axes:
                p1 = self.project(*start)
                p2 = self.project(*end)
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawLine(QPointF(p1['x'], p1['y']), QPointF(p2['x'], p2['y']))

        # Neon glow blending
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        sorted_points = sorted(self.points, key=lambda p: p['depth'])

        # Draw lines
        if self.line_dist > 0 and len(sorted_points) > self.line_dist:
            pen = QPen()
            pen.setWidthF(1.0)
            for i in range(len(sorted_points) - self.line_dist):
                p1 = sorted_points[i]
                p2 = sorted_points[i + self.line_dist]
                dist = math.hypot(p1['x']-p2['x'], p1['y']-p2['y'])
                if dist < 80 * self.zoom:
                    color = QColor(p1['color'])
                    color.setAlphaF(0.4 * p1['scale'])
                    pen.setColor(color)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(p1['x'], p1['y']), QPointF(p2['x'], p2['y']))

        # Draw points
        painter.setPen(Qt.NoPen)
        for p in sorted_points:
            radius = max(0.5, self.point_size * p['scale'])
            
            halo_color = QColor(p['color'])
            halo_color.setAlphaF(0.2 * p['scale'])
            painter.setBrush(QBrush(halo_color))
            painter.drawEllipse(QPointF(p['x'], p['y']), radius * 2.5, radius * 2.5)
            
            core_color = QColor(p['color'])
            core_color.setAlphaF(min(1.0, 0.8 + p['scale']*0.2))
            painter.setBrush(QBrush(core_color))
            painter.drawEllipse(QPointF(p['x'], p['y']), radius, radius)

        # Export Overlay
        if self.is_exporting:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 150))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, f"RENDERING VIDEO\n{self.export_progress*100:.1f}%")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.is_exporting:
            self.isDragging = True
            self.lastX = event.position().x()
            self.lastY = event.position().y()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.isDragging = False
            self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if self.isDragging and not self.is_exporting:
            dx = event.position().x() - self.lastX
            dy = event.position().y() - self.lastY
            self.velY = dx * 0.005
            self.velX = dy * 0.005
            self.rotY += self.velY
            self.rotX += self.velX
            self.lastX = event.position().x()
            self.lastY = event.position().y()

    def wheelEvent(self, event):
        if not self.is_exporting:
            delta = event.angleDelta().y()
            if delta > 0: self.zoom *= 1.1
            else: self.zoom *= 0.9
            self.zoom = max(0.1, min(self.zoom, 5.0))


# --- VIDEO EXPORT DIALOG ---
class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export MP4 Video")
        self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: #141423; color: white;")
        
        layout = QFormLayout(self)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "YouTube Video (1920x1080 16:9)",
            "YouTube Video (1280x720 16:9)",
            "YouTube Shorts (1080x1920 9:16)",
            "YouTube Shorts (720x1280 9:16)"
        ])
        layout.addRow("Format:", self.format_combo)
        
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["30 FPS", "60 FPS"])
        layout.addRow("Frame Rate:", self.fps_combo)
        
        self.duration_spin = QSlider(Qt.Horizontal)
        self.duration_spin.setMinimum(1)
        self.duration_spin.setMaximum(30)
        self.duration_spin.setValue(5)
        self.duration_lbl = QLabel("5 seconds")
        self.duration_spin.valueChanged.connect(lambda v: self.duration_lbl.setText(f"{v} seconds"))
        
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(self.duration_spin)
        dur_layout.addWidget(self.duration_lbl)
        layout.addRow("Duration:", dur_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("QPushButton { background-color: #333; border: 1px solid #555; padding: 5px 15px; border-radius: 4px; }"
                              "QPushButton:hover { background-color: #00f2ff; color: black; }")
        layout.addRow(buttons)

    def get_settings(self):
        fmt = self.format_combo.currentText()
        if "1920x1080" in fmt: dims = (1920, 1080)
        elif "1280x720" in fmt: dims = (1280, 720)
        elif "1080x1920" in fmt: dims = (1080, 1920)
        else: dims = (720, 1280)
        
        fps = 60 if "60" in self.fps_combo.currentText() else 30
        duration = self.duration_spin.value()
        return dims, fps, duration


# --- MAIN APPLICATION WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Formula Engine 3D - Studio")
        self.resize(1200, 800)
        
        self.pointCount = 1500
        self.theme = "Rainbow"
        self.is_playing = True
        self.time_speed = 1.0
        
        self.param_a = 1.0
        self.param_b = 1.0
        self.param_c = 1.0
        
        self.fnX, self.fnY, self.fnZ = None, None, None
        self.is_exporting = False
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QHBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.canvas = MathCanvas()
        self.layout.addWidget(self.canvas, 1)

        self.panel = QFrame()
        self.panel.setFixedWidth(360)
        self.panel.setStyleSheet("QFrame { background-color: #141423; border-right: 1px solid #333; }")
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(12)

        title = QLabel("Formula Engine 3D Studio")
        title.setStyleSheet("color: #00f2ff; font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)

        hint = QLabel("Variables:\nn (seed), t (time), N (total)\na, b, c (param sliders)")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        panel_layout.addWidget(self._styled_label("Preset Formulas"))
        panel_layout.addWidget(self.preset_combo)

        self.input_x = QLineEdit()
        self.input_y = QLineEdit()
        self.input_z = QLineEdit()
        for inp in [self.input_x, self.input_y, self.input_z]:
            inp.setStyleSheet(self._input_style())
        
        self.input_x.textChanged.connect(lambda: self.update_formulas('x'))
        self.input_y.textChanged.connect(lambda: self.update_formulas('y'))
        self.input_z.textChanged.connect(lambda: self.update_formulas('z'))

        panel_layout.addWidget(self._styled_label("X ="))
        panel_layout.addWidget(self.input_x)
        panel_layout.addWidget(self._styled_label("Y ="))
        panel_layout.addWidget(self.input_y)
        panel_layout.addWidget(self._styled_label("Z ="))
        panel_layout.addWidget(self.input_z)

        # Param Sliders
        param_layout = QHBoxLayout()
        self.slider_a, self.lbl_a = self._create_slider(1, 50, 10)
        self.slider_a.valueChanged.connect(lambda v: self.update_param('a', v))
        param_layout.addWidget(self._build_param_widget("a", self.slider_a, self.lbl_a))
        
        self.slider_b, self.lbl_b = self._create_slider(1, 50, 10)
        self.slider_b.valueChanged.connect(lambda v: self.update_param('b', v))
        param_layout.addWidget(self._build_param_widget("b", self.slider_b, self.lbl_b))
        
        self.slider_c, self.lbl_c = self._create_slider(1, 50, 10)
        self.slider_c.valueChanged.connect(lambda v: self.update_param('c', v))
        param_layout.addWidget(self._build_param_widget("c", self.slider_c, self.lbl_c))
        panel_layout.addLayout(param_layout)

        # Animation Layout
        anim_layout = QHBoxLayout()
        self.play_btn = QPushButton("Pause")
        self.play_btn.setCheckable(True)
        self.play_btn.setStyleSheet(self._btn_style())
        self.play_btn.clicked.connect(self.toggle_play)
        anim_layout.addWidget(self.play_btn)
        
        self.speed_slider, self.speed_label = self._create_slider(0, 50, 10)
        self.speed_slider.valueChanged.connect(self.update_speed)
        anim_layout.addWidget(QLabel("Speed:"))
        anim_layout.addWidget(self.speed_label)
        anim_layout.addWidget(self.speed_slider)
        panel_layout.addLayout(anim_layout)

        # Visuals Layout
        vis_layout = QHBoxLayout()
        self.size_slider, self.size_label = self._create_slider(1, 10, 3)
        self.size_slider.valueChanged.connect(self.update_point_size)
        vis_layout.addWidget(QLabel("Size:"))
        vis_layout.addWidget(self.size_label)
        vis_layout.addWidget(self.size_slider)
        panel_layout.addLayout(vis_layout)

        self.line_slider, self.line_val_label = self._create_slider(0, 8, 0)
        self.line_slider.valueChanged.connect(self.update_line_dist)
        panel_layout.addWidget(self._styled_label("Spiral Connections:"))
        panel_layout.addWidget(self.line_val_label)
        panel_layout.addWidget(self.line_slider)

        self.count_slider, self.count_val_label = self._create_slider(100, 3000, 1500)
        self.count_slider.valueChanged.connect(self.update_count)
        panel_layout.addWidget(self._styled_label("Seed Count (N):"))
        panel_layout.addWidget(self.count_val_label)
        panel_layout.addWidget(self.count_slider)

        # Checkboxes
        check_layout = QHBoxLayout()
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(lambda s: setattr(self.canvas, 'show_grid', bool(s)))
        check_layout.addWidget(self.grid_check)
        
        self.axes_check = QCheckBox("Axes (RGB)")
        self.axes_check.setChecked(True)
        self.axes_check.stateChanged.connect(lambda s: setattr(self.canvas, 'show_axes', bool(s)))
        check_layout.addWidget(self.axes_check)
        
        self.auto_rot_check = QCheckBox("Auto-Rotate")
        self.auto_rot_check.stateChanged.connect(lambda s: setattr(self.canvas, 'auto_rotate', bool(s)))
        check_layout.addWidget(self.auto_rot_check)
        panel_layout.addLayout(check_layout)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Rainbow", "Fire", "Ice", "Gold", "Depth"])
        self.theme_combo.currentTextChanged.connect(lambda t: setattr(self, 'theme', t))
        panel_layout.addWidget(self._styled_label("Color Theme"))
        panel_layout.addWidget(self.theme_combo)

        # Buttons Layout
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.setStyleSheet(self._btn_style())
        reset_btn.clicked.connect(self.reset_camera)
        btn_layout.addWidget(reset_btn)
        
        export_btn = QPushButton("Export MP4")
        export_btn.setStyleSheet("""
            QPushButton { background-color: #aa00ff; color: white; border: 1px solid #cc44ff; 
                          border-radius: 6px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #cc44ff; }
        """)
        export_btn.clicked.connect(self.start_video_export)
        btn_layout.addWidget(export_btn)
        panel_layout.addLayout(btn_layout)

        panel_layout.addStretch()
        self.layout.addWidget(self.panel)

        self.load_preset(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def _build_param_widget(self, name, slider, label):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        hdr = QLabel(name)
        hdr.setStyleSheet("color: #ff00ff; font-weight: bold;")
        hdr.setAlignment(Qt.AlignCenter)
        layout.addWidget(hdr)
        layout.addWidget(label)
        layout.addWidget(slider)
        return widget

    def _input_style(self):
        return """
            QLineEdit { background-color: #111; color: white; border: 1px solid #444; 
                        border-radius: 4px; padding: 8px; font-family: Consolas, monospace; }
            QLineEdit:focus { border: 1px solid #00f2ff; }
            QLineEdit[error="true"] { border: 1px solid #ff4d4d; }
        """

    def _btn_style(self):
        return """
            QPushButton { background-color: #333; color: white; border: 1px solid #555; 
                          border-radius: 6px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #444; }
            QPushButton:checked { background-color: #00f2ff; color: black; }
        """

    def _styled_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; font-size: 12px; text-transform: uppercase;")
        return lbl

    def _create_slider(self, min_val, max_val, init_val):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #333; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #00f2ff; width: 14px; height: 14px; 
                                         margin: -5px 0; border-radius: 7px; }
        """)
        label = QLabel(str(init_val))
        label.setStyleSheet("color: #00f2ff; font-weight: bold;")
        label.setFixedWidth(30)
        label.setAlignment(Qt.AlignRight)
        return slider, label

    def load_preset(self, index):
        preset_name = self.preset_combo.itemText(index)
        p = PRESETS[preset_name]
        for inp, text in [(self.input_x, p['x']), (self.input_y, p['y']), (self.input_z, p['z'])]:
            inp.blockSignals(True)
            inp.setText(text)
            inp.setProperty('error', 'false')
            inp.style().unpolish(inp)
            inp.style().polish(inp)
            inp.blockSignals(False)
        self.update_formulas()

    def update_formulas(self, changed_axis=None):
        fn_x = compile_formula(self.input_x.text(), self.pointCount)
        fn_y = compile_formula(self.input_y.text(), self.pointCount)
        fn_z = compile_formula(self.input_z.text(), self.pointCount)
        
        for inp, fn in [(self.input_x, fn_x), (self.input_y, fn_y), (self.input_z, fn_z)]:
            is_error = fn is None
            inp.setProperty('error', 'true' if is_error else 'false')
            inp.style().unpolish(inp)
            inp.style().polish(inp)
            
        if fn_x and fn_y and fn_z:
            self.fnX, self.fnY, self.fnZ = fn_x, fn_y, fn_z

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.setText("Play" if not self.is_playing else "Pause")

    def update_speed(self, val):
        self.time_speed = val / 10.0
        self.speed_label.setText(str(val))

    def update_param(self, name, val):
        setattr(self, f'param_{name}', val / 10.0)
        getattr(self, f'lbl_{name}').setText(f"{val/10.0:.1f}")

    def update_point_size(self, val):
        self.canvas.point_size = float(val)
        self.size_label.setText(str(val))

    def update_line_dist(self, val):
        self.canvas.line_dist = val
        self.line_val_label.setText(str(val))

    def update_count(self, val):
        self.pointCount = val
        self.count_val_label.setText(str(val))
        self.update_formulas()

    def reset_camera(self):
        self.canvas.rotX = 0.5
        self.canvas.rotY = 0.0
        self.canvas.zoom = 1.0
        self.canvas.velX = 0
        self.canvas.velY = 0

    def calculate_frame(self):
        """Core math calculation. Separated so the exporter can call it manually."""
        if not (self.fnX and self.fnY and self.fnZ):
            self.canvas.points = []
            return

        points_data = []
        t = self.canvas.time
        
        for n in range(self.pointCount):
            try:
                x = self.fnX(n, t, self.param_a, self.param_b, self.param_c)
                y = self.fnY(n, t, self.param_a, self.param_b, self.param_c)
                z = self.fnZ(n, t, self.param_a, self.param_b, self.param_c)
                
                if None in (x, y, z) or math.isnan(x) or math.isnan(y) or math.isnan(z):
                    continue
                    
                p = self.canvas.project(x, y, z)
                p['color'] = self.get_color(n, p['scale'])
                points_data.append(p)
            except Exception:
                pass 
                
        self.canvas.points = points_data

    def animate(self):
        if self.is_exporting:
            return # The export function handles the loop during rendering

        # FPS Calculation
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.canvas.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

        # Camera physics
        if self.canvas.auto_rotate and not self.canvas.isDragging:
            self.canvas.rotY += 0.005
        elif not self.canvas.isDragging:
            self.canvas.rotY += self.canvas.velY
            self.canvas.rotX += self.canvas.velX
            self.canvas.velX *= 0.95
            self.canvas.velY *= 0.95

        if self.is_playing:
            self.canvas.time += 0.02 * self.time_speed

        self.calculate_frame()
        self.canvas.update()

    def get_color(self, n, scale):
        hue, sat, light = 0, 100, 50 + (scale * 20)
        if self.theme == "Fire": hue = (n / self.pointCount) * 60
        elif self.theme == "Ice": hue = 180 + (n / self.pointCount) * 60
        elif self.theme == "Gold":
            hue = 45
            light = 40 + (n / self.pointCount) * 40
        elif self.theme == "Depth":
            hue = 200
            sat = 0
            light = 30 + (scale * 60)
        else: 
            hue = (n / self.pointCount) * 360
            
        light = min(80, light)
        return f"hsl({hue}, {sat}%, {light}%)"

    def start_video_export(self):
        # Check dependencies
        try:
            import imageio.v2 as imageio
            import numpy as np
        except ImportError:
            QMessageBox.critical(self, "Missing Dependencies", 
                                 "To export MP4 videos, please install the required libraries:\n\n"
                                 "pip install imageio imageio-ffmpeg numpy")
            return

        dialog = ExportDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
            
        dims, fps, duration = dialog.get_settings()
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Video", "formula_animation.mp4", "MP4 Files (*.mp4)")
        if not file_name:
            return

        self.is_exporting = True
        self.canvas.is_exporting = True
        self.setEnabled(False) # Block user input
        
        # Temporarily hide UI panel for pure 16:9 / 9:16 canvas recording
        self.panel.setVisible(False)
        original_geometry = self.geometry()
        
        # Resize window to exact target dimension
        self.resize(dims[0], dims[1])
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - dims[0]) // 2, (screen.height() - dims[1]) // 2)
        
        # Force Qt to process the resize before we start grabbing frames
        QApplication.processEvents()
        time.sleep(0.1) 
        QApplication.processEvents()

        # Setup Video Writer
        writer = imageio.get_writer(file_name, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        
        total_frames = fps * duration
        time_step = 1.0 / fps

        try:
            for i in range(total_frames):
                # Update logic
                self.canvas.time += time_step * self.time_speed
                if self.canvas.auto_rotate:
                    self.canvas.rotY += 0.005
                    
                self.calculate_frame()
                
                # Update overlay
                self.canvas.export_progress = (i + 1) / total_frames
                self.canvas.update()
                QApplication.processEvents() # Force canvas to redraw the frame
                
                # Grab frame
                pixmap = self.canvas.grab()
                qimage = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
                
                # Convert QPixmap to numpy array for imageio
                width = qimage.width()
                height = qimage.height()
                ptr = qimage.bits()
                ptr.setsize(height * width * 4)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
                arr_rgb = arr[:, :, :3] # Drop alpha channel for MP4
                
                writer.append_data(arr_rgb)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during rendering:\n{str(e)}")
        finally:
            writer.close()
            
            # Restore UI
            self.canvas.is_exporting = False
            self.is_exporting = False
            self.panel.setVisible(True)
            self.setEnabled(True)
            self.setGeometry(original_geometry)
            QMessageBox.information(self, "Export Complete", f"Video successfully saved to:\n{file_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("Inter")
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())