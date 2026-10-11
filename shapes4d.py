import sys
import numpy as np
import itertools
import imageio.v2 as imageio
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QSlider, QLabel, QCheckBox, 
                               QComboBox, QGroupBox, QGridLayout, QFileDialog, 
                               QSpinBox, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class FourDShapes:
    """Generates vertices and edges for various 4D (and 5D) shapes"""
    
    @staticmethod
    def find_edges_by_min_distance(vertices):
        """Vectorized edge finding based on the minimum non-zero distance."""
        diff = vertices[:, None, :] - vertices[None, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        valid_dists = dists_sq[dists_sq > 1e-6]
        if len(valid_dists) == 0:
            return np.array([])
        min_dist = np.min(valid_dists)
        mask = np.isclose(dists_sq, min_dist) & np.triu(np.ones_like(dists_sq, dtype=bool), k=1)
        i_idx, j_idx = np.where(mask)
        return np.array(list(zip(i_idx, j_idx)))

    @staticmethod
    def _generate_from_base(base):
        """Helper to generate shapes from permutations of a base coordinate set."""
        perms = set(itertools.permutations(base))
        verts = []
        for p in perms:
            nonzero_idx = [i for i, x in enumerate(p) if x != 0]
            for signs in itertools.product([-1, 1], repeat=len(nonzero_idx)):
                v = [0, 0, 0, 0]
                for idx, sign in zip(nonzero_idx, signs):
                    v[idx] = p[idx] * sign
                verts.append(v)
        vertices = np.array(list(set(verts)), dtype=float)
        edges = FourDShapes.find_edges_by_min_distance(vertices)
        return vertices, edges

    @staticmethod
    def _rectify_shape(verts, edges):
        """Dynamically generates a rectified shape by taking edge midpoints."""
        new_verts = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
        return new_verts, FourDShapes.find_edges_by_min_distance(new_verts)

    @staticmethod
    def _truncate_shape(verts, edges):
        """Dynamically generates a truncated shape by slicing edges at 1/3 and 2/3."""
        new_verts = []
        for i, j in edges:
            new_verts.append(verts[i] + (verts[j] - verts[i]) / 3.0)
            new_verts.append(verts[i] + 2.0 * (verts[j] - verts[i]) / 3.0)
        new_verts = np.array(new_verts)
        return new_verts, FourDShapes.find_edges_by_min_distance(new_verts)

    @staticmethod
    def get_tesseract():
        return FourDShapes._generate_from_base([1, 1, 1, 1])

    @staticmethod
    def get_16_cell():
        return FourDShapes._generate_from_base([1, 0, 0, 0])

    @staticmethod
    def get_5_cell():
        s = 1 / np.sqrt(5)
        vertices = np.array([
            [1, 1, 1, -s],
            [1, -1, -1, -s],
            [-1, 1, -1, -s],
            [-1, -1, 1, -s],
            [0, 0, 0, np.sqrt(5) - s]
        ], dtype=float)
        edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
        return vertices, np.array(edges)

    @staticmethod
    def get_rectified_5_cell():
        v, e = FourDShapes.get_5_cell()
        return FourDShapes._rectify_shape(v, e)

    @staticmethod
    def get_truncated_5_cell():
        v, e = FourDShapes.get_5_cell()
        return FourDShapes._truncate_shape(v, e)

    @staticmethod
    def get_24_cell():
        return FourDShapes._generate_from_base([1, 1, 0, 0])

    @staticmethod
    def get_rectified_24_cell():
        v, e = FourDShapes.get_24_cell()
        return FourDShapes._rectify_shape(v, e)

    @staticmethod
    def get_truncated_24_cell():
        v, e = FourDShapes.get_24_cell()
        return FourDShapes._truncate_shape(v, e)

    @staticmethod
    def get_clifford_torus(resolution=16):
        theta = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        phi = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        vertices = []
        for t, p in zip(theta_grid.flatten(), phi_grid.flatten()):
            vertices.append([np.cos(t), np.sin(t), np.cos(p), np.sin(p)])
            
        vertices = np.array(vertices, dtype=float)
        
        edges = []
        res = resolution
        for i in range(res):
            for j in range(res):
                idx = i * res + j
                edges.append((idx, ((i+1)%res)*res + j))
                edges.append((idx, i*res + (j+1)%res))
                
        return vertices, np.array(edges)

    @staticmethod
    def even_permutations(seq):
        from itertools import permutations
        seen = set()
        for p in permutations(seq):
            if p in seen:
                continue
            inv = sum(1 for i in range(len(p)) for j in range(i+1, len(p)) if p[i] > p[j])
            if inv % 2 == 0:
                seen.add(p)
                yield p

    @staticmethod
    def get_600_cell():
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        verts = []
        
        for s in itertools.product([-1, 1], repeat=4):
            verts.append(s)
            
        for i in range(4):
            for s in [-2, 2]:
                v = [0]*4; v[i] = s
                verts.append(tuple(v))
                
        base = [0, 1, phi, inv_phi]
        for p in FourDShapes.even_permutations(base):
            nonzero_idx = [i for i, x in enumerate(p) if x != 0]
            for signs in itertools.product([-1, 1], repeat=3):
                v = [0, 0, 0, 0]
                for idx, sign in zip(nonzero_idx, signs):
                    v[idx] = p[idx] * sign
                verts.append(tuple(v))
                
        vertices = np.array(list(set(verts)), dtype=float)
        edges = FourDShapes.find_edges_by_min_distance(vertices)
        return vertices, edges

    @staticmethod
    def get_hypersphere(resolution=8):
        t1 = np.linspace(0, np.pi/2, resolution)
        t2 = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        t3 = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        
        vertices = []
        for a in t1:
            for b in t2:
                for c in t3:
                    x = np.cos(a) * np.cos(b)
                    y = np.cos(a) * np.sin(b)
                    z = np.sin(a) * np.cos(c)
                    w = np.sin(a) * np.sin(c)
                    vertices.append([x, y, z, w])
                    
        vertices = np.array(vertices, dtype=float) * 1.8
        
        res1, res2, res3 = resolution, resolution, resolution
        edges = []
        for i in range(res1):
            for j in range(res2):
                for k in range(res3):
                    idx = (i * res2 + j) * res3 + k
                    if i < res1 - 1:
                        edges.append((idx, ((i+1) * res2 + j) * res3 + k))
                    edges.append((idx, (i * res2 + (j+1)%res2) * res3 + k))
                    edges.append((idx, (i * res2 + j) * res3 + (k+1)%res3))
                    
        return vertices, np.array(edges)

    @staticmethod
    def get_duocylinder(res=12):
        theta = np.linspace(0, 2*np.pi, res, endpoint=False)
        phi = np.linspace(0, 2*np.pi, res, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        vertices = []
        for t, p in zip(theta_grid.flatten(), phi_grid.flatten()):
            vertices.append([np.cos(t), np.sin(t), np.cos(p), np.sin(p)])
        vertices = np.array(vertices, dtype=float)
        
        edges = []
        for i in range(res):
            for j in range(res):
                idx = i * res + j
                edges.append((idx, ((i+1)%res)*res + j))
                edges.append((idx, i*res + (j+1)%res))
        return vertices, np.array(edges)

    @staticmethod
    def get_octahedral_prism():
        oct_verts = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]
        h = np.sqrt(2) / 2
        verts = []
        for v in oct_verts:
            verts.append([v[0], v[1], v[2], -h])
            verts.append([v[0], v[1], v[2], h])
        vertices = np.array(verts, dtype=float)
        edges = FourDShapes.find_edges_by_min_distance(vertices)
        return vertices, edges

    @staticmethod
    def get_tetrahedral_prism():
        tet_verts = [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]
        h = np.sqrt(8) / 2
        verts = []
        for v in tet_verts:
            verts.append([v[0], v[1], v[2], -h])
            verts.append([v[0], v[1], v[2], h])
        vertices = np.array(verts, dtype=float) * 0.8
        edges = FourDShapes.find_edges_by_min_distance(vertices)
        return vertices, edges

    # --- Tesseract Family (Truncations) ---
    
    @staticmethod
    def get_truncated_tesseract():
        return FourDShapes._generate_from_base([1, 1, 1, 3])

    @staticmethod
    def get_rectified_tesseract():
        return FourDShapes._generate_from_base([1, 1, 1, 0])

    @staticmethod
    def get_cantellated_tesseract():
        return FourDShapes._generate_from_base([1, 1, 1, np.sqrt(2)])

    @staticmethod
    def get_runcinated_tesseract():
        return FourDShapes._generate_from_base([1, 1, 3, 3])

    @staticmethod
    def get_truncated_16_cell():
        return FourDShapes._generate_from_base([0, 0, 1, 2])

    @staticmethod
    def get_bitruncated_tesseract():
        return FourDShapes._generate_from_base([0, 1, 2, 3])

    @staticmethod
    def get_cantitruncated_tesseract():
        return FourDShapes._generate_from_base([1, 3, 4, 4])

    @staticmethod
    def get_runcitruncated_tesseract():
        return FourDShapes._generate_from_base([0, 1, 3, 4])

    @staticmethod
    def get_omnitruncated_tesseract():
        return FourDShapes._generate_from_base([1, 3, 5, 7])

    # --- 5D and 4D Manifold Shapes ---

    @staticmethod
    def get_penteract():
        """5D Cube (32 vertices, 80 edges) projected into 4D."""
        v5 = np.array(list(itertools.product([-1, 1], repeat=5)), dtype=float)
        edges = []
        for i in range(32):
            for j in range(i+1, 32):
                if np.sum(np.abs(v5[i] - v5[j])) == 2:
                    edges.append((i, j))
        # Project 5D to 4D
        w_dist = 5.0
        w_factor = 1.0 / (w_dist - v5[:, 4])
        v4 = v5[:, :4] * w_factor[:, np.newaxis]
        v4 = v4 * 4.0  # Scale to fit standard limits
        return v4, np.array(edges)

    @staticmethod
    def get_hexateron():
        """5D Simplex (6 vertices, 15 edges) projected from 6D into 4D."""
        v6 = np.eye(6) - (1.0/6.0)
        edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
        # Project 6D to 5D
        w1_dist = 4.0
        w1_factor = 1.0 / (w1_dist - v6[:, 5])
        v5 = v6[:, :5] * w1_factor[:, np.newaxis]
        # Project 5D to 4D
        w2_dist = 3.0
        w2_factor = 1.0 / (w2_dist - v5[:, 4])
        v4 = v5[:, :4] * w2_factor[:, np.newaxis]
        v4 = v4 * 10.0  # Scale to fit standard limits
        return v4, np.array(edges)

    @staticmethod
    def get_klein_bottle(res=16):
        """4D Klein Bottle surface (non-intersecting in 4D)."""
        u = np.linspace(0, 2*np.pi, res, endpoint=False)
        v = np.linspace(0, 2*np.pi, res, endpoint=False)
        u_grid, v_grid = np.meshgrid(u, v)
        
        vertices = []
        for ui, vi in zip(u_grid.flatten(), v_grid.flatten()):
            r = 1 + 0.2 * np.cos(vi)
            x = r * np.cos(ui)
            y = r * np.sin(ui)
            z = 0.2 * np.sin(vi) * np.cos(ui/2)
            w = 0.2 * np.sin(vi) * np.sin(ui/2)
            vertices.append([x, y, z, w])
        vertices = np.array(vertices, dtype=float) * 1.2
        
        edges = []
        for i in range(res):
            for j in range(res):
                idx = i * res + j
                edges.append((idx, ((i+1)%res)*res + j))
                edges.append((idx, i*res + (j+1)%res))
        return vertices, np.array(edges)


class ShapeCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor='black')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        
        self.angles = {'xw': 0, 'yw': 0, 'zw': 0}
        self.speeds = {'xw': 0.01, 'yw': 0.015, 'zw': 0.0}
        self.is_playing = True
        self.line_width = 2.5
        self.vertex_size = 20
        self.palette = self.get_palettes()['Cyberpunk']
        self.auto_camera = True
        self.camera_azimuth = 0
        
        self.pulse_enabled = True
        self.color_shift_enabled = True
        self.glow_enabled = True
        self.time = 0.0
        
        self.line_collection = None
        self.glow_collections = [None, None, None]
        self.scatter = None
        
        self.load_shape("Tesseract (8-Cell)")
        self.update_plot(init=True)

    def load_shape(self, shape_name):
        shapes = {
            "Tesseract (8-Cell)": FourDShapes.get_tesseract,
            "Truncated Tesseract": FourDShapes.get_truncated_tesseract,
            "Rectified Tesseract": FourDShapes.get_rectified_tesseract,
            "Cantellated Tesseract": FourDShapes.get_cantellated_tesseract,
            "Runcinated Tesseract": FourDShapes.get_runcinated_tesseract,
            "Bitruncated Tesseract": FourDShapes.get_bitruncated_tesseract,
            "Cantitruncated Tesseract": FourDShapes.get_cantitruncated_tesseract,
            "Runcitruncated Tesseract": FourDShapes.get_runcitruncated_tesseract,
            "Omnitruncated Tesseract": FourDShapes.get_omnitruncated_tesseract,
            "16-Cell (Hyperoctahedron)": FourDShapes.get_16_cell,
            "Truncated 16-Cell": FourDShapes.get_truncated_16_cell,
            "5-Cell (Simplex)": FourDShapes.get_5_cell,
            "Rectified 5-Cell": FourDShapes.get_rectified_5_cell,
            "Truncated 5-Cell": FourDShapes.get_truncated_5_cell,
            "24-Cell (Icositetrachoron)": FourDShapes.get_24_cell,
            "Rectified 24-Cell": FourDShapes.get_rectified_24_cell,
            "Truncated 24-Cell": FourDShapes.get_truncated_24_cell,
            "600-Cell (Hexacosichoron)": FourDShapes.get_600_cell,
            "Clifford Torus (4D Donut)": FourDShapes.get_clifford_torus,
            "Duocylinder": FourDShapes.get_duocylinder,
            "Klein Bottle (4D Surface)": FourDShapes.get_klein_bottle,
            "Octahedral Prism": FourDShapes.get_octahedral_prism,
            "Tetrahedral Prism": FourDShapes.get_tetrahedral_prism,
            "Penteract (5D Cube)": FourDShapes.get_penteract,
            "Hexateron (5D Simplex)": FourDShapes.get_hexateron,
            "4D Sphere (Hypersphere)": FourDShapes.get_hypersphere
        }
        if shape_name in shapes:
            self.vertices_4d, self.edges = shapes[shape_name]()
            
        if self.line_collection:
            self.line_collection.remove()
            self.line_collection = None
        for g in self.glow_collections:
            if g: g.remove()
        self.glow_collections = [None, None, None]
        if self.scatter:
            self.scatter.remove()
            self.scatter = None
            
        self.update_plot(init=True)

    def get_palettes(self):
        return {
            'Cyberpunk': {'v': 'white', 'e1': np.array([0, 1, 1, 0.9]), 'e2': np.array([1, 0, 1, 0.9])},
            'Matrix':    {'v': '#00ff00', 'e1': np.array([0, 1, 0, 0.9]), 'e2': np.array([0, 0.3, 0, 0.9])},
            'Inferno':   {'v': 'white', 'e1': np.array([1, 0.8, 0, 0.9]), 'e2': np.array([1, 0, 0, 0.9])},
            'Ice':       {'v': 'white', 'e1': np.array([0, 0.5, 1, 0.9]), 'e2': np.array([0.8, 0.9, 1, 0.9])},
            'Rainbow':   {'v': 'white', 'e1': np.array([1, 0, 0, 0.9]), 'e2': np.array([0, 0, 1, 0.9])}
        }

    def rotate_4d(self, verts, angles):
        cx, sx = np.cos(angles['xw']), np.sin(angles['xw'])
        cy, sy = np.cos(angles['yw']), np.sin(angles['yw'])
        cz, sz = np.cos(angles['zw']), np.sin(angles['zw'])
        
        rxw = np.array([[cx, 0, 0, -sx], [0, 1, 0, 0], [0, 0, 1, 0], [sx, 0, 0, cx]])
        ryw = np.array([[1, 0, 0, 0], [0, cy, 0, -sy], [0, 0, 1, 0], [0, sy, 0, cy]])
        rzw = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cz, -sz], [0, 0, sz, cz]])
        return np.dot(np.dot(np.dot(verts, rxw), ryw), rzw)

    def project_4d_to_3d(self, verts4d):
        w_distance = 4.0 
        w_factor = 1.0 / (w_distance - verts4d[:, 3])
        verts3d = verts4d[:, :3] * w_factor[:, np.newaxis]
        return verts3d

    def update_plot(self, init=False):
        rotated_4d = self.rotate_4d(self.vertices_4d, self.angles)
        
        if self.pulse_enabled:
            pulse_factor = 1.0 + 0.15 * np.sin(self.time * 2.0)
            rotated_4d = rotated_4d * pulse_factor
            
        verts3d = self.project_4d_to_3d(rotated_4d)
        segments = np.stack((verts3d[self.edges[:, 0]], verts3d[self.edges[:, 1]]), axis=1)
        
        w_coords = rotated_4d[:, 3]
        range_w = w_coords.max() - w_coords.min()
        if range_w < 1e-5:
            norm_w = np.zeros_like(w_coords)
        else:
            norm_w = (w_coords - w_coords.min()) / range_w
            
        avg_w = (norm_w[self.edges[:, 0]] + norm_w[self.edges[:, 1]]) / 2.0
        if self.color_shift_enabled:
            avg_w = (avg_w + self.time * 0.1) % 1.0
            
        c1 = self.palette['e1']
        c2 = self.palette['e2']
        edge_colors = c1 * avg_w[:, None] + c2 * (1 - avg_w)[:, None]
        
        if self.glow_enabled:
            glow_widths = [self.line_width * 5, self.line_width * 3, self.line_width * 1.5]
            glow_alphas = [0.05, 0.1, 0.2]
            for i, lc in enumerate(self.glow_collections):
                if init or lc is None:
                    lc = Line3DCollection(segments, colors=edge_colors.copy(), linewidths=glow_widths[i])
                    self.ax.add_collection3d(lc)
                    self.glow_collections[i] = lc
                else:
                    lc.set_segments(segments)
                    glow_colors = edge_colors.copy()
                    glow_colors[:, 3] *= glow_alphas[i]
                    lc.set_colors(glow_colors)
                    lc.set_linewidths(glow_widths[i])
        else:
            for lc in self.glow_collections:
                if lc is not None:
                    lc.set_segments(np.empty((0, 2, 3)))
        
        if init or self.line_collection is None:
            self.line_collection = Line3DCollection(segments, colors=edge_colors, linewidths=self.line_width)
            self.ax.add_collection3d(self.line_collection)
        else:
            self.line_collection.set_segments(segments)
            self.line_collection.set_colors(edge_colors)
            self.line_collection.set_linewidths(self.line_width)
            
        if init or self.scatter is None:
            self.scatter = self.ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], 
                                           c=self.palette['v'], s=self.vertex_size, depthshade=False)
        else:
            self.scatter._offsets3d = (verts3d[:, 0], verts3d[:, 1], verts3d[:, 2])
            self.scatter.set_sizes([self.vertex_size])
            self.scatter.set_color(self.palette['v'])
        
        limit = 3.0 
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        self.ax.set_axis_off()
        
        if self.auto_camera:
            self.camera_azimuth += 0.2
            self.ax.view_init(elev=20, azim=self.camera_azimuth)
            
        self.fig.tight_layout(pad=0)

    def step_animation(self, dt=1/60):
        if self.is_playing:
            self.time += dt 
            for key in self.angles:
                self.angles[key] += self.speeds[key] * (dt * 60)
            self.update_plot()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4D Geometry Viz - Ultimate Edition v8")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.canvas = ShapeCanvas()
        main_layout.addWidget(self.canvas, stretch=1)
        
        self.controls_widget = QWidget()
        self.controls_widget.setFixedWidth(340)
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        controls_layout.setSpacing(10)
        main_layout.addWidget(self.controls_widget)
        
        self.is_rendering = False
        self.setup_controls()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.canvas.step_animation(1/60))
        self.timer.start(16) 
        
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render_frame)
        self.render_writer = None
        self.render_frame_count = 0
        self.render_total_frames = 0
        
        self.apply_style()

    def setup_controls(self):
        grp_res = QGroupBox("Video Resolution Presets")
        lay_res = QGridLayout()
        lay_res.addWidget(QPushButton("1080p 16:9", clicked=lambda: self.resize(1920, 1080)), 0, 0)
        lay_res.addWidget(QPushButton("1080p 9:16", clicked=lambda: self.resize(1080, 1920)), 0, 1)
        lay_res.addWidget(QPushButton("4K 16:9", clicked=lambda: self.resize(3840, 2160)), 1, 0)
        lay_res.addWidget(QPushButton("4K 9:16", clicked=lambda: self.resize(2160, 3840)), 1, 1)
        grp_res.setLayout(lay_res)
        self.controls_widget.layout().addWidget(grp_res)

        grp_vis = QGroupBox("Visuals & Shapes")
        lay_vis = QVBoxLayout()
        
        lay_vis.addWidget(QLabel("4D Shape:"))
        self.combo_shape = QComboBox()
        self.combo_shape.addItems([
            "Tesseract (8-Cell)", 
            "Truncated Tesseract",
            "Rectified Tesseract",
            "Cantellated Tesseract",
            "Runcinated Tesseract",
            "Bitruncated Tesseract",
            "Cantitruncated Tesseract",
            "Runcitruncated Tesseract",
            "Omnitruncated Tesseract",
            "16-Cell (Hyperoctahedron)", 
            "Truncated 16-Cell",
            "5-Cell (Simplex)",
            "Rectified 5-Cell",
            "Truncated 5-Cell",
            "24-Cell (Icositetrachoron)",
            "Rectified 24-Cell",
            "Truncated 24-Cell",
            "600-Cell (Hexacosichoron)",
            "Clifford Torus (4D Donut)",
            "Duocylinder",
            "Klein Bottle (4D Surface)",
            "Octahedral Prism",
            "Tetrahedral Prism",
            "Penteract (5D Cube)",
            "Hexateron (5D Simplex)",
            "4D Sphere (Hypersphere)"
        ])
        self.combo_shape.currentTextChanged.connect(self.canvas.load_shape)
        lay_vis.addWidget(self.combo_shape)
        
        lay_vis.addWidget(QLabel("Color Palette:"))
        self.combo_palette = QComboBox()
        self.combo_palette.addItems(["Cyberpunk", "Matrix", "Inferno", "Ice", "Rainbow"])
        self.combo_palette.currentTextChanged.connect(self.update_palette)
        lay_vis.addWidget(self.combo_palette)
        
        effects_lay = QHBoxLayout()
        self.chk_pulse = QCheckBox("Pulse")
        self.chk_pulse.setChecked(True)
        self.chk_pulse.stateChanged.connect(lambda s: setattr(self.canvas, 'pulse_enabled', bool(s)))
        effects_lay.addWidget(self.chk_pulse)
        
        self.chk_color_shift = QCheckBox("Flow Colors")
        self.chk_color_shift.setChecked(True)
        self.chk_color_shift.stateChanged.connect(lambda s: setattr(self.canvas, 'color_shift_enabled', bool(s)))
        effects_lay.addWidget(self.chk_color_shift)
        
        self.chk_glow = QCheckBox("Glow/Bloom")
        self.chk_glow.setChecked(True)
        self.chk_glow.stateChanged.connect(lambda s: setattr(self.canvas, 'glow_enabled', bool(s)))
        effects_lay.addWidget(self.chk_glow)
        lay_vis.addLayout(effects_lay)
        
        self.lbl_lw = QLabel("Line Thickness: 2")
        lay_vis.addWidget(self.lbl_lw)
        self.slider_lw = QSlider(Qt.Horizontal)
        self.slider_lw.setMinimum(1)
        self.slider_lw.setMaximum(10)
        self.slider_lw.setValue(2)
        self.slider_lw.valueChanged.connect(lambda v: (setattr(self.canvas, 'line_width', v), self.lbl_lw.setText(f"Line Thickness: {v}")))
        lay_vis.addWidget(self.slider_lw)
        
        grp_vis.setLayout(lay_vis)
        self.controls_widget.layout().addWidget(grp_vis)

        grp_rot = QGroupBox("4D Rotation Animation")
        lay_rot = QVBoxLayout()
        
        lay_rot.addWidget(QLabel("Animation Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Custom / Manual", "Hyper-Spin", "Hypnotic Loop", "Tumble", "Orbit"])
        self.combo_preset.currentTextChanged.connect(self.apply_preset)
        lay_rot.addWidget(self.combo_preset)
        
        grid_rot = QGridLayout()
        grid_rot.addWidget(QLabel("XW Plane:"), 0, 0)
        self.slider_xw = QSlider(Qt.Horizontal)
        self.slider_xw.setValue(10)
        self.slider_xw.valueChanged.connect(lambda v: self.canvas.speeds.update({'xw': v * 0.001}))
        grid_rot.addWidget(self.slider_xw, 0, 1)
        
        grid_rot.addWidget(QLabel("YW Plane:"), 1, 0)
        self.slider_yw = QSlider(Qt.Horizontal)
        self.slider_yw.setValue(15)
        self.slider_yw.valueChanged.connect(lambda v: self.canvas.speeds.update({'yw': v * 0.001}))
        grid_rot.addWidget(self.slider_yw, 1, 1)
        
        grid_rot.addWidget(QLabel("ZW Plane:"), 2, 0)
        self.slider_zw = QSlider(Qt.Horizontal)
        self.slider_zw.setValue(0)
        self.slider_zw.valueChanged.connect(lambda v: self.canvas.speeds.update({'zw': v * 0.001}))
        grid_rot.addWidget(self.slider_zw, 2, 1)
        lay_rot.addLayout(grid_rot)
        
        self.btn_reset = QPushButton("↺ Reset Rotations & Time")
        self.btn_reset.clicked.connect(self.reset_rotations)
        lay_rot.addWidget(self.btn_reset)
        
        grp_rot.setLayout(lay_rot)
        self.controls_widget.layout().addWidget(grp_rot)

        grp_render = QGroupBox("Render to MP4")
        lay_render = QVBoxLayout()
        
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Duration (s):"))
        self.spin_duration = QSpinBox()
        self.spin_duration.setMinimum(1)
        self.spin_duration.setMaximum(300)
        self.spin_duration.setValue(10)
        settings_layout.addWidget(self.spin_duration)
        
        settings_layout.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setMinimum(15)
        self.spin_fps.setMaximum(60)
        self.spin_fps.setValue(30)
        settings_layout.addWidget(self.spin_fps)
        lay_render.addLayout(settings_layout)

        self.btn_render = QPushButton("🎬 Start Rendering to MP4")
        self.btn_render.setStyleSheet("font-weight: bold; background-color: #d40000; border-color: #ff0000;")
        self.btn_render.clicked.connect(self.start_rendering)
        lay_render.addWidget(self.btn_render)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        lay_render.addWidget(self.progress_bar)
        
        grp_render.setLayout(lay_render)
        self.controls_widget.layout().addWidget(grp_render)

        grp_rec = QGroupBox("Live View Tools")
        lay_rec = QVBoxLayout()
        
        self.btn_play = QPushButton("⏸ Pause Live Animation")
        self.btn_play.clicked.connect(self.toggle_play)
        lay_rec.addWidget(self.btn_play)

        self.chk_camera = QCheckBox("Auto-Rotate 3D Camera")
        self.chk_camera.setChecked(True)
        self.chk_camera.stateChanged.connect(lambda s: setattr(self.canvas, 'auto_camera', bool(s)))
        lay_rec.addWidget(self.chk_camera)
        
        self.chk_green = QCheckBox("Green Screen Mode (Chroma Key)")
        self.chk_green.stateChanged.connect(self.toggle_green_screen)
        lay_rec.addWidget(self.chk_green)

        self.chk_present = QCheckBox("Presentation Mode (Hide UI)")
        self.chk_present.stateChanged.connect(self.toggle_presentation)
        lay_rec.addWidget(self.chk_present)
        
        grp_rec.setLayout(lay_rec)
        self.controls_widget.layout().addWidget(grp_rec)
        self.controls_widget.layout().addStretch()

    def reset_rotations(self):
        self.canvas.angles = {'xw': 0, 'yw': 0, 'zw': 0}
        self.canvas.time = 0.0

    def apply_preset(self, preset):
        if preset == "Custom / Manual":
            return
        elif preset == "Hyper-Spin":
            xw, yw, zw = 30, 0, 0
        elif preset == "Hypnotic Loop":
            xw, yw, zw = 10, 0, 7
        elif preset == "Tumble":
            xw, yw, zw = 10, 13, 17
        elif preset == "Orbit":
            xw, yw, zw = 0, 15, 0
        self.slider_xw.setValue(xw)
        self.slider_yw.setValue(yw)
        self.slider_zw.setValue(zw)

    def update_palette(self, name):
        self.canvas.palette = self.canvas.get_palettes()[name]

    def toggle_play(self):
        self.canvas.is_playing = not self.canvas.is_playing
        self.btn_play.setText("▶ Play Live Animation" if not self.canvas.is_playing else "⏸ Pause Live Animation")

    def toggle_green_screen(self, state):
        if state:
            self.canvas.fig.patch.set_facecolor('#00FF00')
            self.canvas.ax.set_facecolor('#00FF00')
            self.setStyleSheet("QMainWindow { background-color: #00FF00; }")
        else:
            self.canvas.fig.patch.set_facecolor('black')
            self.canvas.ax.set_facecolor('black')
            self.apply_style()
        self.canvas.draw_idle()

    def toggle_presentation(self, state):
        self.controls_widget.setHidden(state)

    def set_ui_enabled(self, enabled):
        for widget in self.controls_widget.findChildren(QWidget):
            if widget is not self.btn_render and widget is not self.progress_bar:
                widget.setEnabled(enabled)
        if enabled:
            self.btn_render.setText("🎬 Start Rendering to MP4")
            self.btn_render.setStyleSheet("font-weight: bold; background-color: #d40000; border-color: #ff0000;")
        else:
            self.btn_render.setText("❌ Cancel Render")
            self.btn_render.setStyleSheet("font-weight: bold; background-color: #444; border-color: #888;")

    def start_rendering(self):
        if self.is_rendering:
            self.render_timer.stop()
            if self.render_writer:
                self.render_writer.close()
                self.render_writer = None
            self.is_rendering = False
            self.set_ui_enabled(True)
            self.progress_bar.setValue(0)
            self.timer.start(16)
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save MP4 Video", "4d_animation.mp4", "MP4 Files (*.mp4)")
        if not filename:
            return

        self.is_rendering = True
        self.set_ui_enabled(False)
        self.timer.stop()
        
        fps = self.spin_fps.value()
        duration = self.spin_duration.value()
        self.render_total_frames = fps * duration
        self.render_frame_count = 0
        
        try:
            self.render_writer = imageio.get_writer(filename, fps=fps, quality=10, macro_block_size=1, codec='libx264')
        except Exception as e:
            QMessageBox.critical(self, "Renderer Error", f"Failed to initialize video writer:\n{e}")
            self.set_ui_enabled(True)
            self.is_rendering = False
            self.timer.start(16)
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing...")
        self.render_timer.start(1)

    def render_frame(self):
        if self.render_frame_count < self.render_total_frames:
            dt = 1.0 / self.spin_fps.value()
            self.canvas.step_animation(dt=dt)
            self.canvas.draw()
            
            buf = np.asarray(self.canvas.renderer.buffer_rgba())
            buf = buf[..., :3]
            
            self.render_writer.append_data(buf)
            
            self.render_frame_count += 1
            progress = int((self.render_frame_count / self.render_total_frames) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"Rendering: {progress}%")
        else:
            self.render_timer.stop()
            if self.render_writer:
                self.render_writer.close()
                self.render_writer = None
            
            self.is_rendering = False
            self.set_ui_enabled(True)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Complete!")
            
            QMessageBox.information(self, "Render Complete", "Your MP4 video has been successfully created!")
            self.timer.start(16)

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #111; }
            QWidget { color: #ddd; font-size: 12px; font-family: Arial; }
            QGroupBox { 
                border: 1px solid #333; 
                border-radius: 5px; 
                margin-top: 10px; 
                padding-top: 10px;
                font-weight: bold;
                color: #00d2ff;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 3px; 
            }
            QPushButton { 
                background-color: #2a2a2a; 
                border: 1px solid #444; 
                padding: 8px; 
                border-radius: 4px; 
            }
            QPushButton:hover { background-color: #3a3a3a; border-color: #00d2ff; }
            QPushButton:disabled { color: #555; background-color: #222; }
            QSlider::groove:horizontal { background: #2a2a2a; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #00d2ff; width: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider:disabled::handle:horizontal { background: #555; }
            QCheckBox { padding: 5px; }
            QCheckBox:disabled { color: #555; }
            QComboBox { background-color: #2a2a2a; border: 1px solid #444; padding: 5px; border-radius: 4px; }
            QComboBox:disabled { color: #555; background-color: #222; }
            QSpinBox { background-color: #2a2a2a; border: 1px solid #444; padding: 3px; border-radius: 4px; }
            QSpinBox:disabled { color: #555; background-color: #222; }
            QProgressBar { 
                border: 1px solid #444; 
                border-radius: 4px; 
                text-align: center; 
                background-color: #2a2a2a;
                color: white;
            }
            QProgressBar::chunk { background-color: #00d2ff; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())