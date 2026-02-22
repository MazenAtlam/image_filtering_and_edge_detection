import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSlider, QSpinBox, QTabWidget, QGroupBox, QFileDialog,
                             QScrollArea, QSplitter, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QFont, QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==========================================
# --- STYLING CONSTANTS ---
# ==========================================

class AppStyle:
    # --- DARK THEME PALETTE ---
    DARK_STYLE = """
        QMainWindow { background-color: #1e1e1e; }
        QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 14px; }

        /* Tabs */
        QTabWidget::pane { border: 1px solid #3d3d3d; background: #252526; border-radius: 5px; }
        QTabBar::tab { background: #2d2d2d; color: #aaa; padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
        QTabBar::tab:selected { background: #3e3e42; color: #fff; border-bottom: 2px solid #007acc; }

        /* Groups / Cards */
        QGroupBox { 
            border: 1px solid #3e3e42; 
            border-radius: 8px; 
            margin-top: 20px; 
            background-color: #252526; 
            font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #007acc; }

        /* Controls */
        QPushButton { 
            background-color: #007acc; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            padding: 8px 16px; 
            font-weight: bold;
        }
        QPushButton:hover { background-color: #005f9e; }
        QPushButton:pressed { background-color: #003e66; }
        QPushButton#SecondaryBtn { background-color: #3e3e42; color: #eee; }
        QPushButton#SecondaryBtn:hover { background-color: #4e4e52; }

        QComboBox, QSpinBox { 
            background-color: #333337; 
            border: 1px solid #3e3e42; 
            border-radius: 4px; 
            padding: 5px; 
            color: white; 
        }

        /* Image Display Area */
        QLabel#ImageDisplay { 
            border: 2px dashed #3e3e42; 
            background-color: #1e1e1e; 
            color: #555;
            font-size: 16px;
        }
    """

    # --- LIGHT THEME PALETTE ---
    LIGHT_STYLE = """
        QMainWindow { background-color: #f0f2f5; }
        QWidget { color: #333; font-family: 'Segoe UI', sans-serif; font-size: 14px; }

        /* Tabs */
        QTabWidget::pane { border: 1px solid #ccc; background: #fff; border-radius: 5px; }
        QTabBar::tab { background: #e0e0e0; color: #555; padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
        QTabBar::tab:selected { background: #fff; color: #000; border-bottom: 2px solid #007acc; }

        /* Groups / Cards */
        QGroupBox { 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            margin-top: 20px; 
            background-color: #ffffff; 
            font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #007acc; }

        /* Controls */
        QPushButton { 
            background-color: #007acc; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            padding: 8px 16px; 
            font-weight: bold;
        }
        QPushButton:hover { background-color: #0062a3; }
        QPushButton#SecondaryBtn { background-color: #e0e0e0; color: #333; }
        QPushButton#SecondaryBtn:hover { background-color: #d0d0d0; }

        QComboBox, QSpinBox { 
            background-color: #fff; 
            border: 1px solid #ccc; 
            border-radius: 4px; 
            padding: 5px; 
            color: #333; 
        }

        /* Image Display Area */
        QLabel#ImageDisplay { 
            border: 2px dashed #ccc; 
            background-color: #fafafa; 
            color: #aaa;
            font-size: 16px;
        }
    """


# ==========================================
# --- MAIN APPLICATION ---
# ==========================================

class ComputerVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Toolkit Pro")
        self.resize(1300, 850)

        # State
        self.is_dark_mode = True

        # UI Initialization
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        # Top Toolbar for Theme Toggle
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(10, 5, 10, 5)

        self.theme_btn = QPushButton("Switch to Light Mode")
        self.theme_btn.setObjectName("SecondaryBtn")
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self.toggle_theme)

        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.theme_btn)
        toolbar.setLayout(toolbar_layout)

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(toolbar)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.init_main_tab()
        self.init_hybrid_tab()

    def init_main_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Splitter to allow resizing between Controls and View
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT: SCROLLABLE CONTROLS ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(20)

        # 1. File Operations
        file_group = self.create_group_box("File & History")
        l = QVBoxLayout()
        btn_load = QPushButton("📂 Load Image")
        btn_load.clicked.connect(self.load_image)
        btn_undo = QPushButton("↩ Undo Last Action")
        btn_undo.setObjectName("SecondaryBtn")
        l.addWidget(btn_load)
        l.addWidget(btn_undo)
        file_group.setLayout(l)
        controls_layout.addWidget(file_group)

        # 2. Noise (Task 1: Additive Noise)
        noise_group = self.create_group_box("Add Noise")
        l = QVBoxLayout()
        self.combo_noise = QComboBox()
        #  Uniform, Gaussian, Salt & Pepper
        self.combo_noise.addItems(["Uniform Noise", "Gaussian Noise", "Salt & Pepper"])
        self.slider_noise = QSlider(Qt.Orientation.Horizontal)
        self.slider_noise.setRange(0, 100)
        btn_noise = QPushButton("Apply Noise")
        l.addWidget(QLabel("Type:"))
        l.addWidget(self.combo_noise)
        l.addWidget(QLabel("Intensity:"))
        l.addWidget(self.slider_noise)
        l.addWidget(btn_noise)
        noise_group.setLayout(l)
        controls_layout.addWidget(noise_group)

        # 3. Spatial Filters (Task 2: LPF)
        filter_group = self.create_group_box("Spatial Filters (Smoothing)")
        l = QVBoxLayout()
        self.combo_filter = QComboBox()
        #  Average, Gaussian, Median
        self.combo_filter.addItems(["Average Filter", "Gaussian Filter", "Median Filter"])
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(3, 31)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(3)
        btn_filter = QPushButton("Apply Filter")
        l.addWidget(QLabel("Filter Type:"))
        l.addWidget(self.combo_filter)
        l.addWidget(QLabel("Kernel Size (Odd):"))
        l.addWidget(self.spin_kernel)
        l.addWidget(btn_filter)
        filter_group.setLayout(l)
        controls_layout.addWidget(filter_group)

        # 4. Edge Detection (Task 3: Edges)
        edge_group = self.create_group_box("Edge Detection")
        l = QVBoxLayout()
        self.combo_edge = QComboBox()
        #  Sobel, Roberts, Prewitt, Canny
        self.combo_edge.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        btn_edge = QPushButton("Detect Edges")
        l.addWidget(QLabel("Method:"))
        l.addWidget(self.combo_edge)
        l.addWidget(btn_edge)
        edge_group.setLayout(l)
        controls_layout.addWidget(edge_group)

        # 5. Frequency Domain
        freq_group = self.create_group_box("Frequency Domain")
        l = QVBoxLayout()
        self.combo_freq = QComboBox()
        #  High pass and low pass
        self.combo_freq.addItems(["Low Pass (Blur)", "High Pass (Sharpen)"])
        btn_freq = QPushButton("Apply FFT Filter")
        l.addWidget(self.combo_freq)
        l.addWidget(btn_freq)
        freq_group.setLayout(l)
        controls_layout.addWidget(freq_group)

        # 6. Global Ops
        ops_group = self.create_group_box("Enhancement")
        l = QVBoxLayout()
        # [cite: 302, 303] Equalize, Normalize
        l.addWidget(QPushButton("Equalize Histogram"))
        l.addWidget(QPushButton("Normalize Image"))
        ops_group.setLayout(l)
        controls_layout.addWidget(ops_group)

        controls_layout.addStretch()  # Push everything up
        scroll_area.setWidget(controls_widget)

        # --- RIGHT: DISPLAY AREA ---
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)

        # Image Viewports
        img_container = QWidget()
        img_layout = QHBoxLayout(img_container)

        self.lbl_orig = QLabel("Original\n(Drop Image Here)")
        self.lbl_orig.setObjectName("ImageDisplay")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_proc = QLabel("Processed\nResult")
        self.lbl_proc.setObjectName("ImageDisplay")
        self.lbl_proc.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_layout.addWidget(self.lbl_orig)
        img_layout.addWidget(self.lbl_proc)

        # Histogram Canvas [cite: 306] (R, G, B, CDF)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)

        display_layout.addWidget(img_container, stretch=2)
        display_layout.addWidget(self.canvas, stretch=1)

        # Add to Splitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(display_widget)
        splitter.setSizes([300, 900])  # Initial widths

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Image Processor")

    def init_hybrid_tab(self):
        #  Task 10 - Hybrid Images
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Controls Side
        controls = QWidget()
        controls.setFixedWidth(300)
        c_layout = QVBoxLayout(controls)

        grp_a = self.create_group_box("Image A (Low Pass)")
        l_a = QVBoxLayout()
        l_a.addWidget(QPushButton("Load Image A"))
        l_a.addWidget(QLabel("Cutoff Frequency:"))
        l_a.addWidget(QSlider(Qt.Orientation.Horizontal))
        grp_a.setLayout(l_a)

        grp_b = self.create_group_box("Image B (High Pass)")
        l_b = QVBoxLayout()
        l_b.addWidget(QPushButton("Load Image B"))
        l_b.addWidget(QLabel("Cutoff Frequency:"))
        l_b.addWidget(QSlider(Qt.Orientation.Horizontal))
        grp_b.setLayout(l_b)

        btn_mix = QPushButton("✨ Make Hybrid")
        btn_mix.setMinimumHeight(50)

        c_layout.addWidget(grp_a)
        c_layout.addWidget(grp_b)
        c_layout.addWidget(btn_mix)
        c_layout.addStretch()

        # Display Side
        display = QWidget()
        d_layout = QVBoxLayout(display)

        # Top: Inputs
        top_row = QHBoxLayout()
        self.lbl_hybrid_a = QLabel("Img A");
        self.lbl_hybrid_a.setObjectName("ImageDisplay")
        self.lbl_hybrid_b = QLabel("Img B");
        self.lbl_hybrid_b.setObjectName("ImageDisplay")
        self.lbl_hybrid_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hybrid_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(self.lbl_hybrid_a)
        top_row.addWidget(self.lbl_hybrid_b)

        # Bottom: Result
        self.lbl_hybrid_res = QLabel("Hybrid Result");
        self.lbl_hybrid_res.setObjectName("ImageDisplay")
        self.lbl_hybrid_res.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hybrid_res.setStyleSheet("border: 2px solid #007acc;")

        d_layout.addLayout(top_row)
        d_layout.addWidget(self.lbl_hybrid_res, stretch=2)

        layout.addWidget(controls)
        layout.addWidget(display)

        self.tabs.addTab(tab, "Hybrid Lab")

    # ==========================================
    # --- HELPER FUNCTIONS ---
    # ==========================================

    def create_group_box(self, title):
        group = QGroupBox(title)
        return group

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet(AppStyle.DARK_STYLE)
            self.theme_btn.setText("☀️ Light Mode")
            # Update Matplotlib colors for Dark Mode
            self.figure.patch.set_facecolor('#1e1e1e')
            self.figure.gca().set_facecolor('#252526')
            self.figure.gca().tick_params(colors='white')
            self.figure.gca().xaxis.label.set_color('white')
            self.figure.gca().yaxis.label.set_color('white')
            self.figure.gca().title.set_color('white')
        else:
            self.setStyleSheet(AppStyle.LIGHT_STYLE)
            self.theme_btn.setText("🌙 Dark Mode")
            # Update Matplotlib colors for Light Mode
            self.figure.patch.set_facecolor('#f0f2f5')
            self.figure.gca().set_facecolor('white')
            self.figure.gca().tick_params(colors='black')
            self.figure.gca().xaxis.label.set_color('black')
            self.figure.gca().yaxis.label.set_color('black')
            self.figure.gca().title.set_color('black')

        self.canvas.draw()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg)")
        if file_name:
            print(f"Loading {file_name}")
            # Backend hook: Display image on self.lbl_orig


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Fusion style allows for better custom coloring
    window = ComputerVisionApp()
    window.show()
    sys.exit(app.exec())