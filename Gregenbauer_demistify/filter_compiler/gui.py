"""
Gegenbauer Filter Compiler - Linux GUI Workbench
================================================
Graphical desktop application for interactive DSP filter synthesis,
visualization, C/C++ header export, and live WAV audio filter testing.
"""

import os
import sys
import shutil
import tempfile
import threading
import queue
import traceback
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy import signal

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dataclasses import dataclass
from filter_compiler.compiler import (
    GegenbauerFilterCompiler,
    FilterSpec,
    FilterResult,
    QuantizedTaps,
    PrecisionType
)


def generate_biorthogonal_header(pair_dict: dict, order_h0: int, order_g0: int, cutoff: float, lam: float, sr: int) -> str:
    h0 = pair_dict["H0"]
    h1 = pair_dict["H1"]
    g0 = pair_dict["G0"]
    g1 = pair_dict["G1"]
    p_res = pair_dict["product_residual"]
    pr_err = pair_dict["pr_error"]

    lines = [
        "// Auto-generated Gegenbauer Biorthogonal Filter Bank Coefficients Header",
        f"// Target Specification: BIORTHOGONAL | Order H0 = {order_h0} | Order G0 = {order_g0} | Cutoff = {cutoff} fs",
        f"// Performance: Product Residual = {p_res:.6e} | PR Error = {pr_err:.6e}",
        "",
        "#ifndef GEGENBAUER_BIORTHOGONAL_FILTER_COEFFS_H",
        "#define GEGENBAUER_BIORTHOGONAL_FILTER_COEFFS_H",
        "",
        "#include <stdint.h>",
        f"#define GEG_ORDER_H0 {order_h0}",
        f"#define GEG_ORDER_G0 {order_g0}",
        f"#define GEG_CUTOFF {cutoff}f",
        f"#define GEG_LAMBDA {lam}f",
        f"#define GEG_PRODUCT_RESIDUAL {p_res}f",
        f"#define GEG_PR_ERROR {pr_err}f",
        f"#define GEG_SAMPLING_RATE {sr}",
        "",
        f"static const double h0_geg_float64[{len(h0.float64_taps)}] = {{ " + ", ".join(f"{val:.12e}" for val in h0.float64_taps) + " };",
        f"static const double h1_geg_float64[{len(h1.float64_taps)}] = {{ " + ", ".join(f"{val:.12e}" for val in h1.float64_taps) + " };",
        f"static const double g0_geg_float64[{len(g0.float64_taps)}] = {{ " + ", ".join(f"{val:.12e}" for val in g0.float64_taps) + " };",
        f"static const double g1_geg_float64[{len(g1.float64_taps)}] = {{ " + ", ".join(f"{val:.12e}" for val in g1.float64_taps) + " };",
        "",
        f"static const int16_t h0_geg_q15[{len(h0.q15_taps)}] = {{ " + ", ".join(str(int(v)) for v in h0.q15_taps) + " };",
        f"static const int16_t g0_geg_q15[{len(g0.q15_taps)}] = {{ " + ", ".join(str(int(v)) for v in g0.q15_taps) + " };",
        "",
        "#endif // GEGENBAUER_BIORTHOGONAL_FILTER_COEFFS_H"
    ]
    return "\n".join(lines)


@dataclass
class BiorthogonalResult:
    order_h0: int
    order_g0: int
    cutoff: float
    sampling_rate: int
    lam: float
    pair_dict: dict
    freq_grid: np.ndarray
    H0_response: np.ndarray
    H1_response: np.ndarray
    G0_response: np.ndarray
    G1_response: np.ndarray
    P_response: np.ndarray
    header_code: str

    def summary(self) -> str:
        lines = [
            "=== Gegenbauer Biorthogonal Filter Bank Execution Summary ===",
            f"Filter Type: BIORTHOGONAL | Order H0: {self.order_h0} | Order G0: {self.order_g0} | Cutoff: {self.cutoff} fs",
            f"Gegenbauer Lambda: {self.lam:.4f} | Sampling Rate: {self.sampling_rate} Hz",
            f"Half-Band Product Residual: {self.pair_dict['product_residual']:.6e}",
            f"Polyphase Perfect Reconstruction Error: {self.pair_dict['pr_error']:.6e}",
            "Filter Bank Taps: H0 (Analysis LP), H1 (Analysis HP), G0 (Synthesis LP), G1 (Synthesis HP)"
        ]
        return "\n".join(lines)
from filter_compiler.audio_processor import (
    read_wav,
    write_wav,
    generate_chirp,
    generate_noise,
    generate_multitone,
    apply_filter,
    apply_qmf_filtering,
    AudioPlayer
)


class GegenbauerFilterGUI(tk.Tk):
    """Main Application Window for Gegenbauer Filter Compiler & Live Audio Test Workbench."""

    def __init__(self):
        super().__init__()
        self.title("Gegenbauer Filter Compiler & Audio Test Workbench")
        self.geometry("1280x850")
        self.minsize(1000, 700)

        self.current_result: Optional[FilterResult] = None
        self.audio_player = AudioPlayer()
        self.temp_dir = tempfile.mkdtemp(prefix="geg_filter_gui_")

        self.loaded_audio_sr: int = 44100
        self.loaded_audio_data: Optional[np.ndarray] = None
        self.filtered_audio_data: Optional[np.ndarray] = None
        self.original_wav_path: str = os.path.join(self.temp_dir, "original.wav")
        self.filtered_wav_path: str = os.path.join(self.temp_dir, "filtered.wav")

        self._create_styles()
        self._build_layout()
        self._generate_default_audio("chirp")
        self._on_compile()

    def _create_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

    def _build_layout(self):
        # Top Title Bar
        title_frame = ttk.Frame(self, padding=10)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        title_label = ttk.Label(
            title_frame,
            text="Gegenbauer DSP Filter Compiler & Live Audio Workbench",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = ttk.Label(
            title_frame,
            text="VIII-Layer Theoretical Framework | Live WAV Testing",
            font=("Helvetica", 10, "italic")
        )
        subtitle_label.pack(side=tk.RIGHT)

        # Main Paned Window (Left Controls, Right Notebook)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Control Panel
        left_frame = ttk.Frame(paned, padding=10, width=320)
        paned.add(left_frame, weight=1)

        # Right Display Area (Notebook Tabs)
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame, weight=3)

        self._build_control_panel(left_frame)
        self._build_display_notebook(right_frame)

    def _build_control_panel(self, parent):
        # Create a scrollable canvas for the control panel to fit all parameters comfortably
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_content = ttk.Frame(canvas, padding=5)

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        spec_group = ttk.LabelFrame(scroll_content, text="Filter Specifications", padding=10)
        spec_group.pack(fill=tk.X, pady=5)
        spec_group.columnconfigure(1, weight=1)

        # Filter Kind
        ttk.Label(spec_group, text="Filter Type:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.kind_var = tk.StringVar(value="lowpass")
        kind_cb = ttk.Combobox(spec_group, textvariable=self.kind_var, values=["lowpass", "highpass", "bandpass", "qmf", "asymmetric_qmf", "biorthogonal"], state="readonly")
        kind_cb.grid(row=0, column=1, sticky=tk.EW, pady=3)
        kind_cb.bind("<<ComboboxSelected>>", self._on_kind_changed)

        # Order N / H0
        ttk.Label(spec_group, text="Order N / H0:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.order_var = tk.IntVar(value=63)
        ttk.Spinbox(spec_group, from_=3, to=512, textvariable=self.order_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=3)

        # Order G0 (Biorthogonal)
        ttk.Label(spec_group, text="Order G0 (Biorthg):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.order_g0_var = tk.IntVar(value=6)
        ttk.Spinbox(spec_group, from_=2, to=512, textvariable=self.order_g0_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=3)

        # Cutoff Frequency
        ttk.Label(spec_group, text="Cutoff (Normalized):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.cutoff_var = tk.DoubleVar(value=0.25)
        ttk.Entry(spec_group, textvariable=self.cutoff_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=3)

        # Sampling Rate
        ttk.Label(spec_group, text="Sampling Rate (Hz):").grid(row=4, column=0, sticky=tk.W, pady=3)
        self.sr_var = tk.IntVar(value=44100)
        ttk.Entry(spec_group, textvariable=self.sr_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=3)

        # Passband Edge wp
        ttk.Label(spec_group, text="Passband Edge wp (f/fs):").grid(row=5, column=0, sticky=tk.W, pady=3)
        self.wp_var = tk.StringVar(value="")
        ttk.Entry(spec_group, textvariable=self.wp_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=3)

        # Stopband Edge ws
        ttk.Label(spec_group, text="Stopband Edge ws (f/fs):").grid(row=6, column=0, sticky=tk.W, pady=3)
        self.ws_var = tk.StringVar(value="")
        ttk.Entry(spec_group, textvariable=self.ws_var, width=10).grid(row=6, column=1, sticky=tk.W, pady=3)

        # Bandpass Upper Passband Edge wp2
        ttk.Label(spec_group, text="Bandpass wp2 (f/fs):").grid(row=7, column=0, sticky=tk.W, pady=3)
        self.wp2_var = tk.StringVar(value="")
        ttk.Entry(spec_group, textvariable=self.wp2_var, width=10).grid(row=7, column=1, sticky=tk.W, pady=3)

        # Bandpass Upper Stopband Edge ws2
        ttk.Label(spec_group, text="Bandpass ws2 (f/fs):").grid(row=8, column=0, sticky=tk.W, pady=3)
        self.ws2_var = tk.StringVar(value="")
        ttk.Entry(spec_group, textvariable=self.ws2_var, width=10).grid(row=8, column=1, sticky=tk.W, pady=3)

        # Passband Ripple dB
        ttk.Label(spec_group, text="Passband Ripple Target (dB):").grid(row=9, column=0, sticky=tk.W, pady=3)
        self.ripple_var = tk.DoubleVar(value=0.1)
        ttk.Entry(spec_group, textvariable=self.ripple_var, width=10).grid(row=9, column=1, sticky=tk.W, pady=3)

        # Stopband Attenuation dB
        ttk.Label(spec_group, text="Stopband Atten Target (dB):").grid(row=10, column=0, sticky=tk.W, pady=3)
        self.atten_var = tk.DoubleVar(value=60.0)
        ttk.Entry(spec_group, textvariable=self.atten_var, width=10).grid(row=10, column=1, sticky=tk.W, pady=3)

        # Algorithmic Parameters Group
        alg_group = ttk.LabelFrame(scroll_content, text="Gegenbauer Framework Parameters", padding=10)
        alg_group.pack(fill=tk.X, pady=5)
        alg_group.columnconfigure(1, weight=1)

        # Lambda
        ttk.Label(alg_group, text="Lambda (λ > -0.5):").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.lambda_var = tk.DoubleVar(value=1.25)
        ttk.Entry(alg_group, textvariable=self.lambda_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=3)

        # Basis Terms
        ttk.Label(alg_group, text="Basis Terms K (Auto=blank):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.basis_terms_var = tk.StringVar(value="")
        ttk.Entry(alg_group, textvariable=self.basis_terms_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=3)

        # Basis Type
        ttk.Label(alg_group, text="Basis Type:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.basis_type_var = tk.StringVar(value="normalized")
        ttk.Combobox(alg_group, textvariable=self.basis_type_var, values=["normalized", "unnormalized"], state="readonly").grid(row=2, column=1, sticky=tk.EW, pady=3)

        # Solver
        ttk.Label(alg_group, text="Solver Algorithm:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.solver_var = tk.StringVar(value="spectral_regularized")
        ttk.Combobox(alg_group, textvariable=self.solver_var, values=["spectral_regularized", "quadrature", "least_squares"], state="readonly").grid(row=3, column=1, sticky=tk.EW, pady=3)

        # Regularization mu
        ttk.Label(alg_group, text="Reg Weight (μ):").grid(row=4, column=0, sticky=tk.W, pady=3)
        self.mu_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(alg_group, textvariable=self.mu_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=3)

        # Reg Power
        ttk.Label(alg_group, text="Reg Power p:").grid(row=5, column=0, sticky=tk.W, pady=3)
        self.reg_power_var = tk.IntVar(value=1)
        ttk.Spinbox(alg_group, from_=1, to=10, textvariable=self.reg_power_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=3)

        # Asymptotic Mode
        ttk.Label(alg_group, text="Asymptotic Mode:").grid(row=6, column=0, sticky=tk.W, pady=3)
        self.asymp_mode_var = tk.StringVar(value="auto")
        ttk.Combobox(alg_group, textvariable=self.asymp_mode_var, values=["auto", "composite", "bessel", "wkb", "none"], state="readonly").grid(row=6, column=1, sticky=tk.EW, pady=3)

        # Grid Samples
        ttk.Label(alg_group, text="Grid Samples L:").grid(row=7, column=0, sticky=tk.W, pady=3)
        self.grid_samples_var = tk.IntVar(value=2048)
        ttk.Entry(alg_group, textvariable=self.grid_samples_var, width=10).grid(row=7, column=1, sticky=tk.W, pady=3)

        # Precision
        ttk.Label(alg_group, text="Precision Context:").grid(row=8, column=0, sticky=tk.W, pady=3)
        self.precision_var = tk.StringVar(value="FLOAT64")
        ttk.Combobox(alg_group, textvariable=self.precision_var, values=["FLOAT32", "FLOAT64", "LONGDOUBLE"], state="readonly").grid(row=8, column=1, sticky=tk.EW, pady=3)

        # Buttons Frame
        btn_frame = ttk.Frame(scroll_content, padding=5)
        btn_frame.pack(fill=tk.X, pady=10)

        self.compile_btn = ttk.Button(btn_frame, text="⚡ Compile Filter", command=self._on_compile)
        self.compile_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.pareto_btn = ttk.Button(btn_frame, text="🔍 Pareto Search", command=self._on_pareto_search)
        self.pareto_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        # Summary Metrics Text Box
        summary_group = ttk.LabelFrame(scroll_content, text="Compilation Certification Summary", padding=5)
        summary_group.pack(fill=tk.BOTH, expand=True, pady=5)

        self.summary_text = tk.Text(summary_group, wrap=tk.WORD, height=12, font=("Courier", 9), state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True)


    def _build_display_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Frequency & Phase Response
        self.tab_freq = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_freq, text="Frequency Response")

        self.fig_freq = Figure(figsize=(7, 5), dpi=100)
        self.ax_freq = self.fig_freq.add_subplot(211)
        self.ax_pass = self.fig_freq.add_subplot(212)
        self.fig_freq.tight_layout(pad=3.0)

        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, master=self.tab_freq)
        self.canvas_freq.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draggable_lines = {}
        self._dragging_param = None

        self.canvas_freq.mpl_connect('button_press_event', self._on_freq_click)
        self.canvas_freq.mpl_connect('motion_notify_event', self._on_freq_drag)
        self.canvas_freq.mpl_connect('button_release_event', self._on_freq_release)

        # Tab 2: Taps & Quantization Noise
        self.tab_taps = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_taps, text="Taps & Quantization")

        self.fig_taps = Figure(figsize=(7, 5), dpi=100)
        self.ax_stem = self.fig_taps.add_subplot(211)
        self.ax_quant = self.fig_taps.add_subplot(212)
        self.fig_taps.tight_layout(pad=3.0)

        self.canvas_taps = FigureCanvasTkAgg(self.fig_taps, master=self.tab_taps)
        self.canvas_taps.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3: Live Audio Filter Testing
        self.tab_audio = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_audio, text="🎵 Live Audio Testing")
        self._build_audio_testing_tab(self.tab_audio)

        # Tab 4: C/C++ Header Export
        self.tab_header = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_header, text="C/C++ Header Export")
        self._build_header_export_tab(self.tab_header)

    def _build_audio_testing_tab(self, parent):
        controls_frame = ttk.LabelFrame(parent, text="Audio Source & Live Filter Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=5, padx=5)

        # Source Selection Buttons
        ttk.Label(controls_frame, text="Generate Test Signal:").grid(row=0, column=0, sticky=tk.W, pady=3)
        gen_btn_frame = ttk.Frame(controls_frame)
        gen_btn_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W)

        ttk.Button(gen_btn_frame, text="Chirp Sweep", command=lambda: self._generate_default_audio("chirp")).pack(side=tk.LEFT, padx=2)
        ttk.Button(gen_btn_frame, text="White Noise", command=lambda: self._generate_default_audio("white")).pack(side=tk.LEFT, padx=2)
        ttk.Button(gen_btn_frame, text="Pink Noise", command=lambda: self._generate_default_audio("pink")).pack(side=tk.LEFT, padx=2)
        ttk.Button(gen_btn_frame, text="Multitone", command=lambda: self._generate_default_audio("multitone")).pack(side=tk.LEFT, padx=2)
        ttk.Button(gen_btn_frame, text="📁 Load WAV File...", command=self._on_load_wav_file).pack(side=tk.LEFT, padx=5)

        # QMF Mixing Mode Controls Frame
        qmf_frame = ttk.Frame(controls_frame, padding=2)
        qmf_frame.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)

        ttk.Label(qmf_frame, text="QMF Test / Subband Mixing Mode:").pack(side=tk.LEFT, padx=2)
        self.qmf_mode_var = tk.StringVar(value="H0 (Lowpass)")
        self.qmf_mode_cb = ttk.Combobox(
            qmf_frame,
            textvariable=self.qmf_mode_var,
            values=[
                "H0 (Lowpass)",
                "H1 (Highpass)",
                "H0 + H1 (Reconstruction Sum)",
                "H0 - H1 (Subband Difference)",
                "Stereo Split (L: H0, R: H1)"
            ],
            state="readonly",
            width=30
        )
        self.qmf_mode_cb.pack(side=tk.LEFT, padx=5)
        self.qmf_mode_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_filter_to_current_audio())

        # Playback Controls
        play_frame = ttk.Frame(controls_frame, padding=5)
        play_frame.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=5)

        ttk.Button(play_frame, text="▶ Play Original Audio", command=self._on_play_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(play_frame, text="🔊 Play Filtered Audio", command=self._on_play_filtered).pack(side=tk.LEFT, padx=5)
        ttk.Button(play_frame, text="⏹ Stop Playback", command=self._on_stop_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(play_frame, text="💾 Export Filtered WAV...", command=self._on_export_filtered_wav).pack(side=tk.RIGHT, padx=5)

        # Audio Waveform & Spectral Visualizer Canvas
        self.fig_audio = Figure(figsize=(7, 4), dpi=100)
        self.ax_wave = self.fig_audio.add_subplot(211)
        self.ax_spec = self.fig_audio.add_subplot(212)
        self.fig_audio.tight_layout(pad=3.0)

        self.canvas_audio = FigureCanvasTkAgg(self.fig_audio, master=parent)
        self.canvas_audio.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_header_export_tab(self, parent):
        top_bar = ttk.Frame(parent, padding=5)
        top_bar.pack(fill=tk.X)

        ttk.Button(top_bar, text="📋 Copy to Clipboard", command=self._on_copy_header).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="💾 Save Header (.h)...", command=self._on_save_header_file).pack(side=tk.LEFT, padx=5)

        self.header_text = tk.Text(parent, wrap=tk.NONE, font=("Courier", 10), state=tk.DISABLED)
        scroll_y = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.header_text.yview)
        scroll_x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.header_text.xview)
        self.header_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.header_text.pack(fill=tk.BOTH, expand=True)

    def _set_busy(self, busy: bool):
        state = tk.DISABLED if busy else tk.NORMAL
        cursor = "watch" if busy else ""
        self.config(cursor=cursor)
        self.compile_btn.config(state=state)
        self.pareto_btn.config(state=state)

    def _on_kind_changed(self, event=None):
        try:
            kind = self.kind_var.get()
            order = min(512, max(3, int(self.order_var.get())))
            order_g0 = min(512, max(3, int(self.order_g0_var.get())))
            if kind == "highpass" and order % 2 == 0:
                self.order_var.set(min(512, order + 1))
            elif kind in ("qmf", "asymmetric_qmf") and order % 2 != 0:
                self.order_var.set(min(512, order + 1 if order > 3 else 4))
            elif kind == "biorthogonal":
                deg_sum = (order - 1) + (order_g0 - 1)
                if deg_sum % 4 != 2:
                    needed_g0_deg = ((order - 1) + 2) % 4
                    # Adjust order_g0 (taps_g0) so (order-1) + (order_g0-1) == 2 mod 4
                    new_g0 = order_g0
                    while ((order - 1) + (new_g0 - 1)) % 4 != 2:
                        new_g0 += 1
                    self.order_g0_var.set(min(512, new_g0))
            else:
                self.order_var.set(order)
        except Exception as e:
            sys.stderr.write(f"Kind change validation error: {e}\n")

    def _read_spec_and_compiler_params(self):
        kind = self.kind_var.get()
        cutoff = float(self.cutoff_var.get())
        sampling_rate = int(self.sr_var.get())

        lam = float(self.lambda_var.get())
        basis_terms = int(self.basis_terms_var.get()) if self.basis_terms_var.get().strip() else None
        basis_type = self.basis_type_var.get()
        solver = self.solver_var.get()
        mu_reg = float(self.mu_var.get())
        reg_power = int(self.reg_power_var.get())
        asymptotic_mode = self.asymp_mode_var.get()
        grid_samples = int(self.grid_samples_var.get())

        prec_str = self.precision_var.get().upper()
        if prec_str == "FLOAT32":
            precision = PrecisionType.FLOAT32
        elif prec_str == "LONGDOUBLE":
            precision = PrecisionType.LONGDOUBLE
        else:
            precision = PrecisionType.FLOAT64

        compiler_kwargs = dict(
            lam=lam,
            basis_terms=basis_terms,
            basis_type=basis_type,
            solver=solver,
            mu_reg=mu_reg,
            reg_power=reg_power,
            asymptotic_mode=asymptotic_mode,
            grid_samples=grid_samples,
            precision=precision
        )

        if kind == "biorthogonal":
            taps_h0 = int(self.order_var.get())
            taps_g0 = int(self.order_g0_var.get())
            deg_sum = (taps_h0 - 1) + (taps_g0 - 1)
            if deg_sum % 4 != 2:
                raise ValueError(f"Sum of H0 and G0 degrees ((taps_h0 - 1) + (taps_g0 - 1)) must be 2 mod 4 for Biorthogonal pairs. Got degrees {taps_h0 - 1} + {taps_g0 - 1} = {deg_sum}.")
            return "biorthogonal", taps_h0, taps_g0, cutoff, sampling_rate, compiler_kwargs

        order = min(512, max(3, int(self.order_var.get())))
        wp = float(self.wp_var.get()) if self.wp_var.get().strip() else None
        ws = float(self.ws_var.get()) if self.ws_var.get().strip() else None
        wp2 = float(self.wp2_var.get()) if self.wp2_var.get().strip() else None
        ws2 = float(self.ws2_var.get()) if self.ws2_var.get().strip() else None

        passband_ripple_db = float(self.ripple_var.get())
        stopband_atten_db = float(self.atten_var.get())

        spec = FilterSpec(
            kind=kind,
            order=order,
            cutoff=cutoff,
            wp=wp,
            ws=ws,
            wp2=wp2,
            ws2=ws2,
            sampling_rate=sampling_rate,
            passband_ripple_db=passband_ripple_db,
            stopband_atten_db=stopband_atten_db
        )

        return spec, compiler_kwargs

    def _on_compile(self):
        try:
            params = self._read_spec_and_compiler_params()
        except Exception as e:
            messagebox.showerror("Invalid Input", f"Please check input fields:\n{e}")
            return

        self._set_busy(True)
        res_q = queue.Queue()

        def worker():
            try:
                if params[0] == "biorthogonal":
                    _, taps_h0, taps_g0, cutoff, sampling_rate, compiler_kwargs = params
                    compiler = GegenbauerFilterCompiler(**compiler_kwargs)
                    pair = compiler.compile_biorthogonal_pair(taps_h0=taps_h0, taps_g0=taps_g0, cutoff=cutoff)

                    K_fft = 4096
                    freq_grid = np.arange(K_fft // 2 + 1) / float(K_fft)
                    H0_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(pair["H0"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    H1_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(pair["H1"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    G0_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(pair["G0"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    G1_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(pair["G1"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    P_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(pair["P"], K_fft)[:K_fft // 2 + 1])))

                    hdr = generate_biorthogonal_header(pair, taps_h0, taps_g0, cutoff, compiler_kwargs["lam"], sampling_rate)

                    biorthg_res = BiorthogonalResult(
                        order_h0=taps_h0,
                        order_g0=taps_g0,
                        cutoff=cutoff,
                        sampling_rate=sampling_rate,
                        lam=compiler_kwargs["lam"],
                        pair_dict=pair,
                        freq_grid=freq_grid,
                        H0_response=H0_resp,
                        H1_response=H1_resp,
                        G0_response=G0_resp,
                        G1_response=G1_resp,
                        P_response=P_resp,
                        header_code=hdr
                    )
                    res_q.put(("ok", biorthg_res))
                else:
                    spec, compiler_kwargs = params
                    compiler = GegenbauerFilterCompiler(**compiler_kwargs)
                    res = compiler.compile(spec)
                    res_q.put(("ok", res))
            except Exception as ex:
                res_q.put(("err", ex))

        threading.Thread(target=worker, daemon=True).start()
        self.after(50, self._poll_compile_result, res_q)

    def _poll_compile_result(self, q: queue.Queue):
        try:
            status, payload = q.get_nowait()
        except queue.Empty:
            self.after(50, self._poll_compile_result, q)
            return

        self._set_busy(False)
        if status == "err":
            sys.stderr.write(f"Compilation error:\n{traceback.format_exc()}\n")
            messagebox.showerror("Compilation Error", str(payload))
            return

        self.current_result = payload
        self._update_summary_display()
        self._update_freq_plots()
        self._update_taps_plots()
        self._update_header_display()

        if self.loaded_audio_data is not None:
            self._apply_filter_to_current_audio()

    def _on_pareto_search(self):
        try:
            params = self._read_spec_and_compiler_params()
        except Exception as e:
            messagebox.showerror("Invalid Input", f"Please check input fields:\n{e}")
            return

        self._set_busy(True)
        res_q = queue.Queue()

        def worker():
            try:
                if params[0] == "biorthogonal":
                    _, taps_h0, taps_g0, cutoff, sampling_rate, compiler_kwargs = params
                    best_pair = None
                    best_score = float('inf')
                    best_lam = compiler_kwargs['lam']
                    for lam in [0.5, 1.0, 1.25, 1.5, 2.0]:
                        ck = dict(compiler_kwargs)
                        ck['lam'] = lam
                        compiler = GegenbauerFilterCompiler(**ck)
                        try:
                            pair = compiler.compile_biorthogonal_pair(taps_h0=taps_h0, taps_g0=taps_g0, cutoff=cutoff)
                            score = pair['product_residual'] * 100.0 + pair['pr_error'] * 10.0
                            if score < best_score:
                                best_score = score
                                best_pair = pair
                                best_lam = lam
                        except Exception:
                            continue

                    if best_pair is None:
                        raise RuntimeError("Biorthogonal pareto search failed to find valid roots.")

                    K_fft = 4096
                    freq_grid = np.arange(K_fft // 2 + 1) / float(K_fft)
                    H0_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(best_pair["H0"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    H1_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(best_pair["H1"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    G0_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(best_pair["G0"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    G1_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(best_pair["G1"].float64_taps, K_fft)[:K_fft // 2 + 1])))
                    P_resp = 20 * np.log10(np.maximum(1e-12, np.abs(np.fft.fft(best_pair["P"], K_fft)[:K_fft // 2 + 1])))

                    hdr = generate_biorthogonal_header(best_pair, taps_h0, taps_g0, cutoff, best_lam, sampling_rate)

                    biorthg_res = BiorthogonalResult(
                        order_h0=taps_h0,
                        order_g0=taps_g0,
                        cutoff=cutoff,
                        sampling_rate=sampling_rate,
                        lam=best_lam,
                        pair_dict=best_pair,
                        freq_grid=freq_grid,
                        H0_response=H0_resp,
                        H1_response=H1_resp,
                        G0_response=G0_resp,
                        G1_response=G1_resp,
                        P_response=P_resp,
                        header_code=hdr
                    )
                    res_q.put(("ok", biorthg_res))
                else:
                    spec, compiler_kwargs = params
                    compiler = GegenbauerFilterCompiler(**compiler_kwargs)
                    best_res = compiler.pareto_search(spec, solver=compiler_kwargs['solver'])
                    res_q.put(("ok", best_res))
            except Exception as ex:
                res_q.put(("err", ex))

        threading.Thread(target=worker, daemon=True).start()
        self.after(50, self._poll_pareto_result, res_q)

    def _poll_pareto_result(self, q: queue.Queue):
        try:
            status, payload = q.get_nowait()
        except queue.Empty:
            self.after(50, self._poll_pareto_result, q)
            return

        self._set_busy(False)
        if status == "err":
            sys.stderr.write(f"Pareto search error:\n{traceback.format_exc()}\n")
            messagebox.showerror("Pareto Search Error", str(payload))
            return

        self.current_result = payload
        if hasattr(payload, 'lam'):
            self.lambda_var.set(payload.lam)
        if hasattr(payload, 'mu_reg'):
            self.mu_var.set(payload.mu_reg)

        self._update_summary_display()
        self._update_freq_plots()
        self._update_taps_plots()
        self._update_header_display()

        if self.loaded_audio_data is not None:
            self._apply_filter_to_current_audio()

        msg_lam = payload.lam if hasattr(payload, 'lam') else 1.5
        messagebox.showinfo("Pareto Optimization", f"Found Pareto-Optimal Lambda = {msg_lam:.4f}")

    def _update_summary_display(self):
        if self.current_result is None:
            return
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, self.current_result.summary())
        self.summary_text.config(state=tk.DISABLED)

    def _update_freq_plots(self):
        if self.current_result is None:
            return

        self.ax_freq.clear()
        self.ax_pass.clear()
        self._draggable_lines.clear()

        if isinstance(self.current_result, BiorthogonalResult):
            res = self.current_result
            self.ax_freq.plot(res.freq_grid, res.H0_response, 'b-', label='H0 (Analysis LP)', linewidth=1.5)
            self.ax_freq.plot(res.freq_grid, res.H1_response, 'r--', label='H1 (Analysis HP)', linewidth=1.5)
            self.ax_freq.plot(res.freq_grid, res.G0_response, 'g-', label='G0 (Synthesis LP)', linewidth=1.5)
            self.ax_freq.plot(res.freq_grid, res.G1_response, 'm--', label='G1 (Synthesis HP)', linewidth=1.5)
            self.ax_freq.plot(res.freq_grid, res.P_response, 'k:', label='Product Filter P(z)', linewidth=1.5)

            line = self.ax_freq.axvline(res.cutoff, color='g', linestyle=':', linewidth=2, label=f'Cutoff={res.cutoff:.3f}', picker=True)
            self._draggable_lines["cutoff"] = (line, res.cutoff)

            self.ax_freq.set_title(f"Biorthogonal Filter Bank Frequency Response (H0={res.order_h0}, G0={res.order_g0}, λ={res.lam:.2f})")
            self.ax_freq.set_ylabel("Magnitude (dB)")
            self.ax_freq.set_ylim(-100, 5)
            self.ax_freq.grid(True)
            self.ax_freq.legend(loc="lower left", fontsize=8)

            # Passband product P(z) detail
            self.ax_pass.plot(res.freq_grid, res.P_response, 'k-', linewidth=1.5, label='P(z) Half-band')
            self.ax_pass.set_title(f"Half-band Product Filter P(z) Response (Residual = {res.pair_dict['product_residual']:.2e})")
            self.ax_pass.set_xlabel("Normalized Frequency (f/fs)")
            self.ax_pass.set_ylabel("Magnitude (dB)")
            self.ax_pass.set_ylim(-10, 5)
            self.ax_pass.grid(True)
            self.fig_freq.tight_layout(pad=2.0)
            self.canvas_freq.draw()
            return

        res = self.current_result
        spec = res.spec

        # Plot 1: Full Frequency Response
        self.ax_freq.plot(res.freq_grid, res.H0_response, 'b-', label='H0 (Lowpass)' if spec.kind in ('qmf', 'asymmetric_qmf') else '|H(e^{jw})|', linewidth=1.5)
        if res.H1_response is not None:
            self.ax_freq.plot(res.freq_grid, res.H1_response, 'r-', label='H1 (Highpass)', linewidth=1.5)

        # Draw interactive band limit lines depending on filter type
        if spec.kind == "bandpass":
            lines_spec = [
                ("ws", spec.ws, 'r', '--', f'ws={spec.ws:.3f}'),
                ("wp", spec.wp, 'b', '--', f'wp={spec.wp:.3f}'),
                ("wp2", spec.wp2, 'b', '-.', f'wp2={spec.wp2:.3f}'),
                ("ws2", spec.ws2, 'r', '-.', f'ws2={spec.ws2:.3f}'),
            ]
        else:
            lines_spec = [("cutoff", spec.cutoff, 'g', ':', f'Cutoff={spec.cutoff:.3f}')]
            if spec.wp is not None:
                lines_spec.append(("wp", spec.wp, 'b', '--', f'wp={spec.wp:.3f}'))
            if spec.ws is not None:
                lines_spec.append(("ws", spec.ws, 'r', '--', f'ws={spec.ws:.3f}'))

        for param_name, val, col, ls, lbl in lines_spec:
            line = self.ax_freq.axvline(val, color=col, linestyle=ls, linewidth=2, label=lbl, picker=True)
            self._draggable_lines[param_name] = (line, val)

        self.ax_freq.set_title(f"Frequency Response ({spec.kind.upper()}, N={spec.order}, λ={res.lam:.2f}) [Drag vertical lines to adjust]")
        self.ax_freq.set_ylabel("Magnitude (dB)")
        self.ax_freq.set_ylim(-100, 5)
        self.ax_freq.grid(True)
        self.ax_freq.legend(loc="lower left", fontsize=8)

        # Plot 2: Passband Detail
        if spec.kind == "highpass":
            pass_mask = res.freq_grid >= spec.wp
        elif spec.kind == "bandpass":
            pass_mask = (res.freq_grid >= spec.wp) & (res.freq_grid <= spec.wp2)
        else:
            pass_mask = res.freq_grid <= spec.wp

        if np.any(pass_mask):
            self.ax_pass.plot(res.freq_grid[pass_mask], res.H0_response[pass_mask], 'b-', linewidth=1.5)
            self.ax_pass.set_title(f"Passband Detail (Ripple = {res.passband_ripple_actual:.4f} dB)")
            self.ax_pass.set_xlabel("Normalized Frequency (f/fs)")
            self.ax_pass.set_ylabel("Magnitude (dB)")
            self.ax_pass.grid(True)

        self.fig_freq.tight_layout(pad=2.0)
        self.canvas_freq.draw()

    def _on_freq_click(self, event):
        if event.inaxes != self.ax_freq or event.button != 1 or event.xdata is None:
            return

        click_x = event.xdata
        best_param = None
        min_dist = float('inf')

        for param_name, (line, val) in self._draggable_lines.items():
            dist = abs(click_x - val)
            if dist < 0.03 and dist < min_dist:
                min_dist = dist
                best_param = param_name

        if best_param is not None:
            self._dragging_param = best_param

    def _on_freq_drag(self, event):
        if self._dragging_param is None or event.inaxes != self.ax_freq or event.xdata is None:
            return

        new_val = round(float(np.clip(event.xdata, 0.01, 0.49)), 3)
        param_name = self._dragging_param

        if param_name in self._draggable_lines:
            line, _ = self._draggable_lines[param_name]
            line.set_xdata([new_val, new_val])
            self._draggable_lines[param_name] = (line, new_val)

        # Update UI Entry Variable
        if param_name == "cutoff":
            self.cutoff_var.set(new_val)
        elif param_name == "wp":
            self.wp_var.set(str(new_val))
        elif param_name == "ws":
            self.ws_var.set(str(new_val))
        elif param_name == "wp2":
            self.wp2_var.set(str(new_val))
        elif param_name == "ws2":
            self.ws2_var.set(str(new_val))

        self.canvas_freq.draw_idle()

    def _on_freq_release(self, event):
        if self._dragging_param is not None:
            self._dragging_param = None
            self._on_compile()

    def _update_taps_plots(self):
        if self.current_result is None:
            return

        self.ax_stem.clear()
        self.ax_quant.clear()

        if isinstance(self.current_result, BiorthogonalResult):
            res = self.current_result
            h0 = res.pair_dict["H0"]
            g0 = res.pair_dict["G0"]

            self.ax_stem.stem(range(len(h0.float64_taps)), h0.float64_taps, linefmt='b-', markerfmt='bo', basefmt='r-', label='h0 taps (Analysis LP)')
            self.ax_stem.stem(range(len(g0.float64_taps)), g0.float64_taps, linefmt='g--', markerfmt='gs', basefmt='r-', label='g0 taps (Synthesis LP)')
            self.ax_stem.set_title("Biorthogonal Filter Bank Impulse Response Taps")
            self.ax_stem.set_ylabel("Amplitude")
            self.ax_stem.grid(True)
            self.ax_stem.legend(fontsize=8)

            err_h0_q15 = np.abs(h0.float64_taps - h0.q15_taps / h0.q15_scale)
            err_g0_q15 = np.abs(g0.float64_taps - g0.q15_taps / g0.q15_scale)

            self.ax_quant.semilogy(range(len(h0.float64_taps)), np.maximum(1e-16, err_h0_q15), 'r-o', label='h0 Q15 Error', markersize=4)
            self.ax_quant.semilogy(range(len(g0.float64_taps)), np.maximum(1e-16, err_g0_q15), 'm-s', label='g0 Q15 Error', markersize=4)
            self.ax_quant.set_title("Fixed-Point Quantization Noise per Tap")
            self.ax_quant.set_xlabel("Tap Index n")
            self.ax_quant.set_ylabel("Absolute Error")
            self.ax_quant.grid(True)
            self.ax_quant.legend(fontsize=8)

            self.fig_taps.tight_layout(pad=2.0)
            self.canvas_taps.draw()
            return

        res = self.current_result
        spec = res.spec

        # Stem Plot of Taps
        self.ax_stem.stem(range(spec.order), res.h0_taps.float64_taps, linefmt='b-', markerfmt='bo', basefmt='r-', label='h0 taps')
        if spec.kind in ("qmf", "asymmetric_qmf") and res.h1_taps is not None:
            self.ax_stem.stem(range(spec.order), res.h1_taps.float64_taps, linefmt='r--', markerfmt='rx', basefmt='r-', label='h1 taps')
            self.ax_stem.legend(fontsize=8)
        self.ax_stem.set_title("Impulse Response Taps h[n]")
        self.ax_stem.set_ylabel("Amplitude")
        self.ax_stem.grid(True)

        # Quantization Noise Plot
        h_float = res.h0_taps.float64_taps
        h_q15_recon = res.h0_taps.q15_taps / res.h0_taps.q15_scale
        h_q31_recon = res.h0_taps.q31_taps / res.h0_taps.q31_scale

        err_q15 = np.abs(h_float - h_q15_recon)
        err_q31 = np.abs(h_float - h_q31_recon)

        self.ax_quant.semilogy(range(spec.order), np.maximum(1e-16, err_q15), 'r-o', label='h0 Q15 Error', markersize=4)
        self.ax_quant.semilogy(range(spec.order), np.maximum(1e-16, err_q31), 'g-s', label='h0 Q31 Error', markersize=4)

        if spec.kind in ("qmf", "asymmetric_qmf") and res.h1_taps is not None:
            h1_float = res.h1_taps.float64_taps
            h1_q15_recon = res.h1_taps.q15_taps / res.h1_taps.q15_scale
            err_h1_q15 = np.abs(h1_float - h1_q15_recon)
            self.ax_quant.semilogy(range(spec.order), np.maximum(1e-16, err_h1_q15), 'm--x', label='h1 Q15 Error', markersize=4)

        self.ax_quant.set_title("Fixed-Point Quantization Noise per Tap")
        self.ax_quant.set_xlabel("Tap Index n")
        self.ax_quant.set_ylabel("Absolute Error")
        self.ax_quant.grid(True)
        self.ax_quant.legend(fontsize=8)

        self.fig_taps.tight_layout(pad=2.0)
        self.canvas_taps.draw()

    def _update_header_display(self):
        if self.current_result is None:
            return
        self.header_text.config(state=tk.NORMAL)
        self.header_text.delete("1.0", tk.END)
        self.header_text.insert(tk.END, self.current_result.header_code)
        self.header_text.config(state=tk.DISABLED)

    def _generate_default_audio(self, signal_type: str):
        sr = self.sr_var.get()
        if signal_type == "chirp":
            sr, data = generate_chirp(duration=3.0, sample_rate=sr)
        elif signal_type == "white":
            sr, data = generate_noise(duration=3.0, sample_rate=sr, noise_type="white")
        elif signal_type == "pink":
            sr, data = generate_noise(duration=3.0, sample_rate=sr, noise_type="pink")
        elif signal_type == "multitone":
            sr, data = generate_multitone(duration=3.0, sample_rate=sr)
        else:
            return

        self.loaded_audio_sr = sr
        self.loaded_audio_data = data
        write_wav(self.original_wav_path, sr, data)

        if self.current_result is not None:
            self._apply_filter_to_current_audio()

    def _on_load_wav_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Test WAV Audio File",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        try:
            sr, data = read_wav(filepath)
            self.loaded_audio_sr = sr
            self.loaded_audio_data = data
            self.sr_var.set(sr)
            write_wav(self.original_wav_path, sr, data)
            self._apply_filter_to_current_audio()
        except Exception as e:
            messagebox.showerror("Audio Load Error", str(e))

    def _apply_filter_to_current_audio(self):
        if self.loaded_audio_data is None or self.current_result is None:
            return

        if isinstance(self.current_result, BiorthogonalResult):
            self.qmf_mode_cb.config(state="readonly")
            b_res = self.current_result
            h0 = b_res.pair_dict["H0"].float64_taps
            h1 = b_res.pair_dict["H1"].float64_taps
            g0 = b_res.pair_dict["G0"].float64_taps
            g1 = b_res.pair_dict["G1"].float64_taps

            sel = self.qmf_mode_var.get()
            if "Analysis Highpass" in sel or "H1" in sel:
                self.filtered_audio_data = apply_filter(h1, self.loaded_audio_data)
            elif "Synthesis Lowpass" in sel or "G0" in sel:
                self.filtered_audio_data = apply_filter(g0, self.loaded_audio_data)
            elif "Synthesis Highpass" in sel or "G1" in sel:
                self.filtered_audio_data = apply_filter(g1, self.loaded_audio_data)
            elif "Full PR" in sel or "Reconstruction" in sel:
                sub0 = apply_filter(g0, apply_filter(h0, self.loaded_audio_data))
                sub1 = apply_filter(g1, apply_filter(h1, self.loaded_audio_data))
                self.filtered_audio_data = sub0 + sub1
            elif "Stereo" in sel:
                f_h0 = apply_filter(h0, self.loaded_audio_data)
                f_h1 = apply_filter(h1, self.loaded_audio_data)
                if self.loaded_audio_data.ndim > 1:
                    f_h0 = f_h0[:, 0]
                    f_h1 = f_h1[:, 0]
                self.filtered_audio_data = np.column_stack([f_h0, f_h1])
            else: # Analysis Lowpass H0 default
                self.filtered_audio_data = apply_filter(h0, self.loaded_audio_data)

            write_wav(self.filtered_wav_path, self.loaded_audio_sr, self.filtered_audio_data)
            self._update_audio_plots()
            return

        spec = self.current_result.spec
        h0_taps = self.current_result.h0_taps.float64_taps

        if spec.kind in ("qmf", "asymmetric_qmf") and self.current_result.h1_taps is not None:
            self.qmf_mode_cb.config(state="readonly")
            h1_taps = self.current_result.h1_taps.float64_taps
            qmf_selection = self.qmf_mode_var.get()

            if "Lowpass" in qmf_selection:
                mode_str = "lowpass"
            elif "Highpass" in qmf_selection:
                mode_str = "highpass"
            elif "Reconstruction Sum" in qmf_selection:
                mode_str = "reconstruction_sum"
            elif "Subband Difference" in qmf_selection:
                mode_str = "subband_diff"
            elif "Stereo Split" in qmf_selection:
                mode_str = "stereo_split"
            else:
                mode_str = "reconstruction_sum"

            self.filtered_audio_data = apply_qmf_filtering(
                h0_taps=h0_taps,
                h1_taps=h1_taps,
                audio_data=self.loaded_audio_data,
                mode=mode_str
            )
        else:
            self.qmf_mode_cb.config(state="disabled")
            self.filtered_audio_data = apply_filter(h0_taps, self.loaded_audio_data)

        write_wav(self.filtered_wav_path, self.loaded_audio_sr, self.filtered_audio_data)
        self._update_audio_plots()

    def _update_audio_plots(self):
        if self.loaded_audio_data is None or self.filtered_audio_data is None:
            return

        sr = self.loaded_audio_sr
        orig = self.loaded_audio_data
        filt = self.filtered_audio_data

        if orig.ndim > 1:
            orig = orig[:, 0]
        if filt.ndim > 1:
            filt = filt[:, 0]

        t = np.arange(len(orig)) / float(sr)

        self.ax_wave.clear()
        self.ax_spec.clear()

        # Waveform Display
        n_wave = min(1000, len(orig))
        self.ax_wave.plot(t[:n_wave], orig[:n_wave], 'b-', alpha=0.6, label='Original')

        if filt.ndim > 1 and filt.shape[1] == 2:
            self.ax_wave.plot(t[:n_wave], filt[:n_wave, 0], 'r-', alpha=0.8, label='Filtered Left (H0)')
            self.ax_wave.plot(t[:n_wave], filt[:n_wave, 1], 'm--', alpha=0.8, label='Filtered Right (H1)')
        else:
            self.ax_wave.plot(t[:n_wave], filt[:n_wave], 'r-', alpha=0.8, label='Filtered Output')

        self.ax_wave.set_title("Time Domain Waveform Comparison (First 1000 samples)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True)
        self.ax_wave.legend(loc="upper right", fontsize=8)

        # Spectral Display via Welch's Power Spectral Density across full signal duration
        nperseg = min(2048, len(orig))
        f_orig, psd_orig = signal.welch(orig, fs=sr, nperseg=nperseg)
        orig_db = 10 * np.log10(np.maximum(1e-12, psd_orig))

        self.ax_spec.plot(f_orig, orig_db, 'b-', alpha=0.6, label='Original PSD')

        if filt.ndim > 1 and filt.shape[1] == 2:
            f_f0, psd_f0 = signal.welch(filt[:, 0], fs=sr, nperseg=nperseg)
            f_f1, psd_f1 = signal.welch(filt[:, 1], fs=sr, nperseg=nperseg)
            self.ax_spec.plot(f_f0, 10 * np.log10(np.maximum(1e-12, psd_f0)), 'r-', alpha=0.8, label='Filtered L (H0) PSD')
            self.ax_spec.plot(f_f1, 10 * np.log10(np.maximum(1e-12, psd_f1)), 'm--', alpha=0.8, label='Filtered R (H1) PSD')
        else:
            f_filt, psd_filt = signal.welch(filt, fs=sr, nperseg=nperseg)
            filt_db = 10 * np.log10(np.maximum(1e-12, psd_filt))
            self.ax_spec.plot(f_filt, filt_db, 'r-', alpha=0.8, label='Filtered PSD')

        self.ax_spec.set_title("Welch Power Spectral Density Comparison (dB/Hz across full audio)")
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Power Density (dB/Hz)")
        self.ax_spec.grid(True)
        self.ax_spec.legend(loc="lower left", fontsize=8)

        self.fig_audio.tight_layout(pad=2.0)
        self.canvas_audio.draw()

    def _on_play_original(self):
        if not os.path.exists(self.original_wav_path):
            messagebox.showwarning("Audio Error", "No original audio loaded.")
            return
        success = self.audio_player.play(self.original_wav_path)
        if not success:
            messagebox.showwarning("Playback Warning", "System audio player (aplay/ffplay) unavailable or sound hardware missing.")

    def _on_play_filtered(self):
        if not os.path.exists(self.filtered_wav_path):
            messagebox.showwarning("Audio Error", "No filtered audio generated.")
            return
        success = self.audio_player.play(self.filtered_wav_path)
        if not success:
            messagebox.showwarning("Playback Warning", "System audio player (aplay/ffplay) unavailable or sound hardware missing.")

    def _on_stop_audio(self):
        self.audio_player.stop()

    def _on_export_filtered_wav(self):
        if self.filtered_audio_data is None:
            messagebox.showwarning("Export Warning", "No filtered audio available to export.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save Filtered WAV Audio",
            defaultextension=".wav",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if filepath:
            write_wav(filepath, self.loaded_audio_sr, self.filtered_audio_data)
            messagebox.showinfo("Export Success", f"Filtered WAV audio successfully saved to:\n{filepath}")

    def _on_copy_header(self):
        self.header_text.config(state=tk.NORMAL)
        code = self.header_text.get("1.0", tk.END)
        self.header_text.config(state=tk.DISABLED)
        self.clipboard_clear()
        self.clipboard_append(code)
        messagebox.showinfo("Clipboard", "C/C++ Header code copied to clipboard!")

    def _on_save_header_file(self):
        self.header_text.config(state=tk.NORMAL)
        code = self.header_text.get("1.0", tk.END)
        self.header_text.config(state=tk.DISABLED)
        filepath = filedialog.asksaveasfilename(
            title="Save C/C++ Header File",
            defaultextension=".h",
            filetypes=[("C/C++ Header Files", "*.h"), ("All Files", "*.*")]
        )
        if filepath:
            with open(filepath, "w") as f:
                f.write(code)
            messagebox.showinfo("Save Success", f"Header successfully written to:\n{filepath}")

    def destroy(self):
        self.audio_player.stop()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().destroy()


def main():
    app = GegenbauerFilterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
