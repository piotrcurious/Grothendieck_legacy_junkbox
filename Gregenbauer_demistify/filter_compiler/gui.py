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
from typing import Optional, List, Tuple
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

from filter_compiler.compiler import (
    GegenbauerFilterCompiler,
    FilterSpec,
    FilterResult
)
from filter_compiler.audio_processor import (
    read_wav,
    write_wav,
    generate_chirp,
    generate_noise,
    generate_multitone,
    apply_filter,
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
        spec_group = ttk.LabelFrame(parent, text="Filter Specifications", padding=10)
        spec_group.pack(fill=tk.X, pady=5)
        spec_group.columnconfigure(1, weight=1)

        # Filter Kind
        ttk.Label(spec_group, text="Filter Type:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.kind_var = tk.StringVar(value="lowpass")
        kind_cb = ttk.Combobox(spec_group, textvariable=self.kind_var, values=["lowpass", "highpass", "bandpass", "qmf"], state="readonly")
        kind_cb.grid(row=0, column=1, sticky=tk.EW, pady=3)
        kind_cb.bind("<<ComboboxSelected>>", self._on_kind_changed)

        # Order N
        ttk.Label(spec_group, text="Order N (taps):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.order_var = tk.IntVar(value=63)
        ttk.Spinbox(spec_group, from_=3, to=512, textvariable=self.order_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=3)

        # Cutoff Frequency
        ttk.Label(spec_group, text="Cutoff (Normalized):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.cutoff_var = tk.DoubleVar(value=0.25)
        ttk.Entry(spec_group, textvariable=self.cutoff_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=3)

        # Sampling Rate
        ttk.Label(spec_group, text="Sampling Rate (Hz):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.sr_var = tk.IntVar(value=44100)
        ttk.Entry(spec_group, textvariable=self.sr_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=3)

        # Algorithmic Parameters Group
        alg_group = ttk.LabelFrame(parent, text="Gegenbauer Framework Parameters", padding=10)
        alg_group.pack(fill=tk.X, pady=5)
        alg_group.columnconfigure(1, weight=1)

        # Lambda
        ttk.Label(alg_group, text="Lambda (λ > -0.5):").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.lambda_var = tk.DoubleVar(value=1.25)
        ttk.Entry(alg_group, textvariable=self.lambda_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=3)

        # Solver
        ttk.Label(alg_group, text="Solver Algorithm:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.solver_var = tk.StringVar(value="spectral_regularized")
        ttk.Combobox(alg_group, textvariable=self.solver_var, values=["spectral_regularized", "quadrature", "wls"], state="readonly").grid(row=1, column=1, sticky=tk.EW, pady=3)

        # Regularization mu
        ttk.Label(alg_group, text="Reg Weight (μ):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.mu_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(alg_group, textvariable=self.mu_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=3)

        # Buttons Frame
        btn_frame = ttk.Frame(parent, padding=5)
        btn_frame.pack(fill=tk.X, pady=10)

        self.compile_btn = ttk.Button(btn_frame, text="⚡ Compile Filter", command=self._on_compile)
        self.compile_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.pareto_btn = ttk.Button(btn_frame, text="🔍 Pareto Search", command=self._on_pareto_search)
        self.pareto_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        # Summary Metrics Text Box
        summary_group = ttk.LabelFrame(parent, text="Compilation Certification Summary", padding=5)
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

        # Playback Controls
        play_frame = ttk.Frame(controls_frame, padding=5)
        play_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=5)

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
            if kind == "highpass" and order % 2 == 0:
                self.order_var.set(min(512, order + 1))
            elif kind == "qmf" and order % 2 != 0:
                self.order_var.set(min(512, order + 1 if order > 3 else 4))
            else:
                self.order_var.set(order)
        except Exception as e:
            sys.stderr.write(f"Kind change validation error: {e}\n")

    def _on_compile(self):
        try:
            kind = self.kind_var.get()
            order = min(512, max(3, int(self.order_var.get())))
            cutoff = float(self.cutoff_var.get())
            sampling_rate = int(self.sr_var.get())
            lam = float(self.lambda_var.get())
            solver = self.solver_var.get()
            mu_reg = float(self.mu_var.get())
        except Exception as e:
            messagebox.showerror("Invalid Input", f"Please check input fields:\n{e}")
            return

        self._set_busy(True)
        res_q = queue.Queue()

        def worker():
            try:
                spec = FilterSpec(
                    kind=kind,
                    order=order,
                    cutoff=cutoff,
                    sampling_rate=sampling_rate
                )
                compiler = GegenbauerFilterCompiler(
                    lam=lam,
                    solver=solver,
                    mu_reg=mu_reg
                )
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
            kind = self.kind_var.get()
            order = min(512, max(3, int(self.order_var.get())))
            cutoff = float(self.cutoff_var.get())
            sampling_rate = int(self.sr_var.get())
            solver = self.solver_var.get()
        except Exception as e:
            messagebox.showerror("Invalid Input", f"Please check input fields:\n{e}")
            return

        self._set_busy(True)
        res_q = queue.Queue()

        def worker():
            try:
                spec = FilterSpec(
                    kind=kind,
                    order=order,
                    cutoff=cutoff,
                    sampling_rate=sampling_rate
                )
                compiler = GegenbauerFilterCompiler(solver=solver)
                best_res = compiler.pareto_search(spec, solver=solver)
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
        self.lambda_var.set(payload.lam)
        self.mu_var.set(payload.mu_reg)

        self._update_summary_display()
        self._update_freq_plots()
        self._update_taps_plots()
        self._update_header_display()

        if self.loaded_audio_data is not None:
            self._apply_filter_to_current_audio()

        messagebox.showinfo("Pareto Optimization", f"Found Pareto-Optimal Lambda = {payload.lam:.4f}, μ = {payload.mu_reg:.2e}")

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

        res = self.current_result
        spec = res.spec

        self.ax_freq.clear()
        self.ax_pass.clear()

        # Plot 1: Full Frequency Response
        self.ax_freq.plot(res.freq_grid, res.H0_response, 'b-', label='H0 (Lowpass)' if spec.kind == 'qmf' else '|H(e^{jw})|', linewidth=1.5)
        if res.H1_response is not None:
            self.ax_freq.plot(res.freq_grid, res.H1_response, 'r-', label='H1 (Highpass)', linewidth=1.5)
        self.ax_freq.axvline(spec.cutoff, color='g', linestyle=':', label=f'Cutoff = {spec.cutoff}')
        self.ax_freq.set_title(f"Frequency Response ({spec.kind.upper()}, N={spec.order}, λ={res.lam:.2f})")
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

    def _update_taps_plots(self):
        if self.current_result is None:
            return

        res = self.current_result
        spec = res.spec

        self.ax_stem.clear()
        self.ax_quant.clear()

        # Stem Plot of Taps
        self.ax_stem.stem(range(spec.order), res.h0_taps.float64_taps, linefmt='b-', markerfmt='bo', basefmt='r-', label='h0 taps')
        if spec.kind == "qmf" and res.h1_taps is not None:
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

        self.ax_quant.semilogy(range(spec.order), np.maximum(1e-16, err_q15), 'r-o', label='h0 Q15 Quant Error', markersize=4)
        self.ax_quant.semilogy(range(spec.order), np.maximum(1e-16, err_q31), 'g-s', label='h0 Q31 Quant Error', markersize=4)
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

        taps = self.current_result.h0_taps.float64_taps
        self.filtered_audio_data = apply_filter(taps, self.loaded_audio_data)
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
        self.ax_wave.plot(t[:n_wave], filt[:n_wave], 'r-', alpha=0.8, label='Filtered')
        self.ax_wave.set_title("Time Domain Waveform Comparison (First 1000 samples)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True)
        self.ax_wave.legend(loc="upper right", fontsize=8)

        # Spectral Display via Welch's Power Spectral Density across full signal duration
        nperseg = min(2048, len(orig))
        f_orig, psd_orig = signal.welch(orig, fs=sr, nperseg=nperseg)
        f_filt, psd_filt = signal.welch(filt, fs=sr, nperseg=nperseg)

        orig_db = 10 * np.log10(np.maximum(1e-12, psd_orig))
        filt_db = 10 * np.log10(np.maximum(1e-12, psd_filt))

        self.ax_spec.plot(f_orig, orig_db, 'b-', alpha=0.6, label='Original PSD')
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
