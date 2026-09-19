"""
Unit and Integration Tests for Gegenbauer Filter Compiler GUI & Audio Processor
================================================================================
"""

import os
import sys
import time
import unittest
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

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
from filter_compiler.compiler import (
    GegenbauerFilterCompiler,
    FilterSpec
)
from filter_compiler.gui import GegenbauerFilterGUI


class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp_wav = "/tmp/test_audio_proc.wav"

    def tearDown(self):
        if os.path.exists(self.tmp_wav):
            os.remove(self.tmp_wav)

    def test_wav_io(self):
        sr, data = generate_chirp(duration=0.5, sample_rate=16000)
        write_wav(self.tmp_wav, sr, data)
        self.assertTrue(os.path.exists(self.tmp_wav))

        read_sr, read_data = read_wav(self.tmp_wav)
        self.assertEqual(sr, read_sr)
        self.assertEqual(len(data), len(read_data))
        self.assertAlmostEqual(float(np.max(np.abs(data - read_data))), 0.0, delta=1e-3)

    def test_noise_and_multitone(self):
        sr, white = generate_noise(duration=0.5, sample_rate=8000, noise_type="white")
        self.assertEqual(len(white), 4000)

        sr, pink = generate_noise(duration=0.5, sample_rate=8000, noise_type="pink")
        self.assertEqual(len(pink), 4000)

        sr, multi = generate_multitone(duration=0.5, sample_rate=8000)
        self.assertEqual(len(multi), 4000)

    def test_apply_filter(self):
        spec = FilterSpec(kind="lowpass", order=31, cutoff=0.2)
        compiler = GegenbauerFilterCompiler(lam=1.5)
        res = compiler.compile(spec)
        taps = res.h0_taps.float64_taps

        sr, audio = generate_chirp(duration=0.5, sample_rate=8000, f_start=50, f_end=3500)
        filtered = apply_filter(taps, audio)

        self.assertEqual(len(audio), len(filtered))
        self.assertFalse(np.allclose(audio, filtered))

    def test_qmf_filtering_modes(self):
        spec = FilterSpec(kind="qmf", order=32, cutoff=0.25)
        compiler = GegenbauerFilterCompiler(lam=1.25)
        res = compiler.compile(spec)

        h0 = res.h0_taps.float64_taps
        h1 = res.h1_taps.float64_taps
        sr, audio = generate_chirp(duration=0.5, sample_rate=8000)

        f_h0 = apply_qmf_filtering(h0, h1, audio, mode="lowpass")
        f_h1 = apply_qmf_filtering(h0, h1, audio, mode="highpass")
        f_sum = apply_qmf_filtering(h0, h1, audio, mode="reconstruction_sum")
        f_diff = apply_qmf_filtering(h0, h1, audio, mode="subband_diff")
        f_stereo = apply_qmf_filtering(h0, h1, audio, mode="stereo_split")

        self.assertEqual(len(f_h0), len(audio))
        self.assertEqual(len(f_h1), len(audio))
        self.assertTrue(np.allclose(f_sum, f_h0 + f_h1))
        self.assertTrue(np.allclose(f_diff, f_h0 - f_h1))
        self.assertEqual(f_stereo.shape, (len(audio), 2))

    def test_interactive_drag_band_limits(self):
        app = GegenbauerFilterGUI()
        spec = FilterSpec(kind="lowpass", order=31, cutoff=0.25)
        comp = GegenbauerFilterCompiler()
        app.current_result = comp.compile(spec)
        app._update_freq_plots()

        class FakeEvent:
            def __init__(self, inaxes, button, xdata):
                self.inaxes = inaxes
                self.button = button
                self.xdata = xdata

        evt_click = FakeEvent(app.ax_freq, 1, 0.252)
        app._on_freq_click(evt_click)
        self.assertEqual(app._dragging_param, "cutoff")

        evt_drag = FakeEvent(app.ax_freq, 1, 0.32)
        app._on_freq_drag(evt_drag)
        self.assertAlmostEqual(app.cutoff_var.get(), 0.32, places=2)

        app.destroy()

    def test_audio_player_stub(self):
        player = AudioPlayer()
        self.assertFalse(player.is_playing())
        player.stop()


def _wait_for_result(app, prev_res=None, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        app.update()
        if app.current_result is not None and app.current_result is not prev_res:
            return True
        time.sleep(0.02)
    return False


class TestGUIIntegration(unittest.TestCase):
    def test_gui_async_compile_and_pareto(self):
        app = GegenbauerFilterGUI()

        # Wait for initial async compile to complete
        _wait_for_result(app)

        self.assertIsNotNone(app.current_result)
        self.assertEqual(app.current_result.spec.kind, "lowpass")

        # Change kind to highpass and compile
        prev_res = app.current_result
        app.kind_var.set("highpass")
        app.order_var.set(65)
        app._on_compile()

        _wait_for_result(app, prev_res)

        self.assertEqual(app.current_result.spec.kind, "highpass")

        # Test Pareto search async execution
        prev_res = app.current_result
        app.solver_var.set("spectral_regularized")
        app._on_pareto_search()

        _wait_for_result(app, prev_res, timeout=10.0)

        self.assertIsNotNone(app.current_result)
        self.assertEqual(app.mu_var.get(), app.current_result.mu_reg)

        # Test audio test trigger and WAV sampling rate sync
        app._generate_default_audio("pink")
        self.assertIsNotNone(app.loaded_audio_data)
        self.assertIsNotNone(app.filtered_audio_data)

        app.destroy()

    def test_gui_asymmetric_qmf_compilation(self):
        app = GegenbauerFilterGUI()

        _wait_for_result(app)

        prev_res = app.current_result
        app.kind_var.set("asymmetric_qmf")
        app.order_var.set(32)
        app.cutoff_var.set(0.23)
        app.wp_var.set("0.20")
        app.ws_var.set("0.26")
        app.solver_var.set("least_squares")
        app.basis_type_var.set("unnormalized")
        app._on_compile()

        _wait_for_result(app, prev_res)

        self.assertIsNotNone(app.current_result)
        self.assertEqual(app.current_result.spec.kind, "asymmetric_qmf")

        # Test live audio filtering in Reconstruction Sum mode
        app.qmf_mode_var.set("H0 + H1 (Reconstruction Sum)")
        app._apply_filter_to_current_audio()
        self.assertIsNotNone(app.filtered_audio_data)

        app.destroy()

    def test_gui_biorthogonal_compilation_and_audio(self):
        app = GegenbauerFilterGUI()

        _wait_for_result(app)

        prev_res = app.current_result
        app.kind_var.set("biorthogonal")
        app.order_var.set(9)
        app.order_g0_var.set(7)
        app.cutoff_var.set(0.25)
        app.lambda_var.set(1.5)
        app._on_compile()

        _wait_for_result(app, prev_res)

        from filter_compiler.gui import BiorthogonalResult
        self.assertIsInstance(app.current_result, BiorthogonalResult)
        self.assertEqual(app.current_result.order_h0, 9)
        self.assertEqual(app.current_result.order_g0, 7)
        self.assertIn("GEG_ORDER_H0 9", app.current_result.header_code)
        self.assertIn("GEG_ORDER_G0 7", app.current_result.header_code)

        # Test live audio PR reconstruction mode
        app.qmf_mode_var.set("Full PR Reconstruction")
        app._apply_filter_to_current_audio()
        self.assertIsNotNone(app.filtered_audio_data)

        app.destroy()


if __name__ == "__main__":
    unittest.main()
