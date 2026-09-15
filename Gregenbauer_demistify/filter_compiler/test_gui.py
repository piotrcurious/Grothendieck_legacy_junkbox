"""
Unit and Integration Tests for Gegenbauer Filter Compiler GUI & Audio Processor
================================================================================
"""

import os
import sys
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

    def test_audio_player_stub(self):
        player = AudioPlayer()
        self.assertFalse(player.is_playing())
        player.stop()


class TestGUIIntegration(unittest.TestCase):
    def test_gui_creation_and_compile(self):
        app = GegenbauerFilterGUI()
        app.update()

        self.assertIsNotNone(app.current_result)
        self.assertEqual(app.current_result.spec.kind, "lowpass")

        # Change kind to highpass and compile
        app.kind_var.set("highpass")
        app.order_var.set(65)
        app._on_compile()
        app.update()

        self.assertEqual(app.current_result.spec.kind, "highpass")

        # Test audio test trigger
        app._generate_default_audio("pink")
        self.assertIsNotNone(app.loaded_audio_data)
        self.assertIsNotNone(app.filtered_audio_data)

        app.destroy()


if __name__ == "__main__":
    unittest.main()
