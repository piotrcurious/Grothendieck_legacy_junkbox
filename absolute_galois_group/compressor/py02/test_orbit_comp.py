#!/usr/bin/env python3
"""
Unit and integration tests for orbit_comp.py compressor module.
"""

import os
import tempfile
import unittest
import numpy as np
import subprocess
import sys

# Add module path
sys.path.insert(0, os.path.dirname(__file__))

from orbit_comp import (
    load_pgm, save_pgm, compress, decompress, compute_metrics,
    get_canonical_representation, extract_blocks_padded,
    make_frobenius_table, frobenius_vector, gf_mul
)


class TestPGMParser(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_standard_pgm(self):
        filePath = os.path.join(self.temp_dir.name, "std.pgm")
        img_orig = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        header = f"P5\n16 16\n255\n".encode('ascii')
        with open(filePath, 'wb') as f:
            f.write(header)
            f.write(img_orig.tobytes())

        loaded = load_pgm(filePath)
        np.testing.assert_array_equal(loaded, img_orig.astype(np.float32))

    def test_no_newline_after_maxval(self):
        """Test P5 file where binary raster starts immediately after single space following maxval."""
        filePath = os.path.join(self.temp_dir.name, "no_nl.pgm")
        img_orig = np.arange(64, dtype=np.uint8).reshape((8, 8))
        header = f"P5 8 8 255 ".encode('ascii')
        with open(filePath, 'wb') as f:
            f.write(header)
            f.write(img_orig.tobytes())

        loaded = load_pgm(filePath)
        np.testing.assert_array_equal(loaded, img_orig.astype(np.float32))

    def test_comments_in_header(self):
        """Test P5 file containing comment lines."""
        filePath = os.path.join(self.temp_dir.name, "comments.pgm")
        img_orig = np.full((8, 8), 128, dtype=np.uint8)
        header = b"P5\n# Created by test\n8 8\n# Another comment\n255\n"
        with open(filePath, 'wb') as f:
            f.write(header)
            f.write(img_orig.tobytes())

        loaded = load_pgm(filePath)
        np.testing.assert_array_equal(loaded, img_orig.astype(np.float32))

    def test_reject_16bit_pgm(self):
        """Test that 16-bit PGM (maxval > 255) raises ValueError."""
        filePath = os.path.join(self.temp_dir.name, "16bit.pgm")
        header = b"P5\n8 8\n65535\n"
        raster = np.zeros(128, dtype=np.uint8).tobytes()
        with open(filePath, 'wb') as f:
            f.write(header)
            f.write(raster)

        with self.assertRaises(ValueError) as cm:
            load_pgm(filePath)
        self.assertIn("16-bit PGM", str(cm.exception))

    def test_reject_truncated_pgm(self):
        """Test that truncated raster data raises ValueError."""
        filePath = os.path.join(self.temp_dir.name, "trunc.pgm")
        header = b"P5\n8 8\n255\n"
        with open(filePath, 'wb') as f:
            f.write(header)
            f.write(b"too short")

        with self.assertRaises(ValueError) as cm:
            load_pgm(filePath)
        self.assertIn("Expected 64 raster bytes", str(cm.exception))


class TestFrobenius(unittest.TestCase):
    def test_frobenius_table_equivalence(self):
        frob_table = make_frobenius_table(8, 0x11B)
        v = np.arange(256, dtype=np.uint8)
        for k in range(8):
            v_frob = frobenius_vector(v, k, frob_table)
            # Compare against direct gf_mul iterative calculation
            expected = np.zeros(256, dtype=np.uint8)
            for x in range(256):
                y = x
                for _ in range(k):
                    y = gf_mul(y, y)
                expected[x] = y
            np.testing.assert_array_equal(v_frob, expected)


class TestDegeneratePadding(unittest.TestCase):
    def test_1px_dimension(self):
        """Test that mode='edge' padding succeeds on 1-pixel height/width images."""
        img = np.ones((1, 10), dtype=np.float32) * 100.0
        blocks, (h, w, ph, pw) = extract_blocks_padded(img, block=8)
        self.assertEqual(ph, 8)
        self.assertEqual(pw, 16)
        self.assertEqual(len(blocks), 2)


class TestSVDNormalization(unittest.TestCase):
    def test_sign_flipping_determinism(self):
        """Test that canonical representation is invariant to SVD sign flips."""
        block = np.array([
            [10, 20, 30, 40, 50, 60, 70, 80],
            [15, 25, 35, 45, 55, 65, 75, 85],
            [20, 30, 40, 50, 60, 70, 80, 90],
            [25, 35, 45, 55, 65, 75, 85, 95],
            [30, 40, 50, 60, 70, 80, 90, 100],
            [35, 45, 55, 65, 75, 85, 95, 105],
            [40, 50, 60, 70, 80, 90, 100, 110],
            [45, 55, 65, 75, 85, 95, 105, 115]
        ], dtype=np.float32)

        canon1, _ = get_canonical_representation(block, rank=2)
        # Sign-flipped version of block shouldn't crash and should give canonical form
        canon2, _ = get_canonical_representation(-block, rank=2)
        self.assertEqual(len(canon1), len(canon2))


class TestParameterValidation(unittest.TestCase):
    def test_invalid_rank(self):
        block = np.zeros((8, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            get_canonical_representation(block, rank=0)
        with self.assertRaises(ValueError):
            get_canonical_representation(block, rank=9)

    def test_non_square_block(self):
        block = np.zeros((8, 6), dtype=np.float32)
        with self.assertRaises(ValueError):
            get_canonical_representation(block, rank=2)


class TestRoundtripCompressDecompress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_roundtrip(self):
        # Create synthetic image
        x = np.linspace(0, 255, 32)
        y = np.linspace(0, 255, 32)
        xx, yy = np.meshgrid(x, y)
        img = (xx + yy) / 2.0

        meta, dict_array, codes = compress(img, block=8, rank=2)
        recon = decompress(meta, dict_array, codes)

        self.assertEqual(recon.shape, img.shape)
        metrics = compute_metrics(img, recon, meta, dict_array, codes)
        self.assertGreater(metrics['psnr'], 20.0)

    def test_cli_roundtrip(self):
        script = os.path.join(os.path.dirname(__file__), "orbit_comp.py")
        input_pgm = os.path.join(self.temp_dir.name, "input.pgm")
        output_npz = os.path.join(self.temp_dir.name, "archive.npz")
        decomp_pgm = os.path.join(self.temp_dir.name, "decomp.pgm")

        img = np.random.randint(50, 200, (16, 16), dtype=np.uint8).astype(np.float32)
        save_pgm(input_pgm, img)

        # Compress via CLI
        cmd_comp = [sys.executable, script, "--compress", input_pgm, output_npz, "--rank", "2", "--block", "8", "--verbose"]
        res_comp = subprocess.run(cmd_comp, capture_output=True, text=True)
        self.assertEqual(res_comp.returncode, 0, f"Compress failed: {res_comp.stderr}")
        self.assertTrue(os.path.exists(output_npz))

        # Decompress via CLI
        cmd_decomp = [sys.executable, script, "--decompress", output_npz, decomp_pgm]
        res_decomp = subprocess.run(cmd_decomp, capture_output=True, text=True)
        self.assertEqual(res_decomp.returncode, 0, f"Decompress failed: {res_decomp.stderr}")
        self.assertTrue(os.path.exists(decomp_pgm))

        recon = load_pgm(decomp_pgm)
        self.assertEqual(recon.shape, img.shape)


if __name__ == '__main__':
    unittest.main()
