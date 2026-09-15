#!/usr/bin/env python3
"""
SVD + D4 Symmetry + Galois Orbit Compressor (GF(2^8))

Architecture:
  1. Image block decomposition (e.g. 8x8 blocks).
  2. SVD rank reduction (e.g. rank-2 approximation).
  3. Sign-normalization of SVD vectors for deterministic representation.
  4. Quantization of left/right singular vectors and singular values.
  5. Canonicalization over Dihedral D4 symmetries (8 spatial transformations).
  6. Canonicalization over GF(2^8) Galois Field Frobenius orbits (8 automorphisms).
  7. Dictionary encoding of canonical byte keys.

Note on Galois Orbit Equivalence:
  The Frobenius automorphism x -> x^2 in GF(2^8) is applied componentwise to the
  serialized byte representation of the quantized SVD factors (U_q, S_q, V_q).
  This defines an equivalence relation on the serialized byte representation
  (dictionary normalization), rather than an algebraic symmetry of the original
  image block or SVD linear operator itself.

Note on SVD Singular Values:
  When singular values are nearly degenerate (sigma_1 ~ sigma_2), floating-point
  variations across implementations can lead to subspace rotation.
"""

import argparse
import math
import os
import numpy as np
from numpy.linalg import svd

# Default Parameters
DEFAULT_BLOCK = 8
DEFAULT_RANK = 2
DEFAULT_QUANT_SCALE = 127.0
DEFAULT_FIELD_DEGREE = 8
DEFAULT_AES_POLY = 0x11B


# ------------------------ Finite field helpers ------------------------

def gf_mul(a, b, aes_poly=DEFAULT_AES_POLY):
    """Galois Field multiplication in GF(2^8) with irreducible polynomial aes_poly."""
    res = 0
    a = int(a) & 0xFF
    b = int(b) & 0xFF
    for _ in range(8):
        if b & 1:
            res ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= aes_poly & 0xFF
        b >>= 1
    return res & 0xFF


def make_frobenius_table(field_degree=DEFAULT_FIELD_DEGREE, aes_poly=DEFAULT_AES_POLY):
    """Precomputes a lookup table for GF(2^8) Frobenius orbits: x -> x^(2^k)."""
    frob_table = np.empty((field_degree, 256), dtype=np.uint8)
    for k in range(field_degree):
        for x in range(256):
            y = x
            for _ in range(k):
                res = 0
                a = y
                b = y
                for _ in range(8):
                    if b & 1:
                        res ^= a
                    hi = a & 0x80
                    a = (a << 1) & 0xFF
                    if hi:
                        a ^= aes_poly & 0xFF
                    b >>= 1
                y = res & 0xFF
            frob_table[k, x] = y
    return frob_table


def frobenius_vector(v, k, frob_table):
    """Applies Frobenius mapping x -> x^(2^k) to a byte vector using precomputed lookup table."""
    field_degree = frob_table.shape[0]
    return frob_table[k % field_degree][v]


# --------------------------- Robust PGM I/O ---------------------------

def load_pgm(filename):
    """
    Robustly reads binary P5 PGM image files.

    Parses headers byte-by-byte, skipping comments (#...), handling arbitrary
    whitespace between tokens, and correctly parsing raster data even when
    maxval is not immediately followed by a newline. Explicitly rejects 16-bit
    PGMs (maxval > 255).
    """
    with open(filename, 'rb') as f:
        def read_next_token():
            token = bytearray()
            while True:
                b = f.read(1)
                if not b:
                    break
                if b == b'#':
                    # Skip comment until end of line
                    while True:
                        cb = f.read(1)
                        if not cb or cb in b'\r\n':
                            break
                    continue
                if b in b' \t\n\r':
                    if token:
                        break
                    else:
                        continue
                token.extend(b)
            return token.decode('ascii') if token else None

        magic = read_next_token()
        if magic != 'P5':
            raise ValueError(f"Only binary P5 PGM format supported, got '{magic}'")

        w_str = read_next_token()
        h_str = read_next_token()
        maxval_str = read_next_token()

        if not (w_str and h_str and maxval_str):
            raise ValueError("Invalid or incomplete PGM header")

        w = int(w_str)
        h = int(h_str)
        maxval = int(maxval_str)

        if maxval > 255:
            raise ValueError(f"16-bit PGM (maxval={maxval} > 255) is not supported")
        if maxval <= 0:
            raise ValueError(f"Invalid maxval in PGM header: {maxval}")

        # `read_next_token` stopped at the single whitespace character following maxval.
        # Exactly w * h binary raster bytes follow.
        raster = f.read(w * h)
        if len(raster) != w * h:
            raise ValueError(f"Expected {w * h} raster bytes, but read {len(raster)} bytes")

        return np.frombuffer(raster, dtype=np.uint8).astype(np.float32).reshape((h, w))


def save_pgm(filename, img):
    """Saves a 2D numpy array as a binary P5 PGM image file."""
    h, w = img.shape
    img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        header = f"P5\n{w} {h}\n255\n"
        f.write(header.encode('ascii'))
        f.write(img8.tobytes())


# ------------------------- Symmetry Helpers -------------------------

def get_block_symmetries(block):
    """Returns the 8 symmetries of a square block (Dihedral group D4)."""
    syms = []
    curr = block
    for _ in range(4):
        curr = np.rot90(curr)
        syms.append((curr, False))           # Rotation
        syms.append((np.flipud(curr), True))  # Rotation + Flip
    return syms


def apply_inverse_symmetry(block, s_idx):
    """Reverse the specific D4 transformation indexed by s_idx."""
    rotations = (s_idx // 2) + 1
    flipped = (s_idx % 2) == 1
    res = block
    if flipped:
        res = np.flipud(res)
    return np.rot90(res, k=-rotations)


# ---------------------- Block extraction / padding --------------------

def extract_blocks_padded(img, block=DEFAULT_BLOCK):
    """
    Extracts square blocks from image with padding. Uses mode='edge'
    to prevent failure on degenerate 1-pixel dimensions.
    """
    h, w = img.shape
    ph = ((h + block - 1) // block) * block
    pw = ((w + block - 1) // block) * block
    pad_h = ph - h
    pad_w = pw - w
    img_p = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

    blocks = []
    for i in range(0, ph, block):
        for j in range(0, pw, block):
            blocks.append(img_p[i:i+block, j:j+block])
    return np.array(blocks), (h, w, ph, pw)


# ------------------------- Algebraic Reduction ------------------------

def get_canonical_representation(block, rank=DEFAULT_RANK, quant_scale=DEFAULT_QUANT_SCALE,
                                   frob_table=None):
    """
    Finds the smallest byte representation across all D4 symmetries and GF(2^8) Galois orbits.
    Imposes deterministic sign convention on SVD vectors prior to quantization.
    """
    if block.shape[0] != block.shape[1]:
        raise ValueError(f"Block must be square, got shape {block.shape}")
    if not (1 <= rank <= min(block.shape)):
        raise ValueError(f"Rank ({rank}) must be between 1 and min block dimension ({min(block.shape)})")

    if frob_table is None:
        frob_table = make_frobenius_table()

    field_degree = frob_table.shape[0]
    best_repr = None
    best_meta = (0, 0)  # (s_idx, k)
    
    symmetries = get_block_symmetries(block)
    for s_idx, (sym_block, _) in enumerate(symmetries):
        # 1. Singular Value Decomposition
        U, s, Vt = svd(sym_block, full_matrices=False)
        
        # Deterministic sign flip convention per singular vector
        for r in range(rank):
            idx = np.argmax(np.abs(U[:, r]))
            if U[idx, r] < 0:
                U[:, r] *= -1
                Vt[r, :] *= -1

        Uq = np.clip(np.round(U[:, :rank] * quant_scale), -128, 127).astype(np.int8)
        Vq = np.clip(np.round(Vt[:rank, :] * quant_scale), -128, 127).astype(np.int8)

        # Explicit little-endian uint16 serialization for singular values
        Sq = np.round(s[:rank]).astype('<u2')

        # 2. Pack into byte vector
        packed = np.concatenate([
            (Uq.astype(np.int16) & 0xFF).astype(np.uint8).ravel(),
            Sq.view(np.uint8).ravel(),
            (Vq.astype(np.int16) & 0xFF).astype(np.uint8).ravel()
        ])
        
        # 3. Canonicalize over GF(2^8) Frobenius orbits
        for k in range(field_degree):
            cand_vec = frobenius_vector(packed, k, frob_table)
            cand_bytes = cand_vec.tobytes()
            if best_repr is None or cand_bytes < best_repr:
                best_repr = cand_bytes
                best_meta = (s_idx, k)
                
    return np.frombuffer(best_repr, dtype=np.uint8).copy(), best_meta


# ---------------------- Compress / Decompress -------------------------

def compress(img, block=DEFAULT_BLOCK, rank=DEFAULT_RANK, quant_scale=DEFAULT_QUANT_SCALE,
             field_degree=DEFAULT_FIELD_DEGREE, aes_poly=DEFAULT_AES_POLY):
    """
    Compresses image into dictionary of canonical SVD factors and codebook indices.
    """
    frob_table = make_frobenius_table(field_degree, aes_poly)
    blocks, (h, w, ph, pw) = extract_blocks_padded(img, block)
    dictionary, dict_list, codes = {}, [], []

    for b in blocks:
        canon, (s_idx, k) = get_canonical_representation(
            b, rank=rank, quant_scale=quant_scale, frob_table=frob_table
        )
        key = canon.tobytes()
        if key not in dictionary:
            dictionary[key] = len(dict_list)
            dict_list.append(canon)
        codes.append([dictionary[key], s_idx, k])

    meta = {
        'h': h, 'w': w, 'ph': ph, 'pw': pw,
        'block': block, 'rank': rank, 'quant_scale': quant_scale,
        'field_degree': field_degree, 'aes_poly': aes_poly
    }
    return meta, np.stack(dict_list), np.array(codes, dtype=np.int32)


def decompress(meta, dict_array, codes):
    """
    Decompresses image from dictionary and codes using stored codec parameters.
    """
    h, w, ph, pw = meta['h'], meta['w'], meta['ph'], meta['pw']
    b_size = meta.get('block', DEFAULT_BLOCK)
    rank = meta.get('rank', DEFAULT_RANK)
    quant_scale = meta.get('quant_scale', DEFAULT_QUANT_SCALE)
    field_degree = meta.get('field_degree', DEFAULT_FIELD_DEGREE)
    aes_poly = meta.get('aes_poly', DEFAULT_AES_POLY)

    frob_table = make_frobenius_table(field_degree, aes_poly)
    out = np.zeros((ph, pw), dtype=np.float32)
    idx = 0
    
    for i in range(0, ph, b_size):
        for j in range(0, pw, b_size):
            d_idx, s_idx, k = codes[idx]
            
            # 1. Reverse Galois transformation using inverse k
            inv_k = (-k) % field_degree
            vec = frobenius_vector(dict_array[d_idx], inv_k, frob_table)
            
            # 2. Unpack SVD components
            p = 0
            U_sz = b_size * rank
            Ub = vec[p:p+U_sz].astype(np.int16)
            Ub[Ub >= 128] -= 256
            p += U_sz
            
            S_sz = 2 * rank
            Sr = np.frombuffer(vec[p:p+S_sz].tobytes(), dtype='<u2').astype(np.float32)
            p += S_sz
            
            V_sz = rank * b_size
            Vb = vec[p:p+V_sz].astype(np.int16)
            Vb[Vb >= 128] -= 256
            
            U = Ub.reshape(b_size, rank) / quant_scale
            Vt = Vb.reshape(rank, b_size) / quant_scale
            
            # 3. Reconstruct block and apply inverse D4 symmetry
            block_recon = (U * Sr[np.newaxis, :]) @ Vt
            out[i:i+b_size, j:j+b_size] = apply_inverse_symmetry(block_recon, s_idx)
            idx += 1
            
    return out[:h, :w]


def compute_metrics(orig, recon, meta=None, dict_array=None, codes=None, compressed_bytes=None):
    """Computes quality metrics (RMSE, PSNR, MAE) and compression stats."""
    orig_f = orig.astype(np.float32)
    recon_f = recon.astype(np.float32)
    mse = float(np.mean((orig_f - recon_f) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.max(np.abs(orig_f - recon_f)))
    psnr = 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

    stats = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'psnr': psnr
    }

    if codes is not None and dict_array is not None:
        total_blocks = len(codes)
        dict_entries = len(dict_array)
        stats['total_blocks'] = total_blocks
        stats['dictionary_entries'] = dict_entries
        stats['dictionary_ratio'] = dict_entries / total_blocks if total_blocks > 0 else 0.0

    if compressed_bytes is not None and meta is not None:
        orig_bytes = meta['h'] * meta['w']
        stats['original_bytes'] = orig_bytes
        stats['compressed_bytes'] = compressed_bytes
        stats['compression_ratio'] = orig_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

    return stats


# ------------------------------ Main / CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="SVD + D4 + GF(2^8) Orbit Compressor")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--compress', action='store_true', help="Compress PGM image to NPZ archive")
    group.add_argument('--decompress', action='store_true', help="Decompress NPZ archive to PGM image")

    ap.add_argument('input', help="Input file path (.pgm for compress, .npz for decompress)")
    ap.add_argument('output', help="Output file path (.npz for compress, .pgm for decompress)")

    # Compression Parameters
    ap.add_argument('--block', type=int, default=DEFAULT_BLOCK, help=f"Block size (default: {DEFAULT_BLOCK})")
    ap.add_argument('--rank', type=int, default=DEFAULT_RANK, help=f"SVD rank (default: {DEFAULT_RANK})")
    ap.add_argument('--quant-scale', type=float, default=DEFAULT_QUANT_SCALE, help=f"Quantization scale (default: {DEFAULT_QUANT_SCALE})")
    ap.add_argument('--field-degree', type=int, default=DEFAULT_FIELD_DEGREE, help=f"GF field degree (default: {DEFAULT_FIELD_DEGREE})")
    ap.add_argument('--aes-poly', type=lambda x: int(x, 0), default=DEFAULT_AES_POLY, help=f"AES field polynomial (default: {hex(DEFAULT_AES_POLY)})")
    ap.add_argument('--verbose', '-v', action='store_true', help="Print detailed quality metrics and statistics")

    args = ap.parse_args()

    if args.compress:
        img = load_pgm(args.input)
        meta, d, c = compress(
            img,
            block=args.block,
            rank=args.rank,
            quant_scale=args.quant_scale,
            field_degree=args.field_degree,
            aes_poly=args.aes_poly
        )
        np.savez_compressed(args.output, dict=d, codes=c, **meta)

        comp_size = os.path.getsize(args.output)
        recon = decompress(meta, d, c)
        metrics = compute_metrics(img, recon, meta, d, c, comp_size)

        print(f"Compressed {args.input} -> {args.output}")
        print(f"  Image shape: {meta['h']}x{meta['w']} (padded to {meta['ph']}x{meta['pw']})")
        print(f"  Blocks: {metrics['total_blocks']}, Dictionary entries: {metrics['dictionary_entries']} (ratio: {metrics['dictionary_ratio']:.3f})")
        print(f"  Size: {metrics['original_bytes']} bytes -> {metrics['compressed_bytes']} bytes (ratio: {metrics['compression_ratio']:.2f}x)")
        print(f"  Reconstruction: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.1f}, PSNR={metrics['psnr']:.2f} dB")

    elif args.decompress:
        z = np.load(args.input)
        meta = {
            'h': int(z['h']), 'w': int(z['w']), 'ph': int(z['ph']), 'pw': int(z['pw']),
            'block': int(z['block']) if 'block' in z else args.block,
            'rank': int(z['rank']) if 'rank' in z else args.rank,
            'quant_scale': float(z['quant_scale']) if 'quant_scale' in z else args.quant_scale,
            'field_degree': int(z['field_degree']) if 'field_degree' in z else args.field_degree,
            'aes_poly': int(z['aes_poly']) if 'aes_poly' in z else args.aes_poly,
        }
        recon = decompress(meta, z['dict'], z['codes'])
        save_pgm(args.output, recon)
        print(f"Decompressed {args.input} -> {args.output} ({recon.shape[0]}x{recon.shape[1]})")

if __name__ == '__main__':
    main()
