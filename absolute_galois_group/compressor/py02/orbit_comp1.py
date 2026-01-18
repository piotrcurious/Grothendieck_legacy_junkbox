#!/usr/bin/env python3
import argparse
import numpy as np
from numpy.linalg import svd
import math
import sys
import os

# ----------------------------- Parameters -----------------------------
BLOCK = 8
RANK = 2
QUANT_SCALE = 127.0
FIELD_DEGREE = 8
AES_POLY = 0x11B

# ------------------------ Finite field helpers ------------------------
def gf_mul(a, b):
    res = 0
    for _ in range(8):
        if b & 1: res ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi: a ^= AES_POLY & 0xFF
        b >>= 1
    return res & 0xFF

def frobenius_vector(v, k):
    def _frob(x):
        res = int(x) & 0xFF
        for _ in range(k % FIELD_DEGREE):
            res = gf_mul(res, res)
        return res
    return np.fromiter((_frob(x) for x in v), dtype=np.uint8)

# --------------------------- Robust PGM I/O ---------------------------
def load_pgm(filename):
    with open(filename, 'rb') as f:
        magic = f.readline()
        if not magic or magic.strip() != b'P5':
            raise ValueError('Only P5 (binary) PGM supported')
        def _read_token():
            while True:
                line = f.readline()
                if not line: raise ValueError("Unexpected EOF")
                line = line.strip()
                if line.startswith(b'#') or line == b'': continue
                return line
        w = int(_read_token().split()[0])
        h = int(_read_token().split()[0])
        maxv = int(_read_token().split()[0])
        raster = f.read(w * h)
        return np.frombuffer(raster, dtype=np.uint8).astype(np.float32).reshape((h, w))

def save_pgm(filename, img):
    h, w = img.shape
    img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        f.write(f"P5\n{w} {h}\n255\n".encode('ascii'))
        f.write(img8.tobytes())

# ------------------------- Symmetry Helpers -------------------------
def get_block_symmetries(block):
    """Returns the 8 symmetries of a square block (D4 group)."""
    syms = []
    curr = block
    for _ in range(4):
        curr = np.rot90(curr)
        syms.append((curr, False))          # Rotation
        syms.append((np.flipud(curr), True)) # Rotation + Flip
    return syms

def apply_inverse_symmetry(block, s_idx):
    """Undo the D4 symmetry transformation based on index."""
    rotations = (s_idx // 2) + 1
    flipped = (s_idx % 2) == 1
    res = block
    if flipped:
        res = np.flipud(res)
    return np.rot90(res, k=-rotations)

# ---------------------- Block extraction / padding --------------------
def extract_blocks_padded(img, block=BLOCK):
    h, w = img.shape
    ph, pw = (-h) % block, (-w) % block
    img_p = np.pad(img, ((0, ph), (0, pw)), mode='reflect')
    blocks = []
    for i in range(0, img_p.shape[0], block):
        for j in range(0, img_p.shape[1], block):
            blocks.append(img_p[i:i+block, j:j+block])
    return np.array(blocks), img_p.shape

# ------------------------- Algebraic Reduction ------------------------
def get_canonical_representation(block, rank=RANK):
    best_repr = None
    best_meta = (0, 0) # (s_idx, k)
    
    symmetries = get_block_symmetries(block)
    for s_idx, (sym_block, _) in enumerate(symmetries):
        # 1. SVD & Quantize
        U, s, Vt = svd(sym_block, full_matrices=False)
        Uq = np.clip(np.round(U[:, :rank] * QUANT_SCALE), -128, 127).astype(np.int8)
        Vq = np.clip(np.round(Vt[:rank, :] * QUANT_SCALE), -128, 127).astype(np.int8)
        Sq = np.round(s[:rank]).astype(np.uint16)
        
        # 2. Pack
        packed = np.concatenate([
            (Uq.astype(np.int16) & 0xFF).astype(np.uint8).ravel(),
            Sq.view(np.uint8).ravel(),
            (Vq.astype(np.int16) & 0xFF).astype(np.uint8).ravel()
        ])
        
        # 3. Find best Galois automorphism
        cur = packed.copy()
        for k in range(FIELD_DEGREE):
            cand_tuple = tuple(cur.tolist())
            if best_repr is None or cand_tuple < best_repr:
                best_repr = cand_tuple
                best_meta = (s_idx, k)
            cur = frobenius_vector(cur, 1)
                
    return np.array(best_repr, dtype=np.uint8), best_meta

# ---------------------- Compress / Decompress -------------------------
def compress(img, verbose=False):
    blocks, padded_shape = extract_blocks_padded(img, BLOCK)
    dictionary, dict_list, codes = {}, [], []

    for b in blocks:
        canon, (s_idx, k) = get_canonical_representation(b, RANK)
        key = canon.tobytes()
        if key not in dictionary:
            dictionary[key] = len(dict_list)
            dict_list.append(canon)
        codes.append([dictionary[key], s_idx, k])

    meta = {'h': img.shape[0], 'w': img.shape[1], 'ph': padded_shape[0], 'pw': padded_shape[1]}
    return meta, np.stack(dict_list), np.array(codes, dtype=np.int32)

def decompress(meta, dict_array, codes):
    out = np.zeros((meta['ph'], meta['pw']), dtype=np.float32)
    b_size, rank, idx = BLOCK, RANK, 0
    
    for i in range(0, meta['ph'], b_size):
        for j in range(0, meta['pw'], b_size):
            d_idx, s_idx, k = codes[idx]
            # 1. Reverse Galois
            vec = frobenius_vector(dict_array[d_idx], (-k) % FIELD_DEGREE)
            
            # 2. Unpack SVD
            p = 0
            U_sz = b_size * rank
            Ub = vec[p:p+U_sz].astype(np.int16); Ub[Ub >= 128] -= 256; p += U_sz
            
            S_sz = 2 * rank
            Sr = np.frombuffer(vec[p:p+S_sz].tobytes(), dtype='<u2').astype(np.float32); p += S_sz
            
            V_sz = rank * b_size
            Vb = vec[p:p+V_sz].astype(np.int16); Vb[Vb >= 128] -= 256
            
            U, Vt = Ub.reshape(b_size, rank)/QUANT_SCALE, Vb.reshape(rank, b_size)/QUANT_SCALE
            
            # 3. Recon & Reverse Symmetry
            block_recon = (U * Sr[np.newaxis, :]) @ Vt
            out[i:i+b_size, j:j+b_size] = apply_inverse_symmetry(block_recon, s_idx)
            idx += 1
            
    return out[:meta['h'], :meta['w']]

# ------------------------------ CLI -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compress', action='store_true')
    ap.add_argument('--decompress', action='store_true')
    ap.add_argument('input'); ap.add_argument('output')
    args = ap.parse_args()

    if args.compress:
        img = load_pgm(args.input)
        meta, d, c = compress(img)
        np.savez_compressed(args.output, dict=d, codes=c, **meta)
    elif args.decompress:
        z = np.load(args.input)
        meta = {k: int(z[k]) for k in ['h', 'w', 'ph', 'pw']}
        recon = decompress(meta, z['dict'], z['codes'])
        save_pgm(args.output, recon)

if __name__ == '__main__':
    main()
