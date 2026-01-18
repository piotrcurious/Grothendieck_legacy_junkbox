#!/usr/bin/env python3
import argparse
import numpy as np
from numpy.linalg import svd
import math
import sys

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
    """Robustly reads P5 PGM by streaming tokens to avoid 255-truncation."""
    with open(filename, 'rb') as f:
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError('Only P5 (binary) PGM supported')

        def get_tokens():
            for line in f:
                line = line.split(b'#')[0].strip()
                if not line: continue
                for token in line.split():
                    yield token
        
        tokens = get_tokens()
        try:
            w = int(next(tokens))
            h = int(next(tokens))
            maxval = int(next(tokens))
        except StopIteration:
            raise ValueError("Invalid PGM header")

        # After reading header tokens, the remainder of the file is raw bytes
        # However, generators might have consumed the buffer start. 
        # We find the current position carefully.
        header_text = f"{magic.decode()}\n{w} {h}\n{maxval}\n".encode()
        # We need to seek past the header. In binary P5, there's exactly one whitespace after MaxVal.
        # But for simplicity in modern Python, we read the remaining bytes from the current file pointer.
        raster = f.read()
        # If the file was perfectly formed, len(raster) should be w * h. 
        # If there's an extra newline from the header, we take the last w*h bytes.
        data = np.frombuffer(raster[-w*h:], dtype=np.uint8).astype(np.float32)
        return data.reshape((h, w))

def save_pgm(filename, img):
    h, w = img.shape
    img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        header = f"P5\n{w} {h}\n255\n"
        f.write(header.encode('ascii'))
        f.write(img8.tobytes())

# ------------------------- Symmetry Helpers -------------------------
def get_block_symmetries(block):
    syms = []
    curr = block
    for _ in range(4):
        curr = np.rot90(curr)
        syms.append((curr, False))
        syms.append((np.flipud(curr), True))
    return syms

def apply_inverse_symmetry(block, s_idx):
    # Mapping back the D4 transformation
    rotations = (s_idx // 2) + 1
    flipped = (s_idx % 2) == 1
    res = block
    if flipped:
        res = np.flipud(res)
    return np.rot90(res, k=-rotations)

# ------------------------- Core Logic ------------------------
def get_canonical_representation(block, rank=RANK):
    best_repr = None
    best_meta = (0, 0)
    
    symmetries = get_block_symmetries(block)
    for s_idx, (sym_block, _) in enumerate(symmetries):
        U, s, Vt = svd(sym_block, full_matrices=False)
        Uq = np.clip(np.round(U[:, :rank] * QUANT_SCALE), -128, 127).astype(np.int8)
        Vq = np.clip(np.round(Vt[:rank, :] * QUANT_SCALE), -128, 127).astype(np.int8)
        Sq = np.round(s[:rank]).astype(np.uint16)
        
        packed = np.concatenate([
            (Uq.astype(np.int16) & 0xFF).astype(np.uint8).ravel(),
            Sq.view(np.uint8).ravel(),
            (Vq.astype(np.int16) & 0xFF).astype(np.uint8).ravel()
        ])
        
        cur = packed.copy()
        for k in range(FIELD_DEGREE):
            cand_tuple = tuple(cur.tolist())
            if best_repr is None or cand_tuple < best_repr:
                best_repr = cand_tuple
                best_meta = (s_idx, k)
            cur = frobenius_vector(cur, 1)
                
    return np.array(best_repr, dtype=np.uint8), best_meta

def compress(img):
    h, w = img.shape
    ph = ((h + BLOCK - 1) // BLOCK) * BLOCK
    pw = ((w + BLOCK - 1) // BLOCK) * BLOCK
    img_p = np.pad(img, ((0, ph - h), (0, pw - w)), mode='reflect')
    
    blocks = []
    for i in range(0, ph, BLOCK):
        for j in range(0, pw, BLOCK):
            blocks.append(img_p[i:i+BLOCK, j:j+BLOCK])
            
    dictionary, dict_list, codes = {}, [], []
    for b in blocks:
        canon, (s_idx, k) = get_canonical_representation(b, RANK)
        key = canon.tobytes()
        if key not in dictionary:
            dictionary[key] = len(dict_list)
            dict_list.append(canon)
        codes.append([dictionary[key], s_idx, k])

    meta = {'h': h, 'w': w, 'ph': ph, 'pw': pw}
    return meta, np.stack(dict_list), np.array(codes, dtype=np.int32)

def decompress(meta, dict_array, codes):
    out = np.zeros((meta['ph'], meta['pw']), dtype=np.float32)
    idx = 0
    for i in range(0, meta['ph'], BLOCK):
        for j in range(0, meta['pw'], BLOCK):
            d_idx, s_idx, k = codes[idx]
            vec = frobenius_vector(dict_array[d_idx], (-k) % FIELD_DEGREE)
            
            p = 0
            U_sz = BLOCK * RANK
            Ub = vec[p:p+U_sz].astype(np.int16); Ub[Ub >= 128] -= 256; p += U_sz
            S_sz = 2 * RANK
            Sr = np.frombuffer(vec[p:p+S_sz].tobytes(), dtype='<u2').astype(np.float32); p += S_sz
            V_sz = RANK * BLOCK
            Vb = vec[p:p+V_sz].astype(np.int16); Vb[Vb >= 128] -= 256
            
            U = Ub.reshape(BLOCK, RANK) / QUANT_SCALE
            Vt = Vb.reshape(RANK, BLOCK) / QUANT_SCALE
            
            block_recon = (U * Sr[np.newaxis, :]) @ Vt
            out[i:i+BLOCK, j:j+BLOCK] = apply_inverse_symmetry(block_recon, s_idx)
            idx += 1
            
    return out[:meta['h'], :meta['w']]

# ------------------------------ Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--compress', action='store_true')
    group.add_argument('--decompress', action='store_true')
    ap.add_argument('input'); ap.add_argument('output')
    args = ap.parse_args()

    if args.compress:
        img = load_pgm(args.input)
        meta, d, c = compress(img)
        np.savez_compressed(args.output, dict=d, codes=c, **meta)
        print(f"Compressed {args.input} ({img.shape}) -> {args.output}")
    elif args.decompress:
        z = np.load(args.input)
        meta = {k: int(z[k]) for k in ['h', 'w', 'ph', 'pw']}
        recon = decompress(meta, z['dict'], z['codes'])
        save_pgm(args.output, recon)
        print(f"Decompressed {args.input} -> {args.output} ({recon.shape})")

if __name__ == '__main__':
    main()
