#!/usr/bin/env python3
# Debug-ready orbit compressor (GF(2^8) Frobenius canonicalization)
# Replace your current file with this and run.

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
SV_QUANT_SCALE = 100.0
FIELD_DEGREE = 8  # GF(2^8)
AES_POLY = 0x11B
# ---------------------------------------------------------------------

# ------------------------ Finite field helpers ------------------------
def gf_mul(a, b):
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= AES_POLY & 0xFF
        b >>= 1
    return res & 0xFF

def gf_square(a):
    return gf_mul(a, a)

def frobenius(a, k):
    for _ in range(k % FIELD_DEGREE):
        a = gf_square(a)
    return a & 0xFF

def frobenius_vector(v, k):
    return np.fromiter((frobenius(int(x) & 0xFF, k) for x in v), dtype=np.uint8)

# --------------------------- Robust PGM I/O ---------------------------
def load_pgm(filename):
    with open(filename, 'rb') as f:
        magic = f.readline()
        if not magic or magic.strip() != b'P5':
            raise ValueError('Only P5 (binary) PGM supported')
        def _read_token():
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading header")
                line = line.strip()
                if line.startswith(b'#') or line == b'':
                    continue
                return line
        token = _read_token()
        parts = token.split()
        if len(parts) >= 2:
            w = int(parts[0]); h = int(parts[1])
        else:
            w = int(parts[0])
            token = _read_token()
            h = int(token.split()[0])
        token = _read_token()
        maxv = int(token.split()[0])
        if maxv > 255:
            raise ValueError("Only 8-bit PGM supported")
        expected = w * h
        raster = f.read(expected)
        if len(raster) < expected:
            raise ValueError("File truncated: expected %d bytes, got %d" % (expected, len(raster)))
        data = np.frombuffer(raster, dtype=np.uint8).astype(np.float32)
        return data.reshape((h, w))

def save_pgm(filename, img):
    h, w = img.shape
    img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        f.write(b'P5\n')
        f.write(f"{w} {h}\n".encode('ascii'))
        f.write(b'255\n')
        f.write(img8.tobytes())

# ---------------------- Block extraction / padding --------------------
def pad_to_block(img, block=BLOCK):
    h, w = img.shape
    pad_h = (-h) % block
    pad_w = (-w) % block
    if pad_h > 0:
        top = img[:pad_h, :][::-1, :]
        img = np.vstack([img, top])
    if pad_w > 0:
        left = img[:, :pad_w][:, ::-1]
        img = np.hstack([img, left])
    return img

def extract_blocks_padded(img, block=BLOCK):
    img_p = pad_to_block(img, block)
    h, w = img_p.shape
    blocks = []
    coords = []
    for i in range(0, h, block):
        for j in range(0, w, block):
            blocks.append(img_p[i:i+block, j:j+block].astype(np.float32))
            coords.append((i, j))
    return np.array(blocks), coords, img_p.shape

# ------------------------- Algebraic Reduction ------------------------
def svd_truncate(block, rank=RANK):
    U, s, Vt = svd(block, full_matrices=False)
    return U[:, :rank], s[:rank], Vt[:rank, :]

def quantize(U, s, Vt):
    Uq = np.clip(np.round(U * QUANT_SCALE), -128, 127).astype(np.int8)
    Vq = np.clip(np.round(Vt * QUANT_SCALE), -128, 127).astype(np.int8)
    Sq = np.clip(np.round(s * SV_QUANT_SCALE), 0, 255).astype(np.uint8)
    return Uq, Sq, Vq

def pack_repr(Uq, Sq, Vq):
    # Pack into uint8 preserving two's complement for signed parts
    U_bytes = (Uq.astype(np.int16) & 0xFF).astype(np.uint8).ravel()
    S_bytes = Sq.ravel().astype(np.uint8)
    V_bytes = (Vq.astype(np.int16) & 0xFF).astype(np.uint8).ravel()
    return np.concatenate([U_bytes, S_bytes, V_bytes]).astype(np.uint8)

# ======================= True Galois Canonical ========================
def galois_canonical(byte_vec):
    best = None
    best_k = 0
    cur = byte_vec.copy()
    for k in range(FIELD_DEGREE):
        if best is None or tuple(cur.tolist()) < tuple(best.tolist()):
            best = cur.copy()
            best_k = k
        cur = frobenius_vector(cur, 1)
    return best, best_k

# ======================= Compression / Decompression ==================
def compress(img, verbose=False):
    blocks, coords, padded_shape = extract_blocks_padded(img, BLOCK)
    dictionary = {}
    dict_list = []
    codes = np.zeros((len(blocks), 2), dtype=np.int32)  # dict_idx, frob_k

    if verbose:
        print(f"[compress] image orig shape={img.shape}, padded={padded_shape}, blocks={len(blocks)}")

    for idx, b in enumerate(blocks):
        U, s, Vt = svd_truncate(b, RANK)
        if verbose and idx < 3:
            print(f"[compress] block {idx} top singulars (raw): {s[:min(len(s),5)]}")
        Uq, Sq, Vq = quantize(U, s, Vt)
        packed = pack_repr(Uq, Sq, Vq)
        canon, k = galois_canonical(packed)
        key = bytes(canon.tobytes())
        if key not in dictionary:
            dictionary[key] = len(dict_list)
            dict_list.append(canon)
        codes[idx, 0] = dictionary[key]
        codes[idx, 1] = int(k)

    dict_array = np.stack(dict_list, axis=0).astype(np.uint8) if len(dict_list) > 0 else np.zeros((0, (BLOCK*RANK)+RANK+(RANK*BLOCK)), dtype=np.uint8)

    meta = {
        'orig_shape_h': int(img.shape[0]),
        'orig_shape_w': int(img.shape[1]),
        'padded_h': int(padded_shape[0]),
        'padded_w': int(padded_shape[1]),
        'block': int(BLOCK),
        'rank': int(RANK),
        'num_blocks': int(len(blocks)),
        'dict_len': int(dict_array.shape[0]),
    }
    if verbose:
        print(f"[compress] dict_len={meta['dict_len']}, codes.shape={codes.shape}")
        if meta['dict_len'] > 0:
            print(f"[compress] sample dict[0] (first 32 bytes): {dict_array[0,:32].tolist()}")
            uniq = np.unique(codes[:,0])
            print(f"[compress] unique dict indices (first 20): {uniq[:20].tolist()}, min={uniq.min()}, max={uniq.max()}")
    return meta, dict_array, codes

def decompress(meta, dict_array, codes, verbose=False):
    h0, w0 = int(meta['orig_shape_h']), int(meta['orig_shape_w'])
    h_p, w_p = int(meta['padded_h']), int(meta['padded_w'])
    block = int(meta['block'])
    rank = int(meta['rank'])
    out = np.zeros((h_p, w_p), dtype=np.float32)

    dict_len = dict_array.shape[0]
    total_blocks = codes.shape[0]
    if verbose:
        print(f"[decompress] meta: orig=({h0},{w0}), padded=({h_p},{w_p}), dict_len={dict_len}, total_blocks={total_blocks}")

    idx = 0
    for i in range(0, h_p, block):
        for j in range(0, w_p, block):
            if idx >= total_blocks:
                if verbose: print(f"[decompress] ran out of codes at block idx {idx}")
                break
            dict_idx = int(codes[idx, 0])
            k = int(codes[idx, 1])
            if dict_idx < 0 or dict_idx >= dict_len:
                # corrupted index -> fill block with mean gray (safer than zero)
                if verbose and idx < 5:
                    print(f"[decompress] invalid dict_idx {dict_idx} at block {idx} (skipping)")
                out[i:i+block, j:j+block] = 127.0
                idx += 1
                continue
            canon = dict_array[dict_idx]
            vec = frobenius_vector(canon, (-k) % FIELD_DEGREE)
            p = 0
            U_count = block * rank
            U_bytes = vec[p:p+U_count].astype(np.int16); p += U_count
            U_bytes[U_bytes >= 128] -= 256
            Uq = U_bytes.reshape(block, rank).astype(np.float32)
            S_bytes = vec[p:p+rank].astype(np.float32); p += rank
            Sr = S_bytes / SV_QUANT_SCALE
            V_count = rank * block
            V_bytes = vec[p:p+V_count].astype(np.int16); p += V_count
            V_bytes[V_bytes >= 128] -= 256
            Vq = V_bytes.reshape(rank, block).astype(np.float32)

            U = Uq / QUANT_SCALE
            Vt = Vq / QUANT_SCALE
            block_recon = (U * Sr[np.newaxis, :]) @ Vt
            out[i:i+block, j:j+block] = block_recon
            if verbose and idx < 3:
                print(f"[decompress] block {idx} recon min/max {block_recon.min():.2f}/{block_recon.max():.2f}")
            idx += 1

    return out[:h0, :w0]

# ------------------------------ CLI / Test -----------------------------
def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compress', action='store_true')
    ap.add_argument('--decompress', action='store_true')
    ap.add_argument('--debug', action='store_true', help='Run an internal self-test and verbose logging')
    ap.add_argument('input')
    ap.add_argument('output')
    args = ap.parse_args()

    if args.debug:
        # quick self test with a small synthetic image
        print("[debug] Running self-test with random image 16x16")
        test = (np.random.rand(16,16) * 255.0).astype(np.float32)
        meta, dict_array, codes = compress(test, verbose=True)
        recon = decompress(meta, dict_array, codes, verbose=True)
        print("[debug] PSNR synthetic:", psnr(test, recon))
        # save sample files
        np.savez_compressed('debug_sample.npz', meta=meta, dict=dict_array, codes=codes)
        save_pgm('debug_recon.pgm', recon)
        print("[debug] Wrote debug_sample.npz and debug_recon.pgm")
        return

    if args.compress:
        img = load_pgm(args.input)
        meta, dict_array, codes = compress(img, verbose=True)
        # Save explicit entries rather than a single dict to avoid pickle surprises
        np.savez_compressed(args.output,
                            orig_shape_h=meta['orig_shape_h'],
                            orig_shape_w=meta['orig_shape_w'],
                            padded_h=meta['padded_h'],
                            padded_w=meta['padded_w'],
                            block=meta['block'],
                            rank=meta['rank'],
                            num_blocks=meta['num_blocks'],
                            dict_len=meta['dict_len'],
                            dict=dict_array,
                            codes=codes)
        print('[*] Compressed and saved to', args.output)
    elif args.decompress:
        z = np.load(args.input, allow_pickle=True)
        # Reconstruct meta
        meta = {
            'orig_shape_h': int(z['orig_shape_h'].tolist()),
            'orig_shape_w': int(z['orig_shape_w'].tolist()),
            'padded_h': int(z['padded_h'].tolist()),
            'padded_w': int(z['padded_w'].tolist()),
            'block': int(z['block'].tolist()),
            'rank': int(z['rank'].tolist()),
            'num_blocks': int(z['num_blocks'].tolist()),
            'dict_len': int(z['dict_len'].tolist())
        }
        dict_array = z['dict']
        codes = z['codes']
        recon = decompress(meta, dict_array, codes, verbose=True)
        save_pgm(args.output, recon)
        print('[*] Decompressed and wrote', args.output)
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
