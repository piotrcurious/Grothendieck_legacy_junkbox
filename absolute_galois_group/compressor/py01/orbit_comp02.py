"""
Prototype: Algebraic Basis Reduction + TRUE Finite-Field Galois Conjugacy

Bugfix release: fixes for
 - codes array dtype truncation (was uint8 causing wrap / DeprecationWarning)
 - padding and block extraction (previous version silently dropped border pixels)
 - signed quantized values correctly recovered from stored uint8 representation
 - robust saving of metadata (original & padded shapes)

Usage:
  python3 algebraic_basis_reduction_prototype.py --compress in.pgm out.npz
  python3 algebraic_basis_reduction_prototype.py --decompress in.npz out.pgm

"""

import argparse
import numpy as np
from numpy.linalg import svd
import math

# ----------------------------- Parameters -----------------------------
BLOCK = 8
RANK = 2
QUANT_SCALE = 127.0
SV_QUANT_SCALE = 100.0
FIELD_DEGREE = 8  # GF(2^8)
AES_POLY = 0x11B
# ---------------------------------------------------------------------

# ======================= Finite Field GF(2^8) =========================

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

# ============================ PGM I/O =================================

# --------------------------- Robust PGM I/O ----------------------------
def load_pgm(filename):
    """Load binary PGM (P5). Returns float32 ndarray shape (h,w)."""
    with open(filename, 'rb') as f:
        # magic
        magic = f.readline()
        if not magic:
            raise ValueError("Empty file")
        if magic.strip() != b'P5':
            raise ValueError('Only P5 (binary) PGM supported')

        # read header tokens, skipping comments
        def _read_token():
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading header")
                line = line.strip()
                if line.startswith(b'#') or line == b'':
                    continue
                return line

        # read width/height (may be on one or two tokens)
        token = _read_token()
        parts = token.split()
        if len(parts) >= 2:
            w = int(parts[0]); h = int(parts[1])
        else:
            w = int(parts[0])
            token = _read_token()
            h = int(token.split()[0])

        # maxval
        token = _read_token()
        maxv = int(token.split()[0])
        if maxv > 255:
            raise ValueError("Only 8-bit PGM (maxval <= 255) supported")

        # read exactly w*h bytes of raster
        expected = w * h
        raster = f.read(expected)
        if len(raster) < expected:
            raise ValueError("File truncated: expected %d bytes, got %d" % (expected, len(raster)))
        data = np.frombuffer(raster, dtype=np.uint8).astype(np.float32)
        return data.reshape((h, w))


def save_pgm(filename, img):
    """Save ndarray (float or int) as binary PGM (P5)."""
    h, w = img.shape
    img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        # correct bytes literal with newline and closing quote
        f.write(b'P5\n')
        # minimal header; ensure newline termination
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
    # Pack into a 1D uint8 vector preserving two's complement for signed parts
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

def compress(img):
    blocks, coords, padded_shape = extract_blocks_padded(img, BLOCK)
    dictionary = {}
    dict_list = []
    # codes: two columns: dict_idx (int32), frobenius power k (uint8)
    codes = np.zeros((len(blocks), 2), dtype=np.int32)

    for idx, b in enumerate(blocks):
        U, s, Vt = svd_truncate(b, RANK)
        Uq, Sq, Vq = quantize(U, s, Vt)
        packed = pack_repr(Uq, Sq, Vq)
        canon, k = galois_canonical(packed)
        key = bytes(canon.tobytes())
        if key not in dictionary:
            dictionary[key] = len(dict_list)
            dict_list.append(canon)
        codes[idx, 0] = dictionary[key]
        codes[idx, 1] = int(k)

    if len(dict_list) == 0:
        dict_array = np.empty((0, 0), dtype=np.uint8)
    else:
        dict_array = np.stack(dict_list, axis=0).astype(np.uint8)

    meta = {
        'orig_shape': img.shape,
        'padded_shape': padded_shape,
        'block': BLOCK,
        'rank': RANK,
        'num_blocks': len(blocks),
        'dict_len': dict_array.shape[0]
    }
    return meta, dict_array, codes


def decompress(meta, dict_array, codes):
    h0, w0 = meta['orig_shape']
    h_p, w_p = meta['padded_shape']
    block = meta['block']
    rank = meta['rank']
    out = np.zeros((h_p, w_p), dtype=np.float32)

    dict_len = dict_array.shape[0]
    total_blocks = codes.shape[0]
    idx = 0
    for i in range(0, h_p, block):
        for j in range(0, w_p, block):
            dict_idx = int(codes[idx, 0])
            k = int(codes[idx, 1])
            if dict_idx < 0 or dict_idx >= dict_len:
                # corrupted index -> leave block zeros
                idx += 1
                continue
            canon = dict_array[dict_idx]
            vec = frobenius_vector(canon, (-k) % FIELD_DEGREE)
            p = 0
            # U part
            U_count = block * rank
            U_bytes = vec[p:p+U_count].astype(np.int16); p += U_count
            # reinterpret two's complement
            U_bytes[U_bytes >= 128] -= 256
            Uq = U_bytes.reshape(block, rank).astype(np.float32)
            # S part
            S_bytes = vec[p:p+rank].astype(np.float32); p += rank
            Sr = S_bytes / SV_QUANT_SCALE
            # V part
            V_count = rank * block
            V_bytes = vec[p:p+V_count].astype(np.int16); p += V_count
            V_bytes[V_bytes >= 128] -= 256
            Vq = V_bytes.reshape(rank, block).astype(np.float32)

            U = Uq / QUANT_SCALE
            Vt = Vq / QUANT_SCALE
            block_recon = (U * Sr[np.newaxis, :]) @ Vt
            out[i:i+block, j:j+block] = block_recon
            idx += 1
    return out[:h0, :w0]

# =============================== CLI ==================================

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--compress', action='store_true')
    ap.add_argument('--decompress', action='store_true')
    ap.add_argument('input')
    ap.add_argument('output')
    args = ap.parse_args()

    if args.compress:
        img = load_pgm(args.input)
        meta, dict_array, codes = compress(img)
        np.savez_compressed(args.output, meta=meta, dict=dict_array, codes=codes)
        print('[*] Compressed: dict size =', dict_array.shape)
    elif args.decompress:
        z = np.load(args.input, allow_pickle=True)
        meta = dict(z['meta'].item()) if z['meta'].dtype == object else dict(z['meta'].tolist())
        dict_array = z['dict']
        codes = z['codes']
        img = decompress(meta, dict_array, codes)
        save_pgm(args.output, img)
        print('[*] Decompressed written to', args.output)
