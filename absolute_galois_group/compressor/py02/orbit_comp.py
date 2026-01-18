import argparse
import numpy as np
from numpy.linalg import svd
import math

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
    # Vectorized Frobenius for speed
    def _frob(x):
        res = int(x) & 0xFF
        for _ in range(k % FIELD_DEGREE):
            res = gf_mul(res, res)
        return res
    return np.fromiter((_frob(x) for x in v), dtype=np.uint8)

# ------------------------- Symmetry Helpers -------------------------
def get_block_symmetries(block):
    """Returns the 8 symmetries of a square block (D4 group)."""
    syms = []
    curr = block
    for _ in range(4):
        curr = np.rot90(curr)
        syms.append((curr, False))  # Rotations
        syms.append((np.flipud(curr), True)) # Rotation + Flip
    return syms

# ------------------------- Algebraic Reduction ------------------------
def get_canonical_representation(block, rank=RANK):
    """
    Finds the absolute smallest byte-repr across all 8 geometric 
    symmetries and 8 Galois field automorphisms.
    """
    best_repr = None
    best_meta = (0, 0) # (symmetry_idx, frobenius_k)
    
    symmetries = get_block_symmetries(block)
    
    for s_idx, (sym_block, is_flipped) in enumerate(symmetries):
        # 1. SVD and Quantize this specific symmetry
        U, s, Vt = svd(sym_block, full_matrices=False)
        Uq = np.clip(np.round(U[:, :rank] * QUANT_SCALE), -128, 127).astype(np.int8)
        Vq = np.clip(np.round(Vt[:rank, :] * QUANT_SCALE), -128, 127).astype(np.int8)
        Sq = np.round(s[:rank]).astype(np.uint16)
        
        # 2. Pack into bytes
        packed = np.concatenate([
            (Uq.astype(np.int16) & 0xFF).astype(np.uint8).ravel(),
            Sq.astype('<u2').view(np.uint8).ravel(),
            (Vq.astype(np.int16) & 0xFF).astype(np.uint8).ravel()
        ])
        
        # 3. Find best Galois automorphism for THIS symmetry
        for k in range(FIELD_DEGREE):
            candidate = frobenius_vector(packed, k)
            cand_tuple = tuple(candidate.tolist())
            if best_repr is None or cand_tuple < best_repr:
                best_repr = cand_tuple
                best_meta = (s_idx, k)
                
    return np.array(best_repr, dtype=np.uint8), best_meta

def apply_inverse_symmetry(block, s_idx):
    """Undo the D4 symmetry transformation."""
    # This logic must mirror get_block_symmetries exactly
    # s_idx: 0,1=rot90/flip; 2,3=rot180/flip; 4,5=rot270/flip; 6,7=rot360/flip
    rotations = (s_idx // 2) + 1
    flipped = (s_idx % 2) == 1
    
    res = block
    if flipped:
        res = np.flipud(res)
    res = np.rot90(res, k=-rotations)
    return res

# ---------------------- Modified Compress/Decompress ------------------

def compress(img, verbose=False):
    from helpers import extract_blocks_padded # Assuming previous logic
    blocks, coords, padded_shape = extract_blocks_padded(img, BLOCK)
    
    dictionary = {}
    dict_list = []
    # codes: [dict_idx, symmetry_idx, frob_k]
    codes = np.zeros((len(blocks), 3), dtype=np.int32) 

    for idx, b in enumerate(blocks):
        canon, (s_idx, k) = get_canonical_representation(b, RANK)
        key = bytes(canon.tobytes())
        
        if key not in dictionary:
            dictionary[key] = len(dict_list)
            dict_list.append(canon)
            
        codes[idx] = [dictionary[key], s_idx, k]

    dict_array = np.stack(dict_list)
    meta = {'orig_h': img.shape[0], 'orig_w': img.shape[1], 
            'p_h': padded_shape[0], 'p_w': padded_shape[1],
            'block': BLOCK, 'rank': RANK}
            
    return meta, dict_array, codes

def decompress(meta, dict_array, codes):
    out = np.zeros((meta['p_h'], meta['p_w']), dtype=np.float32)
    idx = 0
    b_size = meta['block']
    rank = meta['rank']
    
    for i in range(0, meta['p_h'], b_size):
        for j in range(0, meta['p_w'], b_size):
            d_idx, s_idx, k = codes[idx]
            canon = dict_array[d_idx]
            
            # 1. Reverse Galois
            vec = frobenius_vector(canon, (-k) % FIELD_DEGREE)
            
            # 2. Unpack SVD
            p = 0
            U_sz = b_size * rank
            U_b = vec[p:p+U_sz].astype(np.int16); p += U_sz
            U_b[U_b >= 128] -= 256
            
            S_sz = 2 * rank
            S_b = vec[p:p+S_sz].tobytes(); p += S_sz
            Sr = np.frombuffer(S_b, dtype='<u2').astype(np.float32)
            
            V_sz = rank * b_size
            V_b = vec[p:p+V_sz].astype(np.int16)
            V_b[V_b >= 128] -= 256
            
            U = (U_b.reshape(b_size, rank) / QUANT_SCALE)
            Vt = (V_b.reshape(rank, b_size) / QUANT_SCALE)
            
            # 3. Reconstruct block and Reverse Symmetry
            block_recon = (U * Sr[np.newaxis, :]) @ Vt
            block_final = apply_inverse_symmetry(block_recon, s_idx)
            
            out[i:i+b_size, j:j+b_size] = block_final
            idx += 1
            
    return out[:meta['orig_h'], :meta['orig_w']]
