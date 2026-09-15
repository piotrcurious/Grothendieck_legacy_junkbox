#!/usr/bin/env python3
"""
SVD + D4 Symmetry + Galois Orbit Compressor (py02 / orbit_comp1 wrapper)
"""
from orbit_comp import (
    load_pgm, save_pgm, compress, decompress, compute_metrics,
    get_canonical_representation, get_block_symmetries, apply_inverse_symmetry,
    extract_blocks_padded, frobenius_vector, make_frobenius_table, gf_mul,
    main
)

if __name__ == '__main__':
    main()
