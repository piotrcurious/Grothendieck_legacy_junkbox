# Instructions for Compiling and Testing the LFSR Suite

This directory contains an improved Linear Feedback Shift Register (LFSR) suite.

## Dependencies

The code requires the **NTL (Number Theory Library)** and **GMP (GNU Multi-Precision Library)**.

## Compilation

To compile the code, use `g++` and link against `ntl` and `gmp`. If you have installed these libraries in a custom location (e.g., `/home/jules/local`), you need to provide the include and library paths:

```bash
g++ -O3 -std=c++17 lfsr_improved.cpp -o lfsr_improved \
    -I/path/to/local/include \
    -L/path/to/local/lib \
    -lntl -lgmp -pthread
```

## Running

When running the executable, ensure that the dynamic linker can find the libraries:

```bash
LD_LIBRARY_PATH=/path/to/local/lib ./lfsr_improved
```

## Features

- **Improved Inference:** Uses an $O(n)$ string parser to recover the primitive element from observed sequences, significantly faster than the previous $O(2^w)$ approach.
- **Range Traversal:** A `RangeTraverser` class that can visit all numbers in a range `[0, max_val]` exactly once in a pseudo-random order.
- **Reproducibility:** Supports optional seeding for deterministic sequences.
- **Robustness:** Handles edge cases such as `max_val = 0` and large ranges.

## Testing

The `main()` function in `lfsr_improved.cpp` includes several test cases:
1. **Inference Test:** Verifies that the correct field width can be inferred from a short prefix.
2. **Range Traversal Tests:** Verifies uniqueness and completeness for various range sizes (20, 100, 1000, 0).
3. **Reproducibility Test:** Verifies that two traversers with the same seed produce the same sequence.
