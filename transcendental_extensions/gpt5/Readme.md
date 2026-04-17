# Symbolic Floating-Point Unit (SFPU) - GPT-5 Project

This project explores the concept of a "next-generation FPU" that understands symbolic mathematical entities directly in hardware. By extending the standard IEEE 754 floating-point representation with symbolic tags, we can perform exact symbolic arithmetic on transcendental constants like $\pi$ and $e$, maintaining perfect precision and accelerating complex mathematical workloads.

## The Core Innovation: Symbolic-Numeric Hybrid (SNH)

The key innovation is the **SNH format**, a 72-bit word structured as follows:

`{ Type [3:0], Symbol [3:0], Coefficient [63:0] }`

*   **Type [3:0]**: Defines the nature of the number (NORMAL, SYMBOLIC, HYBRID, ZERO, INF, NaN).
*   **Symbol [3:0]**: Identifies the transcendental constant (PI, E, I, LOG2).
*   **Coefficient [63:0]**: A standard IEEE 754 double-precision floating-point number.

This format allows representing expressions like $2\pi$ exactly as `{Type: HYBRID, Symbol: PI, Coefficient: 2.0}`.

## Project Structure

### C++ Symbolic Engines
A series of C++ implementations demonstrating the evolution of the symbolic engine:
- `1.cpp`: Basic symbolic expression tree with constant folding.
- `2.cpp`: Introduction of morphisms (unary functions) and named constants.
- `3.cpp`: Advanced rewrite rules and recursive simplification.
- `4.cpp`: Regex-based symbolic simplification.
- `5.cpp`: Tree-based pattern matching and rule substitution.
- `6.cpp`: N-ary associative/commutative engine with sequence wildcards and interning.

### Verilog SFPU
- `8.verilog`: A synthesizable Verilog implementation of the Symbolic FPU. It implements hardware "rewrite rules" to handle symbolic operations in a single cycle where possible.
- `tb_sfpu.verilog`: A testbench for verifying the SFPU's symbolic logic.

### Documentation & Rationale
- `rationale.md`: Discusses the historical context of the x86 FPU and the missed mathematical progress over the decades.
- `Readme.md`: This file.

## Verification

To run the automated test suite for the C++ symbolic engines:

```bash
python3 run_all_tests.py
```

## Future Advancements
- **Instruction Fusion**: Recognition of patterns like `FLOG(FEXP(x))` at the pipeline level to eliminate redundant operations.
- **Arbitrary Precision**: Hardware acceleration for multi-precision symbolic entities.
- **Interval Arithmetic**: Native support for interval types to track numerical error bounds.
- **AI-Infused Optimization**: Dynamic learning of new symbolic rules based on execution profiles.

This project moves beyond simple numerical calculation and takes a step toward a processor that has a deeper, more intrinsic "understanding" of mathematics.
