# Finite Field & RNN Compressor

This project implements experimental data compression techniques that combine finite field arithmetic with statistical modeling (RNNs). It is part of the Absolute Galois Group explorations.

## Key Concepts
- **Finite Field Arithmetic**: Mapping data to elements of $\mathbb{F}_p$ to leverage algebraic structures for compression and manipulation.
- **RNN State Machines**: Using Recurrent Neural Networks to predict data sequences and only storing the residuals (errors) in the compressed format.
- **Multidimensional Support**: Handling vectors of finite field elements for complex data structures.

## Files
- `compressor.ino`: Arduino implementation of basic compression concepts.
- `compressor1.py`: Python-based explorations of these algorithms.
- `transformation_params.h`: Shared parameters for transformations.
- `compressor1.md` through `compressor7b.md`: Detailed walkthroughs, code examples, and theoretical justifications for the approach.
- `01/`, `02/`, `02_fix/`, `02_fix_crop/`, `py01/`, `py02/`: Various iterations and experiments in C++ and Python exploring quadtree decomposition, morphism-based compression, and dictionary learning.

## State
Conceptual prototype. Demonstrates the integration of neural sequence prediction with algebraic field theory for non-traditional compression.
