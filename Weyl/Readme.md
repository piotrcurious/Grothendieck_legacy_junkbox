# Weyl Algebra and Quantized Field Solutions

This directory illustrates the concepts of Weyl algebra, non-commutativity, and quantized field solutions applied to image rotation and signal filtering.

## 🌀 Weyl Rotation (Python)

Located in `rotation/`, these scripts demonstrate how the non-commutativity of discrete translation operators (X) and phase operators (P) can be measured and used to improve the quality of discrete rotations.

- **`rotation/Weyl_Rotation_Consolidated.py`**: The primary implementation featuring:
  - `ModularWeylXOperator`: Discrete row shift operator.
  - `ModularWeylPOperator`: Discrete clock (phase) operator. Satisfies $[P, X] \neq 0$.
  - `LemmaBasedCorrectionStrategy`: Uses the commutator norm $Q = \|[P, X]\|$ for uniform scaling.
  - `WeylAlgebraBasedCorrectionStrategy`: Incorporates the commutator structure into the correction matrix.
  - `SymplecticCorrectionStrategy`: A phase-aware correction based on the symplectic structure of the commutator.
- **`rotation/test_weyl_rotation.py`**: Unit tests for the rotation operators and strategies.

## 📡 Weyl Filter (Arduino/ESP32)

Located in the root `Weyl/` directory, these files implement a frequency-domain filter based on quantized field equations.

- **`Weyl_Filter_Consolidated.ino`**: Main Arduino sketch for ESP32.
- **`Weyl_Filter_Utils.h`**: Core fixed-point arithmetic (Q16) and field operator logic. Features:
  - Second-order finite difference (4-point central) for field gradients.
  - `FieldConfig` struct for easy parameter tuning.
- **`test_weyl_filter.cpp`**: Host-side C++ test suite to verify the filtering logic.

### Concepts
The filter applies a regularized field operator in the frequency domain:
$P(F) = F - D + F \cdot (\nabla V)$
where $D$ is the frequency gradient and $V(x)$ is a field potential.

## 📁 Cleanup Note
Older versions of filters (`filter*.ino`) and rotation demos (`demo*.py`) are archived in `old/` subdirectories for historical reference.
