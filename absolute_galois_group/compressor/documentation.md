# Compressor Versions Documentation

This document details the different versions of the compressor in the `absolute_galois_group/compressor` directory, highlighting the unique features of each version and comparing them where necessary.

## C++ Version `01`

### Overview

Version `01` is the foundational C++ implementation of the compressor. It uses a fixed-block size and a dictionary-learning approach to compress images. This version is a practical and straightforward implementation of vector quantization.

### Core Features

*   **Fixed-Block Processing**: The compressor divides the input image into non-overlapping 8x8 blocks. This fixed-size approach simplifies the processing pipeline.

*   **Dictionary Learning with K-Means**: The core of this compressor is a dictionary of "representative" blocks (or "codebook vectors"). This dictionary is learned from the input image using the K-Means clustering algorithm. Each block in the dictionary is an 8x8 matrix of pixels.

*   **Block Representation**: Each 8x8 block in the original image is represented by three components:
    1.  **Basis Index**: An index pointing to the most similar block in the learned dictionary.
    2.  **Scale**: A floating-point value representing the contrast of the block.
    3.  **Offset**: A floating-point value representing the brightness of the block.

*   **Compression Process**: The compression process consists of the following steps:
    1.  The image is loaded and divided into 8x8 blocks.
    2.  A dictionary of 256 representative blocks is learned from the image using K-Means.
    3.  For each block in the image, the best-matching dictionary entry is found, and the scale and offset are calculated.
    4.  The dictionary, along with the index, quantized scale, and offset for each block, is written to a compressed file using zlib.

*   **Decompression Process**: Decompression reverses the process:
    1.  The dictionary and the encoded block data are read from the compressed file.
    2.  For each block, the corresponding dictionary entry is retrieved, and the original block is reconstructed by applying the stored scale and offset.
    3.  The reconstructed blocks are reassembled to form the final image.

### Code Structure

The implementation is contained in a single file, `shared_comp.cpp`. It uses the Eigen library for linear algebra operations and Boost for zlib compression.

### Summary

Version `01` is a solid baseline for the compressor. Its main limitation is the fixed-block-size approach, which does not adapt to the complexity of the image. This can lead to inefficient compression in areas of low detail and artifacts in areas of high detail. This version serves as a starting point for the more advanced versions that follow.

## C++ Version `02`

### Overview

Version `02` represents a significant architectural evolution from the fixed-block approach of its predecessor. It introduces a **quadtree decomposition** mechanism, allowing for **adaptive block sizes**. This enables the compressor to allocate more data to complex regions of an image and less to simpler areas, leading to more efficient compression.

### Core Features

*   **Adaptive Quadtree Decomposition**: The image is no longer split into a uniform grid. Instead, it's treated as a single root block that is recursively split into four quadrants based on its variance (a measure of detail).
    *   If a block's variance is above a certain threshold and it's larger than a minimum size, it is split.
    *   If the variance is low, it is not split, becoming a "leaf" node in the tree. This allows large, smooth areas (like skies) to be represented by a single large block, saving a significant amount of data.

*   **Scale-Invariant Dictionary**: To handle the variable block sizes, a resampling step is introduced. All blocks, regardless of their actual dimensions, are resized to a **canonical 8x8 representation** before being compared to the dictionary. This allows the dictionary to learn abstract shapes and textures that are independent of their scale in the image.

*   **Separated Data Streams (Structure-of-Arrays)**: Instead of storing the index, scale, and offset interleaved for each block, this version separates them into three distinct streams. This is a key optimization for the final zlib compression pass, as grouping similar data together (e.g., all brightness values) improves compression ratios.

*   **Bit-Packed Quadtree Structure**: The structure of the quadtree itself (which nodes are split and which are leaves) is stored efficiently as a stream of bits.

### Identified Flaws

While a major conceptual improvement, this version contained several bugs and design weaknesses, as highlighted in a detailed code review (`02/review.md`):

*   **Poor Resampling**: Both the downsampling (`canonicalize`) and upsampling (`reconstruction_map`) steps used nearest-neighbor sampling, a crude method that introduces significant aliasing and blocky artifacts.
*   **Unsound Randomization**: The dictionary learning process used `rand()`, which is not ideal for reproducible or high-quality results.
*   **Brittle File Format**: The compressed file format lacked essential metadata like a version number or explicit dictionary parameters, making it difficult to update the format in the future.
*   **Flawed Quadtree Logic**: The implementation did not correctly handle images with dimensions that weren't powers of two, leading to processing errors.

### Summary

Version `02` introduces a much more sophisticated and efficient compression strategy with its adaptive quadtree. However, the implementation flaws, particularly the poor-quality resampling, prevent it from achieving high-quality results. These issues are directly addressed in the subsequent `02_fix` versions.

## C++ Versions `02_fix` and `02_fix_crop`

### Overview

These versions are direct responses to the flaws identified in version `02`. They are incremental but critical updates that significantly improve the quality and robustness of the adaptive quadtree compressor.

### Core Fixes and Improvements (`02_fix`)

*   **Bilinear Interpolation**: The most important change is the replacement of nearest-neighbor resampling with **bilinear interpolation**. This is implemented in both the `canonicalize` (downsampling) and `reconstruction_map` (upsampling) functions. This dramatically reduces aliasing and blocky artifacts, resulting in a much higher-quality reconstructed image.

*   **Image Padding**: To correctly handle images with dimensions that are not powers of two, the compressor now pads the image to the next-largest power-of-two resolution before processing. This ensures the quadtree decomposition works correctly.

*   **Robust Quadtree Logic**: The logic for deciding whether to split a block is refined to be more robust, correctly handling edge cases and preventing infinite recursion.

*   **Improved File Format**: The compressed file format is improved to explicitly store the number of leaf nodes in the quadtree. This makes the decompression process less brittle.

### Final Refinement (`02_fix_crop`)

*   **Correct Cropping**: The `02_fix` version produced a decompressed image with the same padded dimensions as the compressed image, often resulting in black bars around the content. The `02_fix_crop` version addresses this by:
    1.  Storing the **original** image dimensions in the compressed file header.
    2.  During decompression, the image is first reconstructed to its full padded size and then **cropped back to its original dimensions**. This ensures the final output is pixel-perfect in size.

### Summary

The `02_fix` and `02_fix_crop` versions represent the maturation of the C++ compressor. By addressing the critical flaws of the previous version, they produce a robust, high-quality, adaptive compression system. The final `02_fix_crop` version is the most complete and practical of the C++ implementations.

## Python Version `py01`

### Overview

The `py01` version marks a radical conceptual departure from the C++ implementations. Instead of relying on geometric similarity (K-Means), this version introduces a highly academic and theoretically dense approach based on **Galois Theory**. The core idea is to define block equivalence not by visual appearance, but by membership in a **Galois conjugacy class** over a finite field.

### Core Features

*   **Finite Field Representation**: Each 8x8 image block is first decomposed using Singular Value Decomposition (SVD) into a low-rank approximation. The resulting SVD components (singular vectors and values) are quantized into bytes. This stream of bytes is then interpreted as a vector of elements in the finite field **GF(2^8)**, using the same irreducible polynomial as the AES encryption standard.

*   **Galois Conjugacy**: The absolute Galois group of GF(2^8) over GF(2) is generated by the **Frobenius automorphism** (φ(a) = a²). Repeated application of this automorphism to a vector generates its "Galois orbit," or conjugacy class. All vectors in an orbit are considered fundamentally equivalent from a field-theoretic perspective.

*   **Canonical Representation**: To compress the data, each block's byte vector is "canonicalized" by finding its **lexicographically smallest conjugate** in its Galois orbit. This canonical vector becomes the representative for that block.

*   **Compression Process**:
    1.  For each 8x8 block, compute its SVD representation and quantize it to a byte vector.
    2.  Find the canonical representative of this vector under the Frobenius automorphism, along with the power `k` of the automorphism needed to transform the canonical form back to the original.
    3.  A dictionary is built containing only the unique canonical representatives.
    4.  Each block is then encoded as a pair: `(dictionary_index, k)`.

### Summary

Version `py01` is a research-oriented prototype that explores a genuinely novel, algebraic approach to image compression. It defines block "similarity" in a way that is completely detached from visual perception, relying instead on the deep structure of finite fields. While fascinating, its practical compression performance is limited by the simplistic quantization and its failure to account for geometric symmetries (like rotations and flips), which are addressed in the next version.

## Python Version `py02`

### Overview

Version `py02` builds directly upon the Galois-theoretic foundation of `py01`, but addresses its most significant limitation: the lack of geometric awareness. This version combines the algebraic canonicalization of the previous version with geometric canonicalization, creating a hybrid system that searches for a "truer" canonical representation across a much larger equivalence class.

### Core Features

*   **Combined Geometric and Algebraic Symmetries**: The core innovation is to consider both geometric and algebraic symmetries simultaneously. For each 8x8 image block, the compressor generates all 8 geometric symmetries (rotations and flips) belonging to the dihedral group D4.

*   **Expanded Canonical Search**: The canonical representation for a block is now the one that is lexicographically smallest across all 64 possible transformations (8 geometric symmetries × 8 Galois automorphisms).

*   **Compression Process**:
    1.  For a given 8x8 block, generate all 8 of its geometric symmetries.
    2.  For each of these 8 symmetric blocks, perform the SVD reduction and quantization as in the previous version.
    3.  For each of the resulting 8 byte vectors, find the lexicographically smallest Galois conjugate.
    4.  Compare these 8 canonical Galois forms and select the one that is the absolute lexicographically smallest.
    5.  The block is then encoded as a triplet: `(dictionary_index, symmetry_index, frobenius_power_k)`.

*   **Improved Quantization**: This version also refines the quantization process, using 16 bits for singular values, which improves fidelity.

### Summary

Version `py02` represents a more complete and powerful implementation of the Galois-theoretic compression concept. By treating geometric and algebraic symmetries as part of a single, larger group of transformations, it can find more fundamental redundancies and achieve better compression. This hybrid approach is a sophisticated method for finding the "essence" of a block, invariant under both visual and algebraic transformations.

## Final Comparison: C++ vs. Python Versions

### Overview

The C++ and Python versions of this compressor represent two fundamentally different philosophies for achieving data compression. The C++ track is an exercise in engineering a practical, high-quality, adaptive image codec. The Python track is a research-oriented exploration of a novel, highly abstract, and theoretically-driven compression paradigm.

### C++ Approach: Practical, Adaptive, and Perceptual

*   **Core Idea**: Reduce redundancy by identifying and sharing visually similar image patches. This is a form of **Vector Quantization**.
*   **Evolution**: The C++ versions show a clear, iterative path toward a better practical codec:
    1.  Start with a simple, fixed-block K-Means approach (`01`).
    2.  Introduce an adaptive quadtree to handle varying image complexity (`02`).
    3.  Fix critical implementation bugs (resampling, padding) to dramatically improve visual quality (`02_fix`, `02_fix_crop`).
*   **Strengths**:
    *   **High Quality**: The final version is capable of producing high-quality images.
    *   **Efficiency**: The adaptive nature of the quadtree is highly efficient, spending bits on complex areas and saving them on simple ones.
    *   **Practicality**: The approach is well-understood, robust, and directly targets perceptual redundancy.
*   **Weaknesses**:
    *   **Heuristic**: The "similarity" is based on a heuristic (variance) and a clustering algorithm (K-Means), which may not be optimal.

### Python Approach: Theoretical, Algebraic, and Abstract

*   **Core Idea**: Reduce redundancy by identifying and sharing algebraically equivalent structures, defined by Galois field theory and geometric symmetries.
*   **Evolution**: The Python versions explore the depth of this algebraic idea:
    1.  Establish the core concept of Galois conjugacy as the basis for equivalence (`py01`).
    2.  Expand the notion of equivalence to include geometric symmetries, creating a more powerful canonicalization (`py02`).
*   **Strengths**:
    *   **Theoretical Novelty**: This is a genuinely unique approach to compression, rooted in deep mathematical concepts.
    *   **Fundamental**: It attempts to find a more fundamental, "absolute" representation of a block, invariant to a wide range of transformations.
*   **Weaknesses**:
    *   **Lower Perceptual Quality**: Because the equivalence is not based on visual similarity, the reconstructed images can have more artifacts.
    *   **Complexity**: The concepts are highly abstract and the implementation is less straightforward.
    *   **Less Practical**: The approach is more of a research prototype than a production-ready codec.

### Conclusion

The two tracks serve different purposes. The **C++ versions** provide a blueprint for a practical, high-performance image compressor. The **Python versions**, on the other hand, serve as a fascinating exploration into the application of abstract algebra to signal processing, offering a glimpse into a completely different paradigm for data compression.
