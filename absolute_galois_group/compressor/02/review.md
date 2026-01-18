# Quick verdict

No — this is not a verbatim copy of a single, well-known algorithm. It is a *hybrid* that recombines several established ideas from image coding:

* **Quadtree partitioning** (variance-based split) — classic.
* **Block canonicalization → fixed 8×8 representation + vector quantization (VQ)** — standard VQ/K-means approach.
* **Gain–shape (separate energy + normalized shape)** — very similar in spirit to Perceptual Vector Quantization (PVQ) / gain-shape schemes used in modern codecs.
* **Dictionary learning via k-means** — related to K-means / K-SVD style dictionary learning (but K-SV D is different in training).
* **SoA streams + zlib** — engineering choices for entropy layering.

So it’s an original assembly of known building blocks rather than a straight copy of one single published algorithm. If you want a precise match, it’s closest in concept to “quadtree + block VQ with gain–shape quantization” and to ideas used in PVQ (Daala/AV1), quadtree VQ research, and dictionary-based compression literature.

---

# Concrete issues, bugs, and fragile points (what I'd fix first)

1. **Random sampling uses `rand()`**
   `rand() % 2` and `rand()` for initial centroids is unsound for reproducibility/quality. Use `std::mt19937` and uniform distributions (you already use `mt19937` later).

2. **K-means assignment uses max dot product, not Euclidean distance**
   You do `(dictionary.transpose() * samples[v]).maxCoeff(&best_k)` which finds the atom with maximum correlation — this is fine *if* dictionary columns and samples are normalized (you do normalize). But be explicit: either use cosine similarity or compute squared Euclidean distances. Current code mixes normalization and max-dot implicitly; document this or change to a proper distance for clarity.

3. **Mean / offset quantization is lossy and clamped unexpectedly**
   You store `beta` (mean) as `uint8_t` clamped to [0,255]. If image pixel range or block mean can be outside that (e.g. after normalization or other transforms), you lose sign / range. Better to store signed integers (int16_t) and encode scale/range metadata.

4. **Scale quantization mapping is crude**
   You quantize `alpha` as `uint8_t` via `alpha*0.5f`. This bounds the dynamic range arbitrarily; add an explicit quantizer and parameterize it (and write quantization params to header).

5. **Header lacks format/version metadata**
   You write width/height and then the dictionary raw floats but never include a version magic, endianness marker, or DICT_SIZE/CANONICAL_SIZE in header. Make the file format explicit so readers are robust to future changes.

6. **Packed bool vector handling**
   You pack `split_tree` into bytes — fine — but you do not store the number of leaves explicitly (you infer it by counting zero bits on decompression). That works but is brittle if tree serialization ever changes; consider storing `leaves` explicitly.

7. **`canonicalize` sampling method is nearest-sample, not true averaging**
   The resize uses nearest sampling of block centers, not area average or bilinear. That produces aliasing for odd block sizes. Use bilinear or box-filter averaging for downsampling.

8. **`reconstruction_map` uses nearest neighbor**
   Same as above — you comment bilinear but implemented nearest. For quality use bilinear interpolation to map 8×8 → arbitrary size.

9. **Potential Eigen assignment / Ref size mismatch**
   `img_block = reconstruction_map(vec, h, w);` assigns a temporary to an `Eigen::Ref<MatrixXf>` subblock. That generally works but can trigger evaluation/copies; test for non-square odd sizes and check Eigen assertions in debug builds.

10. **I/O: write/read counts use implicit sizes**
    Calls like `os.write((char*)data.indices.data(), data.indices.size());` rely on `size()` being in bytes (ok for `uint8_t`) but are less explicit. Prefer `data.indices.size() * sizeof(decltype(data.indices)::value_type)` for clarity.

11. **No error handling on decompression read sizes**
    If the file is truncated, `is.read()` may silently fail. Check `is.good()` and sizes.

12. **Single-threaded K-means updates are using `#pragma omp critical` for a full loop**
    That serializes the accumulation step heavily. Consider `#pragma omp atomic` on counts and reduce centroids with OpenMP reductions (or accumulate into per-thread buffers then merge once outside critical).

---

# Suggested improvements (algorithmic & engineering)

* **Write a small file-format header**: magic bytes, version, CANONICAL_SIZE, DICT_SIZE, endianness, quantization parameters. Makes the format robust.

* **Use proper quantizers for scale/offset**: store scale as quantized mantissa+exponent or use log-scale quantization so dynamic range is covered.

* **Use bilinear (or bicubic) resampling** for canonicalize and reconstruction_map to reduce artifacts.

* **Consider K-SVD or online dictionary learning** if you want sparser, better atoms. K-means is a classical starting point but K-SVD yields more expressive dictionaries.

* **Consider sparse coding / orthogonal matching pursuit** for better fidelity given a dictionary, or use PVQ-style coding if perceptual quality is target.

* **Entropy coding**: zlib is OK for a prototype. For better compression, replace zlib with arithmetic/coder or context adaptive coder (CABAC/ANS) and use appropriate contexts for indices/scales/offsets.

* **Rate-distortion control**: currently split threshold is variance only. Add target bitrate control, RD cost for splitting, and quantizer step size adaptation per level.

* **Parallelism**: vectorize K-means initialization/assignment with BLAS/Eigen optimizations; use proper thread reductions.

* **Safety & portability**: use fixed-width integer types for on-disk storage and document endianness.

---

# Code-style / small fixes

* Replace `rand() % 2` with `std::bernoulli_distribution` + `mt19937`.
* Use `std::clamp` call sites consistently (C++17) and include `<algorithm>` (already included).
* When writing binary buffers, pass explicit byte counts: `os.write((char*)data.indices.data(), data.indices.size()*sizeof(uint8_t));`
* Add checks after `read()` to validate expected bytes were read.

---

# Final recommendation

Your implementation is a creative, coherent hybrid — not a line-by-line copy of a single published codec. If you plan to publish or commercialize it:

1. **Cite related work**: quadtree VQ papers, PVQ/Daala (if you borrow gain–shape ideas), and dictionary learning literature (K-SVD).
2. **Harden the format and quantizers** (so others can interoperate).
3. **Improve resampling and quantization** for visual quality.

If you want, I can:

* produce a short patch with the most important bugfixes (random→mt19937, stronger header, improved packing checks, bilinear sampling),
* or produce a short design note listing which papers to read to justify/compare your approach (PVQ, K-SVD, quadtree VQ, fractal compression).

Which of those would help you next?
