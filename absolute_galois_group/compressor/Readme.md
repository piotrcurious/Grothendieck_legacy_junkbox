You are absolutely correct. My previous example performed **local** algebraic reduction (SVD per block). While mathematically elegant, it fails to exploit **global** redundancy. In information theory terms, it reduced *spatial* redundancy but ignored *inter-block* statistical redundancy.

To achieve high compression ratios while adhering to your "Algebraic/Galois" theme, we must implement **Structure Sharing**.

### The Conceptual Shift: "Global Field Extensions"

Instead of treating every block as a unique universe, we assume there exists a finite set of "Fundamental Algebraic Structures" (a Dictionary or Basis) from which all other blocks are derived via simple group actions (scaling, shifting, or permuting).

1. **The "Base Field" (Dictionary):** We compute a global dictionary of representative block shapes (centroids).
2. **The "Galois Action" (Mapping):** We define an operation that maps a dictionary entry to an image block.
* *Math:* 
* We only store the index , the scalar , and the offset .



This transforms the problem from storing matrices to storing **pointers and group operations**, drastically increasing the compression ratio.

### Implementation: Algebraic Dictionary Coding (Vector Quantization)

This C++ system adds a **Learning Phase** to discover shared structures before compressing.

**Dependencies:** `Eigen3`, `Boost`.



### Why this achieves higher compression

1. **Structure Sharing (The Dictionary):**
We do not store pixels. We do not even store singular values per block. We store a **Reference** to a shared structure.
* *Previous Code:* Stored 2 floats + 2 vectors per block ~ (16-20 floats).
* *This Code:* Stores 1 byte (index) + 1 byte (scale) + 1 byte (offset) = **3 bytes per 8x8 block** (plus small dictionary overhead).


2. **Theoretical Link (Galois/Cohomology):**
* **The Dictionary** acts as the invariant ground field.
* **The Index** selects the specific field extension type.
* **Scale/Offset** act as the cohomology class (the specific "twist" required to align the ideal structure with the messy reality).


3. **Ratio Estimation:**
* Raw 8x8 block: 64 bytes.
* Compressed representation: ~3 bytes.
* **Ratio:** Approx **21:1** (before Zlib even touches it). Zlib will compress the stream of indices further, potentially reaching 30:1 or 40:1 depending on image repetition.



### Further Optimization (For Research Context)

If you were writing a paper on this, the next step would be **"Predictive Indexing"**:
The index of Block  is likely related to the index of Block  (spatial coherence). Instead of storing the raw index, you would store the *difference* in indices (or use a Markov chain to predict the next index), moving deeper into Group Cohomology by analyzing the "relations" between adjacent symmetries.
