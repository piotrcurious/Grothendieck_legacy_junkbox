Nice — this is a fun (and very solvable) place to squeeze a lot more compression out of the pipeline. Below I do three things:

1. Explain the main algorithmic weaknesses and the high-level fixes.


2. Give a concrete, efficient replacement for ManifoldDictionary::train() that accounts for pattern frequency, vectorizes correlation calculations with Eigen, and precomputes geometric transforms so we stop redoing transforms in hot loops.


3. List further optimisations and tuning knobs you can try.



I kept the code compact but practical so you can drop it into your project (you'll still need your existing helper functions like resize_matrix and CryptoMorphismEngine::apply). The key idea: extract canonical patch vectors, group identical/nearby canonical patterns, build dense candidate × tile correlation via matrix multiplication, compute expected residual reduction in one vectorized sweep, then greedily pick best candidates (weighted by frequency) and update residuals only where they help.


---

What's wrong now (short)

You treat every extracted patch as independent candidate; frequency of repetition is ignored.

You compute transforms and per-tile correlations in nested loops — repeated work.

You evaluate candidate usefulness by iterating over tiles per candidate; this can be vectorized.

You use only non-overlapping tiles everywhere — that loses many repeat occurrences. (I make stride configurable.)

You normalize and resize repeatedly inside hot loops.


Main improvements applied

Extract patches with configurable stride (allow overlap).

Canonicalize and normalize patches into flattened vectors (dim = CANONICAL_SIZE*CANONICAL_SIZE).

Hash / quantize canonical vectors to group repeated patterns and compute frequency.

Build tiles_mat (dim × T) and candidate_mat (dim × C') where C' includes geometric transforms (16×).

Compute correlations matrix corr = candidate_mat.transpose() * tiles_mat (C' × T) via one Eigen multiplication.

Compute total reduction per candidate as sum_t before_err[t] * corr[c,t]^2 (vectorized).

Score candidates by reduction * frequency_factor and pick top candidates.

Update residual map only for tiles where candidate gives meaningful improvement.



---

Replacement train() (drop-in style)

This is a single-method replacement for ManifoldDictionary::train() plus a small helper struct. It assumes:

resize_matrix(const MatrixXf&, int out_side) exists.

CANONICAL_SIZE constant exists.

best_approximation remains available (but I show a faster vectorized version idea later).

MatrixXf and VectorXf are Eigen types, and using namespace std; is okay in scope.


// Add these includes at top of file if not already present:
#include <unordered_map>
#include <sstream>
#include <iomanip>

// helper: quantize a normalized vector to a compact string key
static std::string quantize_key(const Eigen::VectorXf &v, int bits = 6) {
    // bits per coefficient: 6 -> 64 levels
    const float levels = float(1 << bits);
    std::string s;
    s.reserve(v.size());
    for (int i = 0; i < v.size(); ++i) {
        int q = int(std::round((v[i] + 1.0f) * 0.5f * (levels - 1))); // map [-1,1] -> [0,levels-1]
        if (q < 0) q = 0;
        if (q >= (1<<bits)) q = (1<<bits)-1;
        s.push_back(static_cast<char>(q));
    }
    return s;
}

void train(const MatrixXf &data, int block_size, int max_entries, float min_var) {
    atoms.clear();
    if (data.rows() < block_size || data.cols() < block_size) return;

    const int stride = std::max(1, block_size / 2); // allow overlap; tuneable
    const int dim = CANONICAL_SIZE * CANONICAL_SIZE;

    // Stage 1: extract tiles (overlapping) and compute canonical normalized vectors
    struct Tile { VectorXf vec; float before_err; int r,c; MatrixXf original_centered; };
    std::vector<Tile> tiles;
    tiles.reserve((data.rows()/stride) * (data.cols()/stride));

    for (int r = 0; r <= data.rows() - block_size; r += stride) {
        for (int c = 0; c <= data.cols() - block_size; c += stride) {
            MatrixXf b = data.block(r, c, block_size, block_size);
            float mu = b.mean();
            MatrixXf centered = b.array() - mu;
            const float variance = centered.squaredNorm() / float(block_size * block_size);
            if (variance < min_var) continue;
            MatrixXf canonM = resize_matrix(centered, CANONICAL_SIZE); // CANONICAL_SIZE x CANONICAL_SIZE
            VectorXf vec(dim);
            Eigen::Map<VectorXf>(vec.data(), dim) = Eigen::Map<VectorXf>(canonM.data(), dim);
            float n = vec.norm();
            if (n < 1e-6f) continue;
            vec /= n;
            float before_err = centered.squaredNorm();
            tiles.push_back(Tile{vec, before_err, r, c, std::move(centered)});
        }
    }
    if (tiles.empty()) return;

    // Stage 2: group identical/nearby canonical patterns by quantized key -> compute representative
    std::unordered_map<std::string, std::vector<int>> groups;
    groups.reserve(tiles.size()*2);
    for (int i = 0; i < (int)tiles.size(); ++i) {
        std::string key = quantize_key(tiles[i].vec, 6); // 6 bits per coeff -> adjustable
        groups[key].push_back(i);
    }

    // Build group representatives (mean of group vectors), plus frequency
    struct Rep { VectorXf vec; float freq; float avg_var; };
    std::vector<Rep> reps;
    reps.reserve(groups.size());
    for (auto &kv : groups) {
        const auto &idxs = kv.second;
        VectorXf meanv = VectorXf::Zero(dim);
        float var_sum = 0.f;
        for (int idx : idxs) { meanv += tiles[idx].vec; var_sum += tiles[idx].before_err; }
        meanv /= float(idxs.size());
        float n = meanv.norm();
        if (n < 1e-6f) continue;
        meanv /= n;
        reps.push_back(Rep{meanv, float(idxs.size()), var_sum / float(idxs.size())});
    }
    if (reps.empty()) return;

    // Stage 3: expand representatives by geometric transforms (precompute all 16 transforms)
    // candidate_mat: dim x C' where C' = reps.size() * 16
    const int G = 16;
    const int Cprime = int(reps.size()) * G;
    MatrixXf candidate_mat(dim, Cprime);
    std::vector<std::pair<int,uint16_t>> candidate_info; candidate_info.reserve(Cprime);
    int col = 0;
    for (int ri = 0; ri < (int)reps.size(); ++ri) {
        // convert rep.vec back to CANONICAL_SIZE x CANONICAL_SIZE matrix for transforms
        MatrixXf repM(CANONICAL_SIZE, CANONICAL_SIZE);
        Eigen::Map<VectorXf>(repM.data(), dim) = reps[ri].vec;
        for (uint16_t m = 0; m < G; ++m) {
            MatrixXf transformed = CryptoMorphismEngine::apply(repM, m); // returns CANONICAL_SIZE matrix
            VectorXf tv(dim);
            Eigen::Map<VectorXf>(tv.data(), dim) = Eigen::Map<VectorXf>(transformed.data(), dim);
            float n = tv.norm(); if (n < 1e-6f) { tv.setZero(); }
            else tv /= n;
            candidate_mat.col(col) = tv;
            candidate_info.emplace_back(ri, m); // mapping back to rep index and morphism
            ++col;
        }
    }

    // Stage 4: build tiles matrix (dim x T)
    const int T = int(tiles.size());
    MatrixXf tiles_mat(dim, T);
    VectorXf before_errs(T);
    for (int t = 0; t < T; ++t) {
        tiles_mat.col(t) = tiles[t].vec;
        before_errs[t] = tiles[t].before_err;
    }

    // Stage 5: compute correlations matrix in one go: C' x T
    // corr = candidate_mat.transpose() * tiles_mat
    MatrixXf corr = candidate_mat.transpose() * tiles_mat; // size C' x T
    // compute squared absolute correlations (we treated normalized vectors, so dot in [-1,1])
    MatrixXf corr2 = corr.array().square();

    // Stage 6: expected total reduction per candidate: reductions[c] = sum_t corr2[c,t] * before_errs[t]
    VectorXf reductions = corr2 * before_errs; // C' x 1

    // Stage 7: compute a score that multiplies by frequency of the representative (so repeated patterns get preferred)
    VectorXf freqs(Cprime);
    for (int c = 0; c < Cprime; ++c) {
        int rep_idx = candidate_info[c].first;
        freqs[c] = reps[rep_idx].freq;
    }
    VectorXf score = reductions.array() * freqs.array(); // simple scoring: reduction weighted by frequency

    // Stage 8: pick top K candidates by score
    struct ScoreIdx { float score; int idx; };
    std::vector<ScoreIdx> scored;
    scored.reserve(Cprime);
    for (int i = 0; i < Cprime; ++i) scored.push_back({score[i], i});
    std::sort(scored.begin(), scored.end(), [](const ScoreIdx &a, const ScoreIdx &b){ return a.score > b.score; });

    // reserve atoms (we still store canonical reps only)
    atoms.reserve(max_entries);
    int added = 0;
    VectorXf used_tiles(T); used_tiles.setZero(); // to avoid double-counting tiles already well-approximated

    for (auto &si : scored) {
        if (added >= max_entries) break;
        if (si.score <= 0) break;

        int cand_idx = si.idx;
        int rep_idx = candidate_info[cand_idx].first;
        uint16_t morphism = candidate_info[cand_idx].second;

        // we will add the representative (canonical) if it is sufficiently diverse vs existing atoms
        VectorXf candvec = candidate_mat.col(cand_idx);
        bool is_diverse = true;
        for (const auto &existing : atoms) {
            // compute correlation between existing atom (canonical rep) and this candidate representative
            VectorXf existing_vec(dim);
            Eigen::Map<VectorXf>(existing_vec.data(), dim) = Eigen::Map<VectorXf>(existing.data(), dim);
            float sim = std::abs(existing_vec.dot(candvec));
            if (sim > 0.92f) { is_diverse = false; break; } // threshold tighter
        }
        if (!is_diverse) continue;

        // push canonical rep (not transformed) into atoms
        MatrixXf canonicalM(CANONICAL_SIZE, CANONICAL_SIZE);
        Eigen::Map<VectorXf>(canonicalM.data(), dim) = reps[rep_idx].vec;
        atoms.push_back(canonicalM);
        ++added;

        // Now update residual: find tiles where this candidate gives good reduction and subtract approximation
        // For each tile t, the best transform among the G transforms of this rep is at indexes cand_base..cand_base+G-1
        int base = rep_idx * G;
        // compute per-tile best corr^2 across the G transforms
        VectorXf best_corr2 = VectorXf::Zero(T);
        VectorXf best_corr_signed = VectorXf::Zero(T);
        for (int g = 0; g < G; ++g) {
            int ci = base + g;
            VectorXf corr_col = corr.row(ci).transpose(); // T x 1
            // for sign we can use corr (not squared) for scaling later
            for (int t = 0; t < T; ++t) {
                float cabs2 = corr_col[t]*corr_col[t];
                if (cabs2 > best_corr2[t]) {
                    best_corr2[t] = cabs2;
                    best_corr_signed[t] = corr_col[t];
                }
            }
        }
        // apply update only on tiles where improvement is meaningful and tile not already well-approximated
        const float corr_threshold = 0.18f;
        for (int t = 0; t < T; ++t) {
            if (best_corr2[t] < corr_threshold*corr_threshold) continue;
            if (used_tiles[t] > 0.9f) continue; // skip tiles already consumed enough (tunable)
            // reconstruct approximation (resize canonical rep to block_size, scale by best_corr_signed*norm)
            MatrixXf atom_resized = resize_matrix(canonicalM, block_size); // block_size x block_size
            float atom_norm = Eigen::Map<VectorXf>(atom_resized.data(), block_size*block_size).norm();
            if (atom_norm < 1e-6f) continue;
            float scale = best_corr_signed[t] * std::sqrt(tiles[t].before_err); // approximate scale
            MatrixXf approx = atom_resized * (scale / atom_norm);
            // subtract from residual_map (use original tile coordinates)
            int rr = tiles[t].r, cc = tiles[t].c;
            // safety: clamp indices
            if (rr >= 0 && rr + block_size <= data.rows() && cc >= 0 && cc + block_size <= data.cols()) {
                // Update the underlying residual (we only have input 'data' const here; in your class keep residual_map as member)
                // For demonstration, you would do: residual_map.block(rr, cc, block_size, block_size) -= approx;
            }
            used_tiles[t] = std::min(1.0f, used_tiles[t] + best_corr2[t]); // mark as used
        }
    } // end picking
}

Notes about the snippet

I left the actual writeback to residual_map commented where you must plug in your residual buffer. (In your original code you had MatrixXf residual_map = data; — keep that and then perform the same subtractions.)

The quantization key (6 bits) is a tradeoff between grouping too much vs too little; tune to your data.

The scoring reductions * freq is intentionally simple: it prefers candidates that explain lots of energy and appear many times. You can use other formulas like reductions * log(1 + freq) or include avg_var to prioritize high-variance patterns.

I compute expected reduction as before_err * corr^2 and sum over tiles; this follows your earlier math but is vectorized.



---

Faster solve_sparse() idea

Precompute for each stored atom the 16 geometric transforms, flatten+normalize them, and put them into a atom_transforms matrix: dim x (atoms.size()*G).

For a target tile build r_canon vector and compute one matrix-vector multiply: corrs = atom_transforms.transpose() * r_canon (size atoms*G).

Choose the top correlation index (abs) — you can find the top K with a partial selection (std::nth_element) or argmax.

This is much faster than looping over atoms×morphisms.



---

Further optimizations & tuning

1. Use BLAS/LAPACK-backed Eigen and compile with -O3 -march=native — the dense matmul is crucially fast when accelerated.


2. Reduce canonical dimension: smaller CANONICAL_SIZE (e.g. 4×4 → 16 dims) is much faster and often sufficient to capture structure.


3. Use PCA / random projection on canonical vectors to reduce dim before grouping and correlation.


4. Use int8 quantization for tiles_mat and candidate_mat and perform approximate dot products if memory/cache is limiting.


5. Maintain a residual energy map and ignore tiles with residual energy below threshold (saves computations).


6. Early stop: after many atoms added, stop if total unexplained energy drops below a cut.


7. Parallelize: building tiles and computing candidate_mat transforms and even block updates are embarrassingly parallel.


8. Smart updates: when subtracting approximations, update only tiles where corr^2 * before_err > some absolute threshold to avoid overfitting tiny noise.


9. Cache transforms for each rep once — I precompute them in candidate_mat so later we can reuse.


10. Storage: store atoms in canonical form; during encoding store atom_id + morphism + scale.




---

Quick performance expectations

If your image has many repeating blocks, grouping by quantized canonical patterns reduces candidate count drastically (often >10× less work).

The heavy cost becomes one dim × T matmul and one dim × C' matmul (or their transpose forms) — dense BLAS does that very efficiently.

Precomputing transforms reduces per-candidate loops (previously 16 transforms × T × candidates).



---

If you want, I can:

convert the above into a full working .cpp function that plugs into your exact class (including residual_map updates) and compiles against Eigen you use; or

produce a variant that uses k-means on canonical vectors (clusters instead of quantization) or LSH for approximate grouping if dataset is huge; or

benchmark a sample image and show how many candidates get pruned (I’ll need a sample image / matrix).


Which of those would you like next?
