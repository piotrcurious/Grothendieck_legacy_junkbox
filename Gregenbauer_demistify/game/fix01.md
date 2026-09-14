# EIGHTH LAYER: Architecture & Mathematical Repair Plan (fix01.md)

This document specifies the exact 22-point repair and improvement plan for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Fix Before Adding Functionality

### 1. Initialize and Publish One Canonical State
Make `RepresentationSnapshot` the single canonical post-evaluation result. `GLCanvas` and `GameUI` HUD both consume only `RepresentationSnapshot`.
```
GameState (Input) ---> GegenbauerCore::evaluate() ---> Immutable RepresentationSnapshot
                                                               ├── Telemetry HUD
                                                               └── GLCanvas Renderer
```

### 2. Remove `gl_canvas->set_game_state(state)`
Eliminate duplicate state consumers in the renderer so `GLCanvas` cannot silently diverge from evaluated mathematical telemetry. `GLCanvas` consumes only `gl_canvas->set_snapshot(snap)`.

### 3. Fix Morph-State Semantics
Centralize transition management (`set_transition(double t)`). When transition parameter $t$ reaches $1.0$, commit `current_layer = target_layer` explicitly.

### 4. Single Animation / Update Scheduler
Unify `mark_dirty_and_schedule()` and `timer_morph_cb()` into a single frame tick pipeline (`state mutation -> dirty flag -> one frame tick -> evaluate once -> render once`) to eliminate redundant evaluations.

### 5. Caching for Expensive Mathematics
Categorize computations into `EvaluationCost::CHEAP` vs `EvaluationCost::EXPENSIVE`. Cache Golub-Welsch $J_m$ eigensolver matrix decompositions and $R_J$ eigenpair residuals to prevent UI slider dragging lag for large $n$ / $m$.

---

## P0 — Make Layer VIII Mathematically Faithful

### 6. Mathematically Real Router AI
The Router AI generates candidate choices, executes feasibility filtering, cost minimization, and applies effective layer/backend:
```
Requested State (state.target_layer, state.backend)
    ---> Router AI Feasibility Filter
            ---> Cost Minimization
                    ---> Effective State (snap.effective_layer, snap.effective_backend)
```

### 7. Separate Requested State from Effective State
Explicitly distinguish `requested_layer` / `requested_backend` from `effective_layer` / `effective_backend` in `RepresentationSnapshot`.

### 8. Explicit Router Reasoning Records
Report candidate rejection reasons explicitly in `RouterDecision`:
* `candidate rejected: domain invalid`
* `candidate rejected: B_K > \epsilon_target`
* `candidate rejected: conditioning \kappa > threshold`
* `selected: Layer IV (Jacobi) + FLOAT64 (Cost = 2.0)`

---

## P1 — Numerical Controls & Certification Semantics

### 9. Endpoint-Sensitive Angle Control
Replace linear slider bounds with non-linear / endpoint-sensitive mapping using `std::numbers::pi` over $(0, \pi)$ to provide high resolution near singular poles $\theta \to 0^+$ ($z_+$) and $\theta \to \pi^-$ ($z_-$).

### 10. Parameter Model Derived Quantities
Model $\lambda = (d-2)/2$, $N = n + \lambda$, $x = \cos\theta$, $z_+ = N\theta$, $z_- = N(\pi-\theta)$ as exact derived quantities in `CoreParameters` / `RepresentationSnapshot`.

### 11. Governed Layer V–VIII Parameters
Expose target error $\epsilon_{\text{target}}$, asymptotic order $K$, Jacobi truncation size $m$, and overlap thresholds.

### 12. 5-Tier Certification Status & Reason Strings
Provide detailed failure reasons for each axis:
```
ALGEBRAIC:   [PASS / FAIL] - Reason string
ARITHMETIC:  [PASS / FAIL] - Reason string
ANALYTIC:    [PASS / FAIL] - Reason string
NUMERICAL:   [PASS / FAIL] - Reason string
```

### 13. Explicit Distinction of Error Concepts
Distinguish:
1. Structural Residual ($R_{\text{rec}}, R_{\text{ODE}}, R_{\text{Schr}}$)
2. Forward Error Estimate ($E_{\text{fwd}}$)
3. Conditioning ($\kappa = 1/\sin\theta$)
4. Backend Discrepancy ($E_{\text{backend}}$)

### 14. Exact Layer VIII Fields
Include explicit fields in `RepresentationSnapshot`:
* `structural_scale`
* `certification_floor`
* `analytic_bound`
* `forward_error_bound`
* `conditioning_ok`
* `domain_valid`

---

## P1 — UI Correctness & Structured Telemetry

### 15. Morph Info Context
In `cb_btn_info()`, display info for both Current Layer and Target Layer when transition $T > 0$.

### 16. Guarded Widget Synchronization
Guard UI widget value updates (`choice_layer`, `choice_backend`) to prevent programmatic re-entrancy loops.

### 17. Formatted Local String Telemetry
Format telemetry text into structured string buffers and set via `text_buffer->text(str.c_str())`.

### 18. Responsive Window Constraints
Centralize UI configuration constants, responsive column positions, and set `main_win->size_range(900, 650)`.

---

## P2 — Game-Specific & Visual Improvements

### 19. Standardized `LayerFrame` Abstraction
Construct `LayerFrame` containing geometry vertices, curve arrays, markers, and annotations for visual interpolation between layers.

### 20. Live Exact / Reference Curve Overlays
In Layers V & VI, render live curve overlays:
* Exact $\phi_{\text{exact}}(\theta)$
* North Bessel $F_{\text{north}}$
* South Bessel $F_{\text{south}}$
* WKB Wave $F_{\text{int}}$
* Composite Wave $F_{\text{comp}}$
* Analytic Error Envelope $B_K$

### 21. Real Backend Properties in Layer VII
Display precision bits, dynamic range, exactness class, operation cost, admissibility, and reconstruction bounds for all 7 backends.

### 22. Detailed Jacobi Spectral Properties in Layer IV
In Layer IV, display truncation size $m$, coupling coefficients $\alpha_k$, eigenvalues $x_k \in (-1, 1)$, eigenvector amplitudes $|v_{k,1}|^2$, and Gauss-Gegenbauer quadrature weights $w_k$.
