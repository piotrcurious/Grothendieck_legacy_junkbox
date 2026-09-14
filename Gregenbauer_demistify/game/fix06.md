# EIGHTH LAYER: Morph Semantic & Transition State Machine Repairs (fix06.md)

This document details the final correctness and performance repairs for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Correctness & Invariants

1. **Direct `<cmath>` Includes:**
   Explicitly include `<cmath>` across all headers and source files for `std::isfinite`.

2. **Same-Layer Transition Invariant:**
   In `begin_layer_transition(target)`:
   ```cpp
   if (target == state.current_layer) {
       cancel_layer_transition();
       state.target_layer = target;
       state.transition = 1.0;
       mark_dirty_and_schedule();
       return;
   }
   ```

3. **Manual Transition Cancellation:**
   When the user manually drags the morph slider (`cb_slider_transition`), `set_transition(t)` cancels any running animation timer (`cancel_layer_transition()`) to prevent `timer_morph_cb` from overwriting state.

4. **Canonical Error Target Index:**
   `error_target_idx` is the sole canonical source of truth for `error_target`. `normalize_state()` updates `st.params.error_target` directly from `targets[st.params.error_target_idx]`.

---

## P1 — Morph Architecture & Evaluation Optimization

5. **Strict `morph_snapshots()` Contract:**
   Interpolates continuous visual/renderable fields ($\phi$, forward error, residuals) while discretely selecting categorical metadata (certification flags, reasons, effective backend/layer names) at $T \ge 0.5$.

6. **Skip Duplicate Core Evaluation:**
   In `publish_snapshot()`, if `state.current_layer == state.target_layer`, skip evaluating `target_snapshot` and assign `target_snapshot = current_snapshot`.

7. **Pure Visual Transition Setters:**
   `set_transition()` updates visual transition state without re-evaluating core mathematics.
