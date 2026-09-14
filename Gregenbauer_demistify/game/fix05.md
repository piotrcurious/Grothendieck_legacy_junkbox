# EIGHTH LAYER: Two-Snapshot Morphing & State Machine Repairs (fix05.md)

This document specifies the exact repairs and improvements for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Correctness & Transition State Machine

1. **Direct `<chrono>` and `<algorithm>` Includes:**
   Explicitly include `<chrono>` and `<algorithm>` across all headers and implementation files.

2. **Dedicated `set_transition(double t)` API:**
   Replace direct state mutation in `cb_slider_transition()` with a dedicated setter:
   ```cpp
   void GameUI::set_transition(double t) {
       state.transition = std::clamp(t, 0.0, 1.0);
       if (state.transition >= 1.0) {
           commit_layer_transition();
       } else {
           mark_dirty_and_schedule();
       }
   }
   ```

3. **`choice_error_target` Synchronization:**
   Store `error_target_idx` (0: 1e-4, 1: 1e-8, 2: 1e-12, 3: 1e-15) directly in `CoreParameters` to synchronize `choice_error_target` deterministically without floating-point equality recovery.

4. **Sole Transition State-Machine API:**
   Make `begin_layer_transition()`, `cancel_layer_transition()`, `commit_layer_transition()`, and `set_transition()` the sole authoritative transition mutation API, always calling `normalize_state()` before updating or rendering.

5. **`error_target` & Non-Finite Validation:**
   In `normalize_state()`, validate `!std::isfinite(st.params.error_target)`, defaulting to `1e-8`. Check all parameters against `NaN`/`Inf`.

6. **Removal of Unused Headers:**
   Remove `<cstdio>` where unused.

---

## P1 — Two-Snapshot Morph Architecture

7. **Genuine Current $\to$ Target Snapshot Morphing:**
   Maintain three snapshots in `GameUI`:
   * `current_snapshot` (evaluated at `state.current_layer`)
   * `target_snapshot` (evaluated at `state.target_layer`)
   * `render_snapshot` (`GegenbauerCore::morph_snapshots(current_snapshot, target_snapshot, state.transition)`)

8. **Separate Mathematics Evaluation from Morph Animation:**
   Re-evaluate `current_snapshot` and `target_snapshot` ONLY when mathematical parameters change.
   During pure morph animation ($0 < T < 1$), `timer_morph_cb` updates `state.transition`, computes `render_snapshot = morph_snapshots(current_snapshot, target_snapshot, state.transition)`, and pushes `render_snapshot` to `GLCanvas` without running `core.evaluate()` at 60 Hz.
