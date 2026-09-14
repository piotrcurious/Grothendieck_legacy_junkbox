# EIGHTH LAYER: Explicit Evaluation Layer & Morph Rebase Plan (fix09.md)

This document details the exact architectural repairs for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Correctness & Expanded SnapshotKey

1. **Expanded `SnapshotKey`:**
   Expand `SnapshotKey` to include all inputs affecting `core.evaluate()`:
   * `d`, `n`, `theta`, `asymptotic_K`, `jacobi_m`, `error_target_idx`
   * `backend`
   * `evaluation_layer` (explicit, rather than proxy target_layer)
   * `auto_router`

2. **Explicit Evaluation Layer API:**
   Extend core evaluation signature to accept an explicit evaluation layer:
   ```cpp
   RepresentationSnapshot GegenbauerCore::evaluate(const GameState& state, LayerType evaluation_layer) const;
   ```

3. **Immediate Snapshot Evaluation Before Morph Timer:**
   `begin_layer_transition()` evaluates/caches fresh `current_snapshot` and `target_snapshot` immediately BEFORE starting `timer_morph_cb`.

4. **Morph Rebase on Parameter Change:**
   `apply_state_change()` cancels running morph animation (`cancel_layer_transition()`) when mathematical/backend parameters change to prevent animating across stale snapshots.

---

## P1 — Cache Model & Animation Lifecycle

5. **`get_or_evaluate_snapshot()` Helper:**
   Centralize snapshot caching:
   ```cpp
   RepresentationSnapshot GameUI::get_or_evaluate_snapshot(const SnapshotKey& key,
                                                           const GameState& eval_state,
                                                           LayerType eval_layer) {
       if (current_valid && current_key == key) return current_snapshot;
       if (target_valid && target_key == key) return target_snapshot;
       return core.evaluate(eval_state, eval_layer);
   }
   ```

6. **Immediate Snapshot Collapse on Commit:**
   In `commit_layer_transition()`, collapse `current_snapshot = target_snapshot` and `state.current_layer = state.target_layer` immediately.

---

## P2 — Telemetry & HUD

7. **Backend Telemetry:**
   Display `Backend: CURRENT -> TARGET` in HUD telemetry during morph transitions.
