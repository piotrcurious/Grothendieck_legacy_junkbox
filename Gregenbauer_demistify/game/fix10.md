# EIGHTH LAYER: Snapshot Cache & State Invariant Repairs (fix10.md)

This document specifies the exact P0, P1, and P2 repairs for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Cache & API Correctness

1. **Fully Functional `get_or_evaluate_snapshot()`:**
   Implement `get_or_evaluate_snapshot` as the single authoritative cache access path:
   * Key hit on `current_key` -> return `current_snapshot`
   * Key hit on `target_key` -> return `target_snapshot`
   * Key miss -> evaluate via `core.evaluate(eval_state, eval_layer)` -> store in appropriate cache -> return snapshot.

2. **`commit_layer_transition()` Target Validity Fallback:**
   Enforce target snapshot validity on commit:
   ```cpp
   if (!target_valid) {
       target_snapshot = core.evaluate(state, state.target_layer);
       target_key = SnapshotKey{state.params.d, state.params.n, state.params.theta,
                                state.params.asymptotic_K, state.params.jacobi_m,
                                state.params.error_target_idx, state.backend,
                                state.target_layer, state.auto_router};
       target_valid = true;
   }
   current_snapshot = target_snapshot;
   current_key = target_key;
   current_valid = true;
   ```

3. **Guaranteed Target Snapshot Validity Before Manual $T=1$ Commit:**
   `set_transition(1.0)` ensures `target_snapshot` is valid before executing `commit_layer_transition()`.

4. **Documented `morph_snapshots()` Invariants:**
   `morph_snapshots()` explicitly assigns `res.transition = t`, `res.current_layer = snap1.effective_layer`, and `res.target_layer = snap2.effective_layer`.

---

## P1 — Telemetry & UI Responsiveness

5. **Current Diagnostics Labeling During Morphs:**
   During morphs ($0 < T < 1.0$), HUD explicitly labels diagnostic sections:
   `[Diagnostics: CURRENT Representation (Target Telemetry Available on Commit)]`

6. **Requested vs. Effective Router Telemetry:**
   Display both requested and effective layer/backend when AUTO routing is active:
   `Requested: [Layer I / FLOAT64] -> Effective: [Layer V / FLOAT64]`

---

## P2 — Cleanup & Geometry Constraints

7. **Array & Enum Size Constants:**
   Replace magic numbers with `kErrorTargets.size() - 1`, `kNumLayers - 1`, `kNumBackends - 1`.

8. **Minimum Window Height Constraint:**
   Set `main_win->size_range(900, 720)` and clamp canvas height to prevent control overlap on short windows.
