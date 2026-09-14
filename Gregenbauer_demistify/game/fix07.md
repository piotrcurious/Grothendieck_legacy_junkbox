# EIGHTH LAYER: Snapshot Telemetry Authority & State Machine Refinement (fix07.md)

This document specifies the final architectural refinement and performance optimizations for **THE EIGHTH LAYER: The Harmonic Representation Engine** in `Gregenbauer_demistify/game/`.

---

## P0 — Correctness & Snapshot Semantics

1. **Strict `morph_snapshots()` Contract:**
   `morph_snapshots()` interpolates ONLY continuous visual rendering fields ($\phi$, forward error, residuals for GL drawing) and selects discrete, authoritative certification/residual telemetry directly from `snap1` (for $T < 0.5$) or `snap2` (for $T \ge 0.5$). Visual morphing never creates "half-certified" telemetry.

2. **Authoritative HUD Telemetry:**
   Telemetry display reads authoritative snapshot telemetry rather than interpolated metadata during morphing transitions.

3. **Explicit Evaluation Contract:**
   `GegenbauerCore::evaluate(state)` computes `RepresentationSnapshot` using explicit `target_layer` as representation selector.

---

## P1 — Performance & State Invariants

4. **Mathematical Snapshot Caching:**
   Cache evaluated `current_snapshot` and `target_snapshot` using a complete parameter key:
   `(d, n, theta, asymptotic_K, jacobi_m, error_target, backend, layer)`.
   If mathematical parameters do not change, avoid re-evaluating core mathematics.

5. **`advance_transition(double dt)` API:**
   Implement `advance_transition(dt)` as the single authoritative path for time-based animation updates:
   ```cpp
   void GameUI::advance_transition(double dt) {
       constexpr double duration = 0.8;
       state.transition += dt / duration;
       if (state.transition >= 1.0) {
           commit_layer_transition();
       } else {
           render_snapshot = GegenbauerCore::morph_snapshots(current_snapshot, target_snapshot, state.transition);
           gl_canvas->set_snapshot(render_snapshot);
           slider_transition->value(state.transition);
       }
   }
   ```

6. **Immediate Widget Synchronization:**
   `begin_layer_transition()` calls `sync_widgets_from_state()` immediately after setting `state.target_layer`.

7. **Static Pure `normalize_state()`:**
   Make `normalize_state(GameState&)` a static pure helper function.

---

## P2 — Cleanup & Constants

8. **Shared Constants:**
   Define `constexpr std::array<double, 4> kErrorTargets = {1e-4, 1e-8, 1e-12, 1e-15};`, `kNumLayers = 8`, `kNumBackends = 7`.
